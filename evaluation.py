"""
src/evaluation.py
Generation quality evaluation + classifier evaluation + K-Fold.

FID extractor options (in order of quality for this task)
──────────────────────────────────────────────────────────
1. HybridCNNTransformer (best)
2. Domain-adapted InceptionV3 (good)
3. Standard ImageNet InceptionV3 (for paper comparability)

Additional metrics
──────────────────
- KID   : Kernel Inception Distance (polynomial MMD, reliable n≥10)
- MS-SSIM: Multi-Scale Structural Similarity
- LPIPS  : Learned Perceptual Image Patch Similarity   ← Gap 1
- IS     : Inception Score (quality × diversity)        ← Gap 9
- P&R    : Precision & Recall manifold (Kynkäänniemi)  ← Gap 10
- TSNE/UMAP: feature-space alignment visualisation     ← Gap 7

Classifier evaluation extensions
─────────────────────────────────
- AUC-ROC (per-class OvR + macro/micro)                ← Gap 2
- ECE    : Expected Calibration Error (scalar)          ← Gap 6
- Sensitivity / Specificity per class                   ← Gap 13
- Statistical significance: Wilcoxon + McNemar          ← Gap 4
- LaTeX table export                                    ← Gap 15
- Data-efficiency curve                                 ← Gap 19
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.linalg import sqrtm
from scipy import stats as scipy_stats          # Gap 4
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_curve, auc,                             # Gap 2
    average_precision_score,                    # Gap 24
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import NearestNeighbors  # Gap 10
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd

from configs.config import (
    CLF_BATCH_SIZE, CKPT_DIR, IMG_SIZE, KFOLD_SPLITS, MIN_RELIABLE_SAMPLES,
    NUM_CLASSES, RARE_CLASSES, RANDOM_SEED, RESULTS_DIR,
    TRAIN_CSV, VAL_CSV, TEST_CSV, SYNTH_CSV,
    IMAGE_ROOT_DIR, DATA_DIR,
)
from src.dataset import GastroVisionDataset
from src.losses import FocalLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractors for FID
# ─────────────────────────────────────────────────────────────────────────────

def _make_transform(size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _extract_features_generic(df: pd.DataFrame, root_dir: Path,
                               model: nn.Module, transform: T.Compose,
                               hook_list: list, device: torch.device,
                               desc: str = "") -> np.ndarray:
    """Generic feature extraction loop used by all extractor types."""
    feats = []
    n     = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            hook_list.clear()
            with torch.no_grad():
                _ = model(tensor)
            if hook_list:
                feats.append(hook_list[0].flatten())
        except Exception as e:
            if i < 3:
                print(f"    Warning: {row['image_path']}: {e}")
        if (i + 1) % 200 == 0:
            print(f"    {desc}: {i+1}/{n}")
    return np.array(feats) if feats else None


def build_hybrid_extractor(ckpt_path: Path = None, device: torch.device = DEVICE):
    """Uses trained HybridCNNTransformer's fusion layer (1536-dim) as FID space."""
    from src.models import HybridCNNTransformer

    if ckpt_path is None:
        ckpt_path = CKPT_DIR / "sota_hybrid_cnn_transformer.pt"
        if not ckpt_path.exists():
            ckpt_path = CKPT_DIR / "sota_hybrid_cnn_transformer_aug.pt"

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"HybridCNNTransformer checkpoint not found: {ckpt_path}\n"
            f"Train it first with: python scripts/run_train.py "
            f"--models hybrid_cnn_transformer"
        )

    model = HybridCNNTransformer(num_classes=NUM_CLASSES).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  HybridCNNTransformer loaded from {ckpt_path}")

    hook_list = []
    def hook(module, inp, out):
        hook_list.append(out.detach().cpu().numpy())

    handle    = model.head[0].register_forward_hook(hook)
    transform = _make_transform(IMG_SIZE)

    def extract(df: pd.DataFrame, root_dir: Path, desc: str = "") -> np.ndarray:
        return _extract_features_generic(
            df, root_dir, model, transform, hook_list, device, desc=desc
        )

    return model, handle, extract, "HybridCNNTransformer (1536-dim)"


def build_inception_extractor(device: torch.device = DEVICE,
                               domain_adapt: bool = True,
                               train_csv: str = TRAIN_CSV,
                               fine_tune_epochs: int = 5):
    """InceptionV3 feature extractor (avgpool, 2048-dim)."""
    from torchvision.models import inception_v3

    inception = inception_v3(
        pretrained=True, aux_logits=True, transform_input=False
    ).to(device)

    if domain_adapt:
        inception.fc           = nn.Linear(2048, NUM_CLASSES).to(device)
        inception.AuxLogits.fc = nn.Linear(768, NUM_CLASSES).to(device)

        ds     = GastroVisionDataset(train_csv, split="train", mode="classifier")
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
        opt    = torch.optim.Adam(inception.parameters(), lr=1e-4)
        crit   = FocalLoss(gamma=2.0)

        inception.train()
        for ep in range(fine_tune_epochs):
            loss_sum = 0
            for xb, yb in loader:
                xb  = F.interpolate(xb.to(device), size=(299, 299))
                yb  = yb.to(device)
                out = inception(xb)
                loss = (crit(out[0], yb) + 0.4 * crit(out[1], yb)
                        if isinstance(out, tuple) else crit(out, yb))
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()
            print(f"  InceptionV3 fine-tune epoch {ep+1}/{fine_tune_epochs} "
                  f"loss={loss_sum/len(loader):.4f}")

        inception.fc        = nn.Identity()
        inception.AuxLogits = None
        label = "Domain-Adapted InceptionV3 (2048-dim)"
    else:
        inception.fc        = nn.Identity()
        inception.AuxLogits = None
        label = "Standard ImageNet InceptionV3 (2048-dim)"

    inception.eval()

    hook_list = []
    def hook(module, inp, out):
        hook_list.append(out.detach().flatten(1).cpu().numpy())
    handle    = inception.avgpool.register_forward_hook(hook)
    transform = _make_transform(299)

    def extract(df: pd.DataFrame, root_dir: Path, desc: str = "") -> np.ndarray:
        return _extract_features_generic(
            df, root_dir, inception, transform, hook_list, device, desc=desc
        )

    return inception, handle, extract, label


# ─────────────────────────────────────────────────────────────────────────────
# Distance / distribution metrics
# ─────────────────────────────────────────────────────────────────────────────

def frechet_distance(feat_real: np.ndarray, feat_synth: np.ndarray) -> float:
    """Standard Fréchet Distance. Unreliable when n_real < ~50."""
    mu_r, mu_s = feat_real.mean(0), feat_synth.mean(0)
    sigma_r    = np.cov(feat_real, rowvar=False)
    sigma_s    = np.cov(feat_synth, rowvar=False)

    eps = 1e-6 * np.eye(sigma_r.shape[0])
    sigma_r += eps
    sigma_s += eps

    diff    = mu_r - mu_s
    covmean = sqrtm(sigma_r @ sigma_s)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff @ diff
        + np.trace(sigma_r) + np.trace(sigma_s)
        - 2.0 * np.trace(covmean)
    )


def kernel_inception_distance(feat_real: np.ndarray,
                               feat_synth: np.ndarray,
                               degree: int = 3,
                               gamma: float = None,
                               coef: float = 1.0) -> float:
    """Kernel Inception Distance (KID) — polynomial kernel MMD ×1000."""
    from sklearn.metrics.pairwise import polynomial_kernel

    n     = min(len(feat_real), len(feat_synth), 500)
    rng   = np.random.default_rng(RANDOM_SEED)
    r_idx = rng.choice(len(feat_real),  n, replace=False)
    s_idx = rng.choice(len(feat_synth), n, replace=False)
    r     = feat_real[r_idx]
    s     = feat_synth[s_idx]

    gamma = gamma or 1.0 / feat_real.shape[1]
    k_rr  = polynomial_kernel(r, r, degree=degree, gamma=gamma, coef0=coef)
    k_ss  = polynomial_kernel(s, s, degree=degree, gamma=gamma, coef0=coef)
    k_rs  = polynomial_kernel(r, s, degree=degree, gamma=gamma, coef0=coef)

    np.fill_diagonal(k_rr, 0)
    np.fill_diagonal(k_ss, 0)
    mmd = (k_rr.sum() / (n * (n - 1))
           + k_ss.sum() / (n * (n - 1))
           - 2.0 * k_rs.mean())
    return float(mmd * 1000)


def ms_ssim_score(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                  n_pairs: int = 30, device: torch.device = DEVICE) -> dict:
    """Multi-Scale SSIM per rare class. Returns cls → {mean, std, n}."""
    try:
        from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    except ImportError:
        print("  torchmetrics not available — skipping MS-SSIM")
        return {}

    ms_ssim_fn = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(device)
    to_tensor  = T.ToTensor()
    results    = {}

    for cls in RARE_CLASSES:
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]
        scores    = []

        for _, rrow in real_cls.iterrows():
            try:
                real_t = to_tensor(
                    Image.open(IMAGE_ROOT_DIR / rrow["image_path"]).convert("RGB")
                ).unsqueeze(0).to(device)

                sample = synth_cls.sample(
                    min(n_pairs, len(synth_cls)), random_state=RANDOM_SEED
                )
                for _, srow in sample.iterrows():
                    try:
                        synth_t = to_tensor(
                            Image.open(DATA_DIR / srow["image_path"]).convert("RGB")
                        ).unsqueeze(0).to(device)
                        scores.append(ms_ssim_fn(real_t, synth_t).item())
                    except Exception:
                        continue
            except Exception:
                continue

        if scores:
            results[cls] = {
                "mean": float(np.mean(scores)),
                "std":  float(np.std(scores)),
                "n":    len(scores),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gap 1 — LPIPS (Learned Perceptual Image Patch Similarity)
# ─────────────────────────────────────────────────────────────────────────────

def lpips_score(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                rare_classes=None, n_pairs: int = 30,
                device: torch.device = DEVICE) -> dict:
    """
    LPIPS (Zhang et al. CVPR 2018) per rare class using AlexNet backbone.
    Lower is better (perceptual distance). Complements FID/KID.
    Returns cls → {mean, std, n}.
    """
    try:
        import lpips as lpips_lib
    except ImportError:
        print("  lpips not installed — run: pip install lpips")
        return {}

    if rare_classes is None:
        rare_classes = RARE_CLASSES

    loss_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
    # LPIPS expects inputs in [-1, 1]
    to_tensor = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    results = {}

    for cls in rare_classes:
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]
        if len(real_cls) == 0 or len(synth_cls) == 0:
            continue
        scores = []

        for _, rrow in real_cls.iterrows():
            try:
                real_t = to_tensor(
                    Image.open(IMAGE_ROOT_DIR / rrow["image_path"]).convert("RGB")
                ).unsqueeze(0).to(device)

                sample = synth_cls.sample(
                    min(n_pairs, len(synth_cls)), random_state=RANDOM_SEED
                )
                for _, srow in sample.iterrows():
                    try:
                        synth_t = to_tensor(
                            Image.open(DATA_DIR / srow["image_path"]).convert("RGB")
                        ).unsqueeze(0).to(device)
                        with torch.no_grad():
                            d = loss_fn(real_t, synth_t).item()
                        scores.append(d)
                    except Exception:
                        continue
            except Exception:
                continue

        if scores:
            results[cls] = {
                "mean": float(np.mean(scores)),
                "std":  float(np.std(scores)),
                "n":    len(scores),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gap 9 — Inception Score
# ─────────────────────────────────────────────────────────────────────────────

def inception_score(synth_df: pd.DataFrame, root_dir: Path,
                    n_splits: int = 10,
                    device: torch.device = DEVICE):
    """
    Inception Score (Salimans et al. 2016).
    Measures both quality (high per-image confidence) and diversity
    (high entropy of marginal p(y)). Higher IS = better.
    Returns (mean_IS, std_IS).
    """
    from torchvision.models import inception_v3

    inc = inception_v3(pretrained=True, transform_input=True).to(device)
    # Replace fc with identity; we want logits from the pretrained head
    inc.AuxLogits = None
    inc.eval()

    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
    ])

    preds = []
    for _, row in synth_df.iterrows():
        try:
            img = Image.open(root_dir / row["image_path"]).convert("RGB")
            t   = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = inc(t)
                if isinstance(logits, tuple):
                    logits = logits[0]
                p = F.softmax(logits, dim=1)
            preds.append(p.cpu().numpy())
        except Exception:
            continue

    if not preds:
        del inc; torch.cuda.empty_cache()
        return None, None

    preds = np.concatenate(preds, axis=0)     # (N, 1000)
    n     = len(preds)
    split_size = max(1, n // n_splits)

    scores = []
    for i in range(n_splits):
        part  = preds[i * split_size: (i + 1) * split_size]
        if len(part) == 0:
            continue
        p_y   = part.mean(axis=0, keepdims=True)  # marginal
        kl    = part * (np.log(part + 1e-8) - np.log(p_y + 1e-8))
        scores.append(float(np.exp(kl.sum(axis=1).mean())))

    del inc; torch.cuda.empty_cache()
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Gap 10 — Precision & Recall for generative models (Kynkäänniemi et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────

def precision_recall_manifold(feat_real: np.ndarray,
                               feat_synth: np.ndarray,
                               k: int = 3) -> dict:
    """
    Improved Precision & Recall (Kynkäänniemi et al., NeurIPS 2019).
    - Precision: fraction of synth points inside real manifold (quality)
    - Recall:    fraction of real points covered by synth manifold (diversity)
    Both in [0, 1]; higher is better. F1 = harmonic mean.
    """
    def _knn_radii(features: np.ndarray, k: int) -> np.ndarray:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(features)
        dist, _ = nbrs.kneighbors(features)
        return dist[:, -1]   # distance to k-th neighbour (radius of manifold ball)

    n_real  = min(len(feat_real),  5000)
    n_synth = min(len(feat_synth), 5000)
    rng = np.random.default_rng(RANDOM_SEED)
    r   = feat_real[rng.choice(len(feat_real),   n_real,  replace=False)]
    s   = feat_synth[rng.choice(len(feat_synth), n_synth, replace=False)]

    radii_r = _knn_radii(r, k)
    radii_s = _knn_radii(s, k)

    # Precision: is each synth point within any real ball?
    nbrs_r = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(r)
    dist_s_to_r, _ = nbrs_r.kneighbors(s)
    precision = float((dist_s_to_r[:, 0] <= radii_r[nbrs_r.kneighbors(s)[1][:, 0]]).mean())

    # Recall: is each real point within any synth ball?
    nbrs_s = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(s)
    dist_r_to_s, _ = nbrs_s.kneighbors(r)
    recall = float((dist_r_to_s[:, 0] <= radii_s[nbrs_s.kneighbors(r)[1][:, 0]]).mean())

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": float(f1)}


# ─────────────────────────────────────────────────────────────────────────────
# Gap 7 — T-SNE / UMAP feature-space visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_feature_space(real_df: pd.DataFrame, synth_df: pd.DataFrame,
                             rare_classes, extract_fn,
                             label_to_name: dict,
                             save_dir: Path,
                             device: torch.device = DEVICE) -> None:
    """
    Projects real and synthetic rare-class feature vectors into 2D (UMAP or
    t-SNE). Saves side-by-side scatter: real | synthetic, coloured by class.
    """
    save_dir = Path(save_dir)
    real_sub  = real_df[real_df["label"].isin(rare_classes)]
    synth_sub = synth_df[synth_df["label"].isin(rare_classes)]

    print("  Extracting features for TSNE/UMAP...")
    feat_r = extract_fn(real_sub,  IMAGE_ROOT_DIR, desc="UMAP real")
    feat_s = extract_fn(synth_sub, DATA_DIR,       desc="UMAP synth")

    if feat_r is None or feat_s is None:
        print("  Feature extraction failed — skipping UMAP/TSNE visualisation")
        return

    labels_r = real_sub["label"].values[:len(feat_r)]
    labels_s = synth_sub["label"].values[:len(feat_s)]

    all_feats  = np.concatenate([feat_r, feat_s], axis=0)
    all_labels = np.concatenate([labels_r, labels_s])
    all_src    = np.array(["real"] * len(feat_r) + ["synth"] * len(feat_s))

    # UMAP preferred; fall back to t-SNE
    try:
        import umap
        print("  Running UMAP...")
        reducer   = umap.UMAP(n_components=2, random_state=RANDOM_SEED,
                               n_neighbors=15, min_dist=0.1, metric="euclidean")
        embedding = reducer.fit_transform(all_feats)
        method    = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(all_feats) // 8))
        print(f"  Running t-SNE (perplexity={perp})...")
        embedding = TSNE(n_components=2, random_state=RANDOM_SEED,
                         perplexity=perp, n_iter=1000).fit_transform(all_feats)
        method = "t-SNE"

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cmap   = plt.cm.get_cmap("tab10", len(rare_classes))
        c_map  = {cls: cmap(i) for i, cls in enumerate(rare_classes)}
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        for ax, src in zip(axes, ["real", "synth"]):
            mask = all_src == src
            for cls in rare_classes:
                cmask = mask & (all_labels == cls)
                if cmask.sum() == 0:
                    continue
                ax.scatter(embedding[cmask, 0], embedding[cmask, 1],
                           c=[c_map[cls]],
                           label=f"{cls}: {label_to_name.get(cls, '')[:14]}",
                           alpha=0.65, s=18, edgecolors="none")
            ax.set_title(f"{src.capitalize()} images — {method}", fontsize=12)
            ax.legend(fontsize=6, markerscale=2, ncol=2)
            ax.set_xlabel(f"{method}-1"); ax.set_ylabel(f"{method}-2")

        plt.suptitle(f"Rare-class Feature Space: Real vs Synthetic ({method})", fontsize=13)
        plt.tight_layout()
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"feature_space_{method.lower()}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {method} plot saved → {out}")
    except Exception as e:
        print(f"  Warning: could not save {method} plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main quality evaluation — all metrics, all extractor options
# ─────────────────────────────────────────────────────────────────────────────

def compute_generation_quality(
    real_csv            = TRAIN_CSV,
    synth_csv           = SYNTH_CSV,
    rare_classes        = RARE_CLASSES,
    extractor           = "hybrid",
    hybrid_ckpt         = None,
    n_ssim_pairs        = 30,
    n_lpips_pairs       = 30,      # Gap 1
    domain_adapt_epochs = 5,
    run_is              = True,    # Gap 9
    run_pr_manifold     = True,    # Gap 10
    run_tsne            = True,    # Gap 7
):
    """
    Full generation quality evaluation:
      - FID / KID  (chosen feature space)
      - MS-SSIM    (domain-agnostic structural)
      - LPIPS      (perceptual patch similarity)         ← Gap 1
      - IS         (quality × diversity)                 ← Gap 9
      - P&R manifold (precision/recall)                  ← Gap 10
      - TSNE/UMAP  (feature-space visualisation)         ← Gap 7
    """
    real_df  = pd.read_csv(real_csv)
    synth_df = pd.read_csv(synth_csv)
    label_to_name = (
        dict(zip(real_df["label"].astype(int), real_df["class_name"]))
        if "class_name" in real_df.columns else {}
    )

    print(f"\n{'='*70}")
    print(f"Building feature extractor: {extractor}")
    print(f"{'='*70}")

    if extractor == "hybrid":
        try:
            model, handle, extract_fn, ext_label = build_hybrid_extractor(
                ckpt_path=hybrid_ckpt, device=DEVICE
            )
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            print("  Falling back to domain-adapted InceptionV3")
            extractor = "inception_domain"

    if extractor == "inception_domain":
        model, handle, extract_fn, ext_label = build_inception_extractor(
            device=DEVICE, domain_adapt=True,
            fine_tune_epochs=domain_adapt_epochs,
        )
    elif extractor == "inception_imagenet":
        model, handle, extract_fn, ext_label = build_inception_extractor(
            device=DEVICE, domain_adapt=False,
        )

    print(f"\nUsing: {ext_label}")

    # ── Per-class FID and KID ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Per-Class FID / KID")
    print(f"{'='*70}")
    print(f"  {'Class':<6} {'Name':<38} {'N_real':<8} {'N_synth':<8} "
          f"{'FID':>8}  {'KID×1000':>10}  {'Reliable?'}")
    print(f"  {'-'*88}")

    fid_per_class = {}
    kid_per_class = {}

    for cls in rare_classes:
        cls_name  = label_to_name.get(cls, f"class_{cls}")
        real_cls  = real_df[real_df["label"] == cls]
        synth_cls = synth_df[synth_df["label"] == cls]

        if len(real_cls) < 2 or len(synth_cls) < 2:
            print(f"  {cls:<6} {cls_name:<38} too few samples")
            fid_per_class[cls] = None; kid_per_class[cls] = None
            continue

        feat_r = extract_fn(real_cls,  IMAGE_ROOT_DIR, desc=f"real cls {cls}")
        feat_s = extract_fn(synth_cls, DATA_DIR,       desc=f"synth cls {cls}")

        if feat_r is None or feat_s is None or len(feat_r) < 2 or len(feat_s) < 2:
            fid_per_class[cls] = None; kid_per_class[cls] = None
            continue

        fid_score = None
        if len(feat_r) >= 20:
            try:
                fid_score = frechet_distance(feat_r, feat_s)
            except Exception as e:
                print(f"    FID failed for class {cls}: {e}")

        kid_score = None
        if len(feat_r) >= 10:
            try:
                kid_score = kernel_inception_distance(feat_r, feat_s)
            except Exception as e:
                print(f"    KID failed for class {cls}: {e}")

        fid_per_class[cls] = fid_score
        kid_per_class[cls] = kid_score

        fid_str      = f"{fid_score:.1f}" if fid_score is not None else "N/A*"
        kid_str      = f"{kid_score:.3f}" if kid_score is not None else "N/A*"
        reliable_str = "✅" if len(feat_r) >= 20 else "⚠ n<20"

        print(f"  {cls:<6} {cls_name:<38} {len(feat_r):<8} {len(feat_s):<8} "
              f"{fid_str:>8}  {kid_str:>10}  {reliable_str}")

    # ── Pooled FID / KID ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"POOLED FID / KID  (all rare classes combined — report this)")
    print(f"{'='*70}")

    real_pooled  = real_df[real_df["label"].isin(rare_classes)]
    synth_pooled = synth_df[synth_df["label"].isin(rare_classes)]
    feat_r_pool  = extract_fn(real_pooled,  IMAGE_ROOT_DIR, desc="real pooled")
    feat_s_pool  = extract_fn(synth_pooled, DATA_DIR,       desc="synth pooled")

    pooled_fid, pooled_kid = None, None
    pooled_pr = {}
    if feat_r_pool is not None and feat_s_pool is not None:
        pooled_fid = frechet_distance(feat_r_pool, feat_s_pool)
        pooled_kid = kernel_inception_distance(feat_r_pool, feat_s_pool)
        fid_q      = _fid_quality(pooled_fid)
        kid_q      = _kid_quality(pooled_kid)
        print(f"\n  FID     = {pooled_fid:.2f}  ({fid_q})")
        print(f"  KID×1000= {pooled_kid:.3f}  ({kid_q})")
        print(f"  n_real  = {len(feat_r_pool)}, n_synth = {len(feat_s_pool)}")

        # Gap 10 — P&R manifold on pooled features
        if run_pr_manifold:
            print("\n  Computing Precision & Recall manifold...")
            try:
                pooled_pr = precision_recall_manifold(feat_r_pool, feat_s_pool)
                print(f"  P&R manifold → Precision={pooled_pr['precision']:.4f}  "
                      f"Recall={pooled_pr['recall']:.4f}  F1={pooled_pr['f1']:.4f}")
            except Exception as e:
                print(f"  P&R manifold failed: {e}")

        # Gap 7 — TSNE/UMAP
        if run_tsne:
            print("\n  Generating TSNE/UMAP feature-space plot...")
            try:
                visualise_feature_space(
                    real_df, synth_df, rare_classes, extract_fn,
                    label_to_name, RESULTS_DIR
                )
            except Exception as e:
                print(f"  TSNE/UMAP failed: {e}")

    # ── Clean up extractor ───────────────────────────────────────────────────
    handle.remove()
    del model
    torch.cuda.empty_cache()

    # ── MS-SSIM ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MS-SSIM  (structural similarity, domain-agnostic)")
    print(f"{'='*70}")
    msssim_results = ms_ssim_score(real_df, synth_df, n_pairs=n_ssim_pairs)
    valid_ssim     = [v["mean"] for v in msssim_results.values() if v is not None]
    pooled_msssim  = float(np.mean(valid_ssim)) if valid_ssim else None

    print(f"\n  {'Class':<6} {'Name':<38} {'MS-SSIM':>10}  {'Quality'}")
    print(f"  {'-'*62}")
    for cls in rare_classes:
        cls_name = label_to_name.get(cls, f"class_{cls}")
        res      = msssim_results.get(cls)
        if res:
            q = "good" if res["mean"] > 0.4 else "acceptable" if res["mean"] > 0.2 else "poor"
            print(f"  {cls:<6} {cls_name:<38} {res['mean']:>8.4f}±{res['std']:.4f}  {q}")
        else:
            print(f"  {cls:<6} {cls_name:<38} {'—':>10}")
    if pooled_msssim:
        print(f"\n  Pooled MS-SSIM = {pooled_msssim:.4f}")

    # ── Gap 1 — LPIPS ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"LPIPS  (perceptual patch similarity — lower is better)")
    print(f"{'='*70}")
    lpips_results = lpips_score(real_df, synth_df, rare_classes,
                                n_pairs=n_lpips_pairs)
    valid_lpips   = [v["mean"] for v in lpips_results.values() if v is not None]
    pooled_lpips  = float(np.mean(valid_lpips)) if valid_lpips else None

    print(f"\n  {'Class':<6} {'Name':<38} {'LPIPS↓':>10}  {'Quality'}")
    print(f"  {'-'*62}")
    for cls in rare_classes:
        cls_name = label_to_name.get(cls, f"class_{cls}")
        res      = lpips_results.get(cls)
        if res:
            q = "excellent" if res["mean"] < 0.2 else "good" if res["mean"] < 0.4 else "poor"
            print(f"  {cls:<6} {cls_name:<38} {res['mean']:>8.4f}±{res['std']:.4f}  {q}")
        else:
            print(f"  {cls:<6} {cls_name:<38} {'—':>10}")
    if pooled_lpips:
        print(f"\n  Pooled LPIPS = {pooled_lpips:.4f}")

    # ── Gap 9 — Inception Score ──────────────────────────────────────────────
    is_mean, is_std = None, None
    if run_is:
        print(f"\n{'='*70}")
        print(f"Inception Score  (quality × diversity — higher is better)")
        print(f"{'='*70}")
        try:
            is_mean, is_std = inception_score(synth_pooled, DATA_DIR)
            if is_mean is not None:
                print(f"  IS = {is_mean:.3f} ± {is_std:.3f}  (n={len(synth_pooled)})")
            else:
                print("  IS: insufficient samples")
        except Exception as e:
            print(f"  IS failed: {e}")

    # ── Combined summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"COMBINED QUALITY SUMMARY  [{ext_label}]")
    print(f"{'='*70}")
    print(f"  {'Class':<6} {'Name':<28} {'FID↓':>8}  {'KID↓':>10}  "
          f"{'MS-SSIM↑':>9}  {'LPIPS↓':>8}  Verdict")
    print(f"  {'-'*82}")

    for cls in rare_classes:
        cls_name  = label_to_name.get(cls, f"class_{cls}")
        fid_val   = fid_per_class.get(cls)
        kid_val   = kid_per_class.get(cls)
        ssim_val  = msssim_results.get(cls)
        lpips_val = lpips_results.get(cls)

        fid_str   = f"{fid_val:.1f}"             if fid_val   is not None else "N/A"
        kid_str   = f"{kid_val:.3f}"             if kid_val   is not None else "N/A"
        ssim_str  = f"{ssim_val['mean']:.3f}"    if ssim_val  is not None else "N/A"
        lpips_str = f"{lpips_val['mean']:.3f}"   if lpips_val is not None else "N/A"

        if kid_val is not None and ssim_val is not None:
            verdict = ("✅ good"      if kid_val < 2.0 and ssim_val["mean"] > 0.2
                       else "⚠ mixed" if kid_val < 5.0 or ssim_val["mean"] > 0.15
                       else "❌ poor")
        elif kid_val is not None:
            verdict = "✅" if kid_val < 2.0 else "⚠" if kid_val < 5.0 else "❌"
        else:
            verdict = "—"

        print(f"  {cls:<6} {cls_name:<28} {fid_str:>8}  {kid_str:>10}  "
              f"{ssim_str:>9}  {lpips_str:>8}  {verdict}")

    fid_s   = f"{pooled_fid:.2f}"   if pooled_fid   else "N/A"
    kid_s   = f"{pooled_kid:.3f}"   if pooled_kid   else "N/A"
    ssim_s  = f"{pooled_msssim:.4f}" if pooled_msssim else "N/A"
    lpips_s = f"{pooled_lpips:.4f}" if pooled_lpips  else "N/A"
    is_s    = f"{is_mean:.3f}±{is_std:.3f}" if is_mean else "N/A"
    pr_s    = (f"P={pooled_pr.get('precision',0):.3f} R={pooled_pr.get('recall',0):.3f}"
               if pooled_pr else "N/A")
    print(f"\n  POOLED → FID={fid_s}  KID×1000={kid_s}  MS-SSIM={ssim_s}  "
          f"LPIPS={lpips_s}  IS={is_s}  P&R={pr_s}")

    _save_quality_plots(rare_classes, fid_per_class, kid_per_class,
                        msssim_results, lpips_results,
                        label_to_name, ext_label)

    # ── Gap 27 — Diversity metrics ───────────────────────────────────────────
    diversity_results: dict = {}
    try:
        diversity_results = compute_diversity_metrics(
            synth_df, rare_classes, data_dir=DATA_DIR, n_pairs=50,
        )
    except Exception as e:
        print(f"  Diversity metrics failed: {e}")

    # ── Build return dict ────────────────────────────────────────────────────
    results = {}
    for cls in rare_classes:
        ssim_val  = msssim_results.get(cls)
        lpips_val = lpips_results.get(cls)
        div_val   = diversity_results.get(cls)
        results[cls] = {
            "fid":         fid_per_class.get(cls),
            "kid":         kid_per_class.get(cls),
            "msssim":      ssim_val["mean"]    if ssim_val  else None,
            "lpips":       lpips_val["mean"]   if lpips_val else None,
            "div_lpips_mean": div_val["lpips_mean"] if div_val else None,  # Gap 27
            "div_lpips_std":  div_val["lpips_std"]  if div_val else None,
            "div_l2_mean":    div_val["l2_mean"]    if div_val else None,
            "div_l2_std":     div_val["l2_std"]     if div_val else None,
            "n_real":      len(real_df[real_df["label"] == cls]),
            "n_synth":     len(synth_df[synth_df["label"] == cls]),
            "name":        label_to_name.get(cls, f"class_{cls}"),
        }
    div_pool = diversity_results.get("pooled", {})
    results["pooled"] = {
        "fid":          pooled_fid,
        "kid":          pooled_kid,
        "msssim":       pooled_msssim,
        "lpips":        pooled_lpips,
        "is_mean":      is_mean,
        "is_std":       is_std,
        "pr_precision": pooled_pr.get("precision"),
        "pr_recall":    pooled_pr.get("recall"),
        "pr_f1":        pooled_pr.get("f1"),
        "div_lpips_mean": div_pool.get("lpips_mean"),   # Gap 27
        "div_lpips_std":  div_pool.get("lpips_std"),
        "div_l2_mean":    div_pool.get("l2_mean"),
        "div_l2_std":     div_pool.get("l2_std"),
        "n_real":       len(feat_r_pool) if feat_r_pool is not None else 0,
        "n_synth":      len(feat_s_pool) if feat_s_pool is not None else 0,
        "extractor":    ext_label,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gap 27 — Diversity metrics: intra-class LPIPS variance + pairwise L2
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity_metrics(
    synth_df: pd.DataFrame,
    rare_classes,
    data_dir: Path = None,
    n_pairs: int = 50,
    device: str = None,
) -> dict:
    """
    Gap 27: Measures within-class diversity of synthetic images.

    For each rare class in synth_df:
      - Intra-class LPIPS variance: std of pairwise LPIPS distances sampled
        from n_pairs random pairs within the class (high std = varied perceptual
        distances; low std = homogeneous / mode-collapsed output).
      - Mean pairwise L2 in InceptionV3 feature space: captures semantic
        spread independently of low-level pixel variance.

    Returns {cls: {"lpips_mean", "lpips_std", "l2_mean", "l2_std", "n_pairs"}}
    + {"pooled": same keys averaged over rare classes}.
    """
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as tvm
    from PIL import Image

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    # ── Build lightweight LPIPS-like network (VGG-based) ────────────────────
    try:
        import lpips as _lpips_lib
        lpips_fn = _lpips_lib.LPIPS(net="vgg").to(_device)
        lpips_fn.eval()
        has_lpips = True
    except ImportError:
        has_lpips = False
        print("  LPIPS not installed; skipping LPIPS diversity (pip install lpips)")

    # ── InceptionV3 feature extractor ───────────────────────────────────────
    inception = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1)
    inception.fc = nn.Identity()
    inception = inception.to(_device).eval()
    _hook_feats: list = []

    def _fwd_hook(module, inp, out):  # noqa: ARG001
        _hook_feats.append(out.detach().cpu())

    _handle = inception.avgpool.register_forward_hook(_fwd_hook)

    img_tf = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    def _load_img(path_str: str) -> torch.Tensor:
        path = data_dir / path_str if not Path(path_str).is_absolute() else Path(path_str)
        img  = Image.open(path).convert("RGB")
        return img_tf(img)

    def _extract_feats(paths) -> np.ndarray:
        feats = []
        with torch.no_grad():
            for p in paths:
                try:
                    x = _load_img(p).unsqueeze(0).to(_device)
                    _hook_feats.clear()
                    inception(x)
                    if _hook_feats:
                        feats.append(_hook_feats[0].squeeze().numpy())
                except Exception:
                    pass
        return np.stack(feats) if feats else None

    rng = np.random.default_rng(42)
    results: dict = {}

    for cls in rare_classes:
        cls_df  = synth_df[synth_df["label"] == cls]
        paths   = cls_df["filename"].tolist() if "filename" in cls_df.columns else []
        if len(paths) < 4:
            results[cls] = None
            continue

        # Sample up to n_pairs random pairs
        idx_a = rng.integers(0, len(paths), size=n_pairs)
        idx_b = rng.integers(0, len(paths), size=n_pairs)
        same  = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % len(paths)

        # LPIPS pairwise
        lpips_vals: list = []
        if has_lpips:
            tf_01 = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            with torch.no_grad():
                for ia, ib in zip(idx_a, idx_b):
                    try:
                        pa = data_dir / paths[ia] if not Path(paths[ia]).is_absolute() else Path(paths[ia])
                        pb = data_dir / paths[ib] if not Path(paths[ib]).is_absolute() else Path(paths[ib])
                        ta = tf_01(Image.open(pa).convert("RGB")).unsqueeze(0).to(_device)
                        tb = tf_01(Image.open(pb).convert("RGB")).unsqueeze(0).to(_device)
                        # LPIPS expects [-1,1]
                        ta = ta * 2 - 1; tb = tb * 2 - 1
                        d  = lpips_fn(ta, tb).item()
                        lpips_vals.append(d)
                    except Exception:
                        pass

        # InceptionV3 L2 pairwise
        l2_vals: list = []
        sample_paths = list(set(paths[i] for i in list(idx_a) + list(idx_b)))[:min(200, len(paths))]
        feats = _extract_feats(sample_paths)
        if feats is not None and len(feats) >= 2:
            for ia, ib in zip(
                rng.integers(0, len(feats), size=n_pairs),
                rng.integers(0, len(feats), size=n_pairs),
            ):
                if ia != ib:
                    l2_vals.append(float(np.linalg.norm(feats[ia] - feats[ib])))

        results[cls] = {
            "lpips_mean": float(np.mean(lpips_vals)) if lpips_vals else None,
            "lpips_std":  float(np.std(lpips_vals))  if lpips_vals else None,
            "l2_mean":    float(np.mean(l2_vals))    if l2_vals    else None,
            "l2_std":     float(np.std(l2_vals))     if l2_vals    else None,
            "n_pairs":    max(len(lpips_vals), len(l2_vals)),
            "n_synth":    len(paths),
        }

    _handle.remove()
    del inception
    torch.cuda.empty_cache()

    # Pooled averages
    valid = [v for v in results.values() if v is not None]
    pooled: dict = {}
    for key in ("lpips_mean", "lpips_std", "l2_mean", "l2_std"):
        vals = [v[key] for v in valid if v.get(key) is not None]
        pooled[key] = float(np.mean(vals)) if vals else None
    results["pooled"] = pooled

    print(f"\n  Diversity metrics (Gap 27) — intra-class LPIPS variance + L2 spread:")
    print(f"  {'Cls':>4}  {'N_synth':>7}  {'LPIPS mean':>11}  {'LPIPS std':>10}  "
          f"{'L2 mean':>9}  {'L2 std':>8}")
    print(f"  {'-'*60}")
    for cls in rare_classes:
        r = results.get(cls)
        if r is None:
            print(f"  {cls:>4}  {'too few':>7}")
            continue
        lm  = f"{r['lpips_mean']:.4f}" if r["lpips_mean"] is not None else "—"
        ls  = f"{r['lpips_std']:.4f}"  if r["lpips_std"]  is not None else "—"
        l2m = f"{r['l2_mean']:.2f}"    if r["l2_mean"]    is not None else "—"
        l2s = f"{r['l2_std']:.2f}"     if r["l2_std"]     is not None else "—"
        print(f"  {cls:>4}  {r['n_synth']:>7}  {lm:>11}  {ls:>10}  {l2m:>9}  {l2s:>8}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Real vs Generated image comparison (side-by-side grids per rare class)
# ─────────────────────────────────────────────────────────────────────────────

def plot_real_vs_synthetic(
    real_df: pd.DataFrame,
    synth_sd_df: pd.DataFrame,
    rare_classes,
    image_root: Path = None,
    data_dir:   Path = None,
    save_dir:   Path = None,
    n_cols: int = 5,
    synth_gan_df: pd.DataFrame = None,
    label_to_name: dict = None,
) -> None:
    """
    Saves per-rare-class side-by-side image grids:
        [Real] | [SD Synthetic] | [GAN Synthetic (if provided)]
    And a summary mosaic PNG across all rare classes.

    Files written to save_dir:
        real_vs_synth_{cls}.png      ← per-class grid
        real_vs_synth_mosaic.png     ← summary mosaic
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        print(f"  plot_real_vs_synthetic: missing dependency {e}")
        return

    _image_root = Path(image_root) if image_root is not None else IMAGE_ROOT_DIR
    _data_dir   = Path(data_dir)   if data_dir   is not None else DATA_DIR
    _save_dir   = Path(save_dir)   if save_dir   is not None else RESULTS_DIR / "real_vs_synth"
    _save_dir.mkdir(parents=True, exist_ok=True)
    _label_to_name = label_to_name or {}
    has_gan     = synth_gan_df is not None and len(synth_gan_df) > 0
    n_groups    = 3 if has_gan else 2          # real / SD / GAN
    group_labels = ["Real", "SD Synthetic", "GAN Synthetic"] if has_gan else ["Real", "SD Synthetic"]
    rng = np.random.default_rng(0)

    def _load(path_str: str, root: Path) -> np.ndarray | None:
        p = root / path_str if not Path(path_str).is_absolute() else Path(path_str)
        try:
            img = Image.open(p).convert("RGB").resize((128, 128))
            return np.array(img)
        except Exception:
            return None

    mosaic_row_imgs: list = []   # one representative thumbnail per class per group

    for cls in rare_classes:
        cls_name   = _label_to_name.get(cls, f"class_{cls}")
        real_paths = real_df[real_df["label"] == cls]["filename"].tolist() \
            if "filename" in real_df.columns else []
        sd_paths   = synth_sd_df[synth_sd_df["label"] == cls]["filename"].tolist() \
            if "filename" in synth_sd_df.columns else []
        gan_paths  = (synth_gan_df[synth_gan_df["label"] == cls]["filename"].tolist()
                      if has_gan and "filename" in synth_gan_df.columns else [])

        if not real_paths and not sd_paths:
            continue

        # Sample up to n_cols images from each group
        real_sample = [real_paths[i] for i in rng.choice(
            len(real_paths), size=min(n_cols, len(real_paths)), replace=False)]
        sd_sample   = [sd_paths[i]   for i in rng.choice(
            len(sd_paths),   size=min(n_cols, len(sd_paths)),   replace=False)] if sd_paths   else []
        gan_sample  = [gan_paths[i]  for i in rng.choice(
            len(gan_paths),  size=min(n_cols, len(gan_paths)),  replace=False)] if gan_paths  else []

        rows = []
        for sample, root, grp_label in [
            (real_sample, _image_root, "Real"),
            (sd_sample,   _data_dir,   "SD Synthetic"),
            (gan_sample,  _data_dir,   "GAN Synthetic"),
        ]:
            if not sample:
                continue
            imgs = [_load(p, root) for p in sample]
            imgs = [im for im in imgs if im is not None]
            if imgs:
                rows.append((grp_label, imgs))

        if not rows:
            continue

        n_row    = len(rows)
        n_col    = max(len(r[1]) for r in rows)
        fig, axes = plt.subplots(n_row, n_col,
                                  figsize=(n_col * 1.6 + 0.5, n_row * 1.8 + 0.8))
        if n_row == 1:
            axes = axes[np.newaxis, :]
        if n_col == 1:
            axes = axes[:, np.newaxis]

        for ri, (grp_label, imgs) in enumerate(rows):
            for ci in range(n_col):
                ax = axes[ri, ci]
                ax.axis("off")
                if ci < len(imgs):
                    ax.imshow(imgs[ci])
                if ci == 0:
                    ax.set_ylabel(grp_label, fontsize=9, rotation=90, labelpad=4)

        fig.suptitle(f"Cls {cls}: {cls_name}  (Real vs Synthetic)", fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = _save_dir / f"real_vs_synth_{cls}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")

        # Collect one representative image per group for mosaic
        mosaic_row: list = []
        for grp_label, imgs in rows:
            mosaic_row.append((cls, grp_label, imgs[0]))
        mosaic_row_imgs.append(mosaic_row)

    # ── Summary mosaic ───────────────────────────────────────────────────────
    if not mosaic_row_imgs:
        return

    n_cls_valid = len(mosaic_row_imgs)
    fig, axes   = plt.subplots(n_cls_valid, n_groups,
                                figsize=(n_groups * 2.2, n_cls_valid * 2.0 + 0.8))
    if n_cls_valid == 1:
        axes = axes[np.newaxis, :]
    if n_groups == 1:
        axes = axes[:, np.newaxis]

    for ri, row in enumerate(mosaic_row_imgs):
        for ci, (cls, grp_label, img) in enumerate(row):
            ax = axes[ri, ci]
            ax.imshow(img)
            ax.axis("off")
            if ri == 0:
                ax.set_title(grp_label, fontsize=9)
            if ci == 0:
                cls_name = _label_to_name.get(cls, f"Cls {cls}")
                ax.set_ylabel(f"Cls {cls}\n{cls_name[:14]}", fontsize=8, rotation=90, labelpad=4)

    plt.suptitle("Real vs Synthetic — Summary Mosaic", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    mosaic_out = _save_dir / "real_vs_synth_mosaic.png"
    plt.savefig(mosaic_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved summary mosaic → {mosaic_out}")


def _fid_quality(score: float) -> str:
    if score is None:  return "—"
    if score < 50:     return "excellent"
    if score < 100:    return "good"
    if score < 200:    return "acceptable"
    return "poor"


def _kid_quality(score: float) -> str:
    if score is None:  return "—"
    if score < 0.5:    return "excellent"
    if score < 2.0:    return "good"
    if score < 5.0:    return "acceptable"
    return "poor"


def _save_quality_plots(rare_classes, fid_per_class, kid_per_class,
                        msssim_results, lpips_results,
                        label_to_name, ext_label):
    """Save bar-chart plots for KID, FID, MS-SSIM, and LPIPS."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        valid_cls = [c for c in rare_classes if kid_per_class.get(c) is not None]
        if not valid_cls:
            return

        cls_labels = [f"Cls {c}\n{label_to_name.get(c,'')[:12]}" for c in valid_cls]

        has_ssim  = any(msssim_results.get(c)  for c in valid_cls)
        has_lpips = any(lpips_results.get(c)   for c in valid_cls)
        n_plots   = 2 + has_ssim + has_lpips
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 6, 5))
        if n_plots == 1: axes = [axes]

        ax_idx = 0

        # KID
        kid_vals   = [kid_per_class[c] for c in valid_cls]
        kid_colors = ["#6acc65" if v < 0.5 else "#4878cf" if v < 2.0
                      else "#f0a500" if v < 5.0 else "#d65f5f" for v in kid_vals]
        bars = axes[ax_idx].bar(cls_labels, kid_vals, color=kid_colors, edgecolor="white")
        for bar, val in zip(bars, kid_vals):
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                               f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        axes[ax_idx].axhline(0.5, color="#6acc65", linestyle="--", alpha=0.6, label="Excellent")
        axes[ax_idx].axhline(2.0, color="#4878cf", linestyle="--", alpha=0.6, label="Good")
        axes[ax_idx].axhline(5.0, color="#f0a500", linestyle="--", alpha=0.6, label="Acceptable")
        axes[ax_idx].set_ylabel("KID × 1000 (lower is better)")
        axes[ax_idx].set_title(f"KID per Rare Class\n{ext_label}")
        axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(axis="y", alpha=0.3)
        ax_idx += 1

        # FID
        fid_vals   = [fid_per_class.get(c) for c in valid_cls]
        plot_fids  = [v if v is not None else 0 for v in fid_vals]
        fid_colors = ["#6acc65" if v and v < 50 else "#4878cf" if v and v < 100
                      else "#f0a500" if v and v < 200 else "#d65f5f" for v in fid_vals]
        bars = axes[ax_idx].bar(cls_labels, plot_fids, color=fid_colors, edgecolor="white")
        for bar, val in zip(bars, fid_vals):
            axes[ax_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                               f"{val:.1f}" if val else "N/A*", ha="center", va="bottom", fontsize=9)
        axes[ax_idx].axhline(50,  color="#6acc65", linestyle="--", alpha=0.6, label="Excellent (<50)")
        axes[ax_idx].axhline(100, color="#4878cf", linestyle="--", alpha=0.6, label="Good (<100)")
        axes[ax_idx].set_ylabel("FID (lower is better)  *N/A = n<20")
        axes[ax_idx].set_title(f"FID per Rare Class\n{ext_label}")
        axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(axis="y", alpha=0.3)
        ax_idx += 1

        # MS-SSIM
        if has_ssim:
            ssim_vals   = [msssim_results[c]["mean"] if msssim_results.get(c) else 0 for c in valid_cls]
            ssim_stds   = [msssim_results[c]["std"]  if msssim_results.get(c) else 0 for c in valid_cls]
            ssim_colors = ["#6acc65" if v > 0.4 else "#4878cf" if v > 0.2 else "#d65f5f"
                           for v in ssim_vals]
            axes[ax_idx].bar(cls_labels, ssim_vals, yerr=ssim_stds,
                              color=ssim_colors, edgecolor="white", capsize=4)
            axes[ax_idx].axhline(0.4, color="#6acc65", linestyle="--", alpha=0.6, label="Good (>0.4)")
            axes[ax_idx].axhline(0.2, color="#4878cf", linestyle="--", alpha=0.6, label="Acceptable (>0.2)")
            axes[ax_idx].set_ylabel("MS-SSIM (higher is better)")
            axes[ax_idx].set_title("MS-SSIM per Rare Class")
            axes[ax_idx].set_ylim(0, 1.0)
            axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(axis="y", alpha=0.3)
            ax_idx += 1

        # LPIPS  (Gap 1)
        if has_lpips:
            lp_vals  = [lpips_results[c]["mean"] if lpips_results.get(c) else 0 for c in valid_cls]
            lp_stds  = [lpips_results[c]["std"]  if lpips_results.get(c) else 0 for c in valid_cls]
            lp_cols  = ["#6acc65" if v < 0.2 else "#4878cf" if v < 0.4 else "#d65f5f"
                        for v in lp_vals]
            axes[ax_idx].bar(cls_labels, lp_vals, yerr=lp_stds,
                              color=lp_cols, edgecolor="white", capsize=4)
            axes[ax_idx].axhline(0.2, color="#6acc65", linestyle="--", alpha=0.6, label="Excellent (<0.2)")
            axes[ax_idx].axhline(0.4, color="#4878cf", linestyle="--", alpha=0.6, label="Good (<0.4)")
            axes[ax_idx].set_ylabel("LPIPS (lower is better)")
            axes[ax_idx].set_title("LPIPS per Rare Class")
            axes[ax_idx].legend(fontsize=8); axes[ax_idx].grid(axis="y", alpha=0.3)

        plt.suptitle(f"Generation Quality — Rare Classes\nExtractor: {ext_label}", fontsize=12)
        plt.tight_layout()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(RESULTS_DIR / "generation_quality.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Quality plot saved → {RESULTS_DIR / 'generation_quality.png'}")
    except Exception as e:
        print(f"  Warning: could not save quality plot: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Gap 2 — ROC-AUC / PR-AUC curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true: np.ndarray, all_probs: np.ndarray,
                    num_classes: int, rare_classes,
                    save_dir: Path, name: str) -> dict:
    """
    Compute and plot one-vs-rest ROC curves for all classes.
    Returns dict with per_class AUC, macro_auc, micro_auc, rare_macro_auc.
    """
    save_dir  = Path(save_dir)
    safe_name = name.replace(" ", "_").replace("/", "_")
    y_bin     = label_binarize(y_true, classes=list(range(num_classes)))

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(num_classes):
        if y_bin[:, i].sum() == 0:
            fpr_d[i], tpr_d[i], auc_d[i] = np.array([0, 1]), np.array([0, 1]), 0.5
            continue
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])

    # Macro average (interpolated)
    all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr  /= num_classes
    macro_auc  = float(auc(all_fpr, mean_tpr))

    # Micro average (flattened)
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    micro_auc  = float(auc(fpr_micro, tpr_micro))

    # Rare-class macro
    rare_valid   = [c for c in rare_classes if c < num_classes]
    rare_macro   = float(np.mean([auc_d[c] for c in rare_valid])) if rare_valid else 0.0

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: all classes (rare bold)
        for c in range(num_classes):
            lw    = 2.0 if c in rare_classes else 0.4
            alpha = 0.9 if c in rare_classes else 0.3
            lbl   = f"Cls {c} (AUC={auc_d[c]:.2f})" if c in rare_classes else None
            axes[0].plot(fpr_d[c], tpr_d[c], lw=lw, alpha=alpha, label=lbl)
        axes[0].plot(all_fpr, mean_tpr, "k--", lw=2.5,
                     label=f"Macro avg (AUC={macro_auc:.3f})")
        axes[0].plot([0,1],[0,1], "gray", lw=1, linestyle=":")
        axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].set_title(f"{name} — ROC (rare classes = thick)")
        axes[0].legend(fontsize=7, loc="lower right")

        # Right: rare classes only
        cmap = plt.cm.get_cmap("tab10", len(rare_valid))
        for idx, c in enumerate(rare_valid):
            axes[1].plot(fpr_d[c], tpr_d[c], lw=2, color=cmap(idx),
                         label=f"Cls {c} (AUC={auc_d[c]:.3f})")
        axes[1].plot([0,1],[0,1], "gray", lw=1, linestyle=":")
        axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
        axes[1].set_title(f"{name} — Rare-Class ROC (macro={rare_macro:.3f})")
        axes[1].legend(fontsize=8)

        plt.suptitle(f"{name}  macro-AUC={macro_auc:.4f}  micro-AUC={micro_auc:.4f}",
                     fontsize=12)
        plt.tight_layout()
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"roc_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ROC plot saved → {save_dir / f'roc_{safe_name}.png'}")
    except Exception as e:
        print(f"  Warning: could not save ROC plot: {e}")

    return {
        "per_class":      {c: float(v) for c, v in auc_d.items()},
        "macro_auc":      macro_auc,
        "micro_auc":      micro_auc,
        "rare_macro_auc": rare_macro,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gap 24 — PR-AUC (Average Precision) per class
# ─────────────────────────────────────────────────────────────────────────────

def compute_prauc(y_true: np.ndarray, all_probs: np.ndarray,
                  num_classes: int, rare_classes,
                  save_dir: Path = None, name: str = "") -> dict:
    """
    Gap 24: Computes Average Precision (AP = area under PR curve) per class
    using one-vs-rest. Also computes macro and rare-class macro PR-AUC.
    Saves a PR-curve plot when save_dir is provided.
    Returns {"per_class": {c: ap}, "macro_prauc", "rare_macro_prauc"}.
    """
    y_bin   = label_binarize(y_true, classes=list(range(num_classes)))
    ap_dict: dict = {}

    for c in range(num_classes):
        if y_bin.shape[1] <= c or y_bin[:, c].sum() == 0:
            ap_dict[c] = 0.0
            continue
        try:
            ap_dict[c] = float(average_precision_score(y_bin[:, c], all_probs[:, c]))
        except Exception:
            ap_dict[c] = 0.0

    macro_prauc      = float(np.mean(list(ap_dict.values())))
    rare_valid        = [c for c in rare_classes if c < num_classes]
    rare_macro_prauc  = float(np.mean([ap_dict[c] for c in rare_valid])) if rare_valid else 0.0

    print(f"  PR-AUC → macro={macro_prauc:.4f}  rare_macro={rare_macro_prauc:.4f}")

    # Optional PR-curve plot (rare classes only)
    if save_dir is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_curve

            save_dir  = Path(save_dir)
            safe_name = name.replace(" ", "_").replace("/", "_")
            cmap      = plt.cm.get_cmap("tab10", len(rare_valid))

            fig, ax = plt.subplots(figsize=(8, 6))
            for idx, c in enumerate(rare_valid):
                prec, rec, _ = precision_recall_curve(y_bin[:, c], all_probs[:, c])
                ax.plot(rec, prec, lw=2, color=cmap(idx),
                        label=f"Cls {c}  AP={ap_dict[c]:.3f}")
            # Macro iso-F1 curves (dashed background)
            for f_val in [0.2, 0.4, 0.6, 0.8]:
                x_iso = np.linspace(0.01, 1.0, 200)
                y_iso = f_val * x_iso / (2 * x_iso - f_val + 1e-8)
                y_iso = np.clip(y_iso, 0, 1)
                ax.plot(x_iso, y_iso, "--", color="gray", lw=0.8, alpha=0.5)
                ax.annotate(f"F1={f_val}", xy=(0.9, f_val * 0.9 / (2 * 0.9 - f_val + 1e-8)),
                            fontsize=7, color="gray")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"{name} — PR curves (rare classes)\n"
                         f"macro PR-AUC={macro_prauc:.4f}  rare macro={rare_macro_prauc:.4f}")
            ax.legend(fontsize=8, loc="lower left")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            save_dir.mkdir(parents=True, exist_ok=True)
            out = save_dir / f"prauc_{safe_name}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"  PR-AUC plot saved → {out}")
        except Exception as e:
            print(f"  Warning: could not save PR-AUC plot: {e}")

    return {
        "per_class":       {c: float(v) for c, v in ap_dict.items()},
        "macro_prauc":     macro_prauc,
        "rare_macro_prauc": rare_macro_prauc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gap 22 — Bootstrap confidence intervals for rare-class F1
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_rare_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                          rare_classes,
                          n_bootstrap: int = 1000,
                          alpha: float = 0.05,
                          seed: int = 42) -> dict:
    """
    Gap 22: Percentile bootstrap for 95% CI of rare-class F1.
    Resamples (y_true, y_pred) pairs n_bootstrap times with replacement.

    Returns {
      "per_class": {cls: {"mean", "ci_lower", "ci_upper", "std", "n_bootstrap"}},
      "pooled":    {"mean", "ci_lower", "ci_upper", "std"},
    }
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)
    num_classes = int(y_pred.max()) + 1
    ci_lo = alpha / 2 * 100          # e.g. 2.5
    ci_hi = (1 - alpha / 2) * 100    # e.g. 97.5

    per_class_samples: dict = {c: [] for c in rare_classes if c < num_classes}
    pooled_samples: list    = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]; yp_b = y_pred[idx]
        _, _, f1_b, _ = precision_recall_fscore_support(
            yt_b, yp_b,
            labels=list(range(num_classes)),
            average=None, zero_division=0,
        )
        rare_f1_vals = []
        for c in rare_classes:
            if c < len(f1_b):
                per_class_samples[c].append(float(f1_b[c]))
                rare_f1_vals.append(float(f1_b[c]))
        if rare_f1_vals:
            pooled_samples.append(float(np.mean(rare_f1_vals)))

    results: dict = {"per_class": {}, "pooled": {}}

    for c in rare_classes:
        if c not in per_class_samples or not per_class_samples[c]:
            continue
        s = np.array(per_class_samples[c])
        results["per_class"][c] = {
            "mean":        float(s.mean()),
            "ci_lower":    float(np.percentile(s, ci_lo)),
            "ci_upper":    float(np.percentile(s, ci_hi)),
            "std":         float(s.std()),
            "n_bootstrap": n_bootstrap,
        }

    if pooled_samples:
        ps = np.array(pooled_samples)
        results["pooled"] = {
            "mean":        float(ps.mean()),
            "ci_lower":    float(np.percentile(ps, ci_lo)),
            "ci_upper":    float(np.percentile(ps, ci_hi)),
            "std":         float(ps.std()),
            "n_bootstrap": n_bootstrap,
        }

    print(f"  Bootstrap CI (rare-class F1, n={n_bootstrap}): "
          f"pooled mean={results['pooled'].get('mean', 0):.4f}  "
          f"95% CI=[{results['pooled'].get('ci_lower', 0):.4f}, "
          f"{results['pooled'].get('ci_upper', 0):.4f}]")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Gap 6 — Expected Calibration Error (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ece(y_true: np.ndarray, y_pred: np.ndarray,
                 confidences: np.ndarray, n_bins: int = 15) -> float:
    """
    ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|
    Lower is better. Perfect calibration = 0.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc_b  = float((y_true[mask] == y_pred[mask]).mean())
        conf_b = float(confidences[mask].mean())
        ece   += mask.sum() / n * abs(acc_b - conf_b)
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation (extended: ECE + Sensitivity/Specificity + ROC-AUC)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(model_or_ensemble, name, val_loader,
                      num_classes=NUM_CLASSES, is_ensemble=False,
                      save_dir=RESULTS_DIR, min_reliable=MIN_RELIABLE_SAMPLES):
    """
    Full evaluation pipeline:
      - Confusion matrix (raw + normalised)
      - Precision / Recall / F1 per class
      - AUC-ROC per class + macro/micro                  ← Gap 2
      - ECE (Expected Calibration Error scalar)          ← Gap 6
      - Sensitivity + Specificity per class              ← Gap 13
      - Confidence distribution + reliability diagram
    Returns extended dict: acc, y_true, y_pred, all_probs, f1_scores,
                           auc_dict, ece, sensitivity, specificity.
    """
    y_true_list, y_pred_list, probs_list = [], [], []

    if is_ensemble:
        with torch.no_grad():
            for xb, yb in val_loader:
                preds, probs, _ = model_or_ensemble.predict_with_confidence(xb)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())
                probs_list.append(probs.cpu().numpy())
    else:
        model_or_ensemble.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb    = xb.to(DEVICE)
                probs = F.softmax(model_or_ensemble(xb), dim=1)
                preds = probs.argmax(dim=1)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())
                probs_list.append(probs.cpu().numpy())

    y_true    = np.concatenate(y_true_list)
    y_pred    = np.concatenate(y_pred_list)
    all_probs = np.concatenate(probs_list)
    acc       = accuracy_score(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"{name}  val_acc={acc:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    precision, recall, f1_scores, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)),
        average=None, zero_division=0
    )

    # ── Gap 2: ROC-AUC ──────────────────────────────────────────────────────
    auc_dict = {}
    try:
        auc_dict = plot_roc_curves(y_true, all_probs, num_classes,
                                   RARE_CLASSES, Path(save_dir), name)
        print(f"  AUC → macro={auc_dict['macro_auc']:.4f}  "
              f"micro={auc_dict['micro_auc']:.4f}  "
              f"rare_macro={auc_dict['rare_macro_auc']:.4f}")
    except Exception as e:
        print(f"  ROC-AUC failed: {e}")

    # ── Gap 6: ECE ──────────────────────────────────────────────────────────
    top_conf = all_probs.max(axis=1)
    correct  = (y_pred == y_true)
    ece      = _compute_ece(y_true, y_pred, top_conf)
    print(f"  ECE = {ece:.4f}  (lower is better; 0 = perfect calibration)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("/", "_")

        # Confusion matrix
        cm      = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        sns.heatmap(cm,      annot=True, fmt="d",    cmap="Blues", ax=axes[0], annot_kws={"size": 7})
        sns.heatmap(cm_norm, annot=True, fmt=".2f",  cmap="Blues", ax=axes[1], annot_kws={"size": 7})
        for ax, t in zip(axes, ["counts", "normalised"]):
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"{name} — CM ({t})")
        plt.suptitle(f"{name}  val_acc={acc:.4f}", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / f"cm_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # P/R/F1 bar chart
        x     = np.arange(num_classes)
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(14, num_classes * 2), 6))
        ax.bar(x - width, precision, width, label="Precision", color="#4878cf", alpha=0.85)
        ax.bar(x,         recall,    width, label="Recall",    color="#6acc65", alpha=0.85)
        ax.bar(x + width, f1_scores, width, label="F1",        color="#d65f5f", alpha=0.85)
        for cls in RARE_CLASSES:
            if cls < num_classes:
                ax.axvspan(cls - 0.45, cls + 0.45, alpha=0.08, color="yellow")
        ax.axhline(f1_scores.mean(), color="red", linestyle="--",
                   label=f"Mean F1={f1_scores.mean():.3f}")
        ax.set_xlabel("Class"); ax.set_ylabel("Score")
        ax.set_title(f"{name} — P/R/F1 per class  (yellow = rare)")
        ax.legend(); ax.set_ylim(-0.1, 1.15)
        plt.tight_layout()
        plt.savefig(save_dir / f"prf1_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Confidence + calibration (with ECE annotation)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(top_conf[correct],  bins=30, alpha=0.7, color="green", label="Correct")
        axes[0].hist(top_conf[~correct], bins=30, alpha=0.7, color="red",   label="Wrong")
        axes[0].set_xlabel("Confidence"); axes[0].set_title("Confidence Distribution")
        axes[0].legend()
        prob_true, prob_pred = calibration_curve(correct, top_conf, n_bins=15)
        axes[1].plot(prob_pred, prob_true, "s-", label=name)
        axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
        axes[1].fill_between(prob_pred,
                             np.maximum(prob_true, prob_pred),
                             np.minimum(prob_true, prob_pred),
                             alpha=0.15, color="red", label=f"ECE={ece:.4f}")
        axes[1].set_xlabel("Mean confidence"); axes[1].set_ylabel("Fraction correct")
        axes[1].set_title("Reliability Diagram"); axes[1].legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(save_dir / f"calibration_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # ── Gap 13: Sensitivity / Specificity table ──────────────────────────
        sensitivity  = np.diag(cm) / (cm.sum(axis=1) + 1e-8)          # = recall
        specificity  = np.zeros(num_classes)
        for c in range(num_classes):
            tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
            fp = cm[:, c].sum() - cm[c, c]
            specificity[c] = tn / (tn + fp + 1e-8)

        fig, ax = plt.subplots(figsize=(max(14, num_classes * 2), 5))
        ax.bar(x - width/2, sensitivity, width, label="Sensitivity (Recall)",
               color="#6acc65", alpha=0.85)
        ax.bar(x + width/2, specificity, width, label="Specificity",
               color="#4878cf", alpha=0.85)
        for cls in RARE_CLASSES:
            if cls < num_classes:
                ax.axvspan(cls - 0.45, cls + 0.45, alpha=0.08, color="yellow")
        ax.set_xlabel("Class"); ax.set_ylabel("Score")
        ax.set_title(f"{name} — Sensitivity & Specificity per class  (yellow = rare)")
        ax.legend(); ax.set_ylim(-0.05, 1.15)
        plt.tight_layout()
        plt.savefig(save_dir / f"sens_spec_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"  Warning: could not save evaluation plots: {e}")
        cm          = confusion_matrix(y_true, y_pred)
        sensitivity = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        specificity = np.zeros(num_classes)
        for c in range(num_classes):
            tn = cm.sum() - cm[c,:].sum() - cm[:,c].sum() + cm[c,c]
            fp = cm[:,c].sum() - cm[c,c]
            specificity[c] = tn / (tn + fp + 1e-8)

    # Print rare-class clinical summary
    print(f"\n  {'Cls':>4} {'Sensitivity':>13} {'Specificity':>13}  Rare?")
    print(f"  {'-'*45}")
    for c in range(num_classes):
        tag = " ★" if c in RARE_CLASSES else ""
        print(f"  {c:>4} {sensitivity[c]:>12.4f} {specificity[c]:>12.4f}{tag}")

    # ── Gap 24: PR-AUC ──────────────────────────────────────────────────────
    prauc_dict: dict = {}
    try:
        prauc_dict = compute_prauc(
            y_true, all_probs, num_classes,
            RARE_CLASSES, Path(save_dir), name,
        )
    except Exception as e:
        print(f"  PR-AUC failed: {e}")

    # ── Gap 22: Bootstrap CI for rare-class F1 ───────────────────────────────
    ci_dict: dict = {}
    try:
        ci_dict = bootstrap_rare_f1_ci(y_true, y_pred, RARE_CLASSES)
        # Pretty-print per-class CIs
        if ci_dict.get("per_class"):
            print(f"\n  Bootstrap 95% CI — rare-class F1:")
            print(f"  {'Cls':>4}  {'Mean':>8}  {'CI lower':>10}  {'CI upper':>10}")
            print(f"  {'-'*36}")
            for c, stats in sorted(ci_dict["per_class"].items()):
                print(f"  {c:>4}  {stats['mean']:>8.4f}  "
                      f"{stats['ci_lower']:>10.4f}  {stats['ci_upper']:>10.4f}")
    except Exception as e:
        print(f"  Bootstrap CI failed: {e}")

    return (acc, y_true, y_pred, all_probs, f1_scores,
            auc_dict, ece, sensitivity, specificity,
            prauc_dict, ci_dict)


# ─────────────────────────────────────────────────────────────────────────────
# K-Fold evaluation (extended to return raw per-fold F1 lists)
# ─────────────────────────────────────────────────────────────────────────────

def kfold_evaluate(model_name="swin", eval_csv=None,
                   rare_classes=RARE_CLASSES, k=KFOLD_SPLITS,
                   label="baseline"):
    """
    Stratified K-fold cross-validation.
    Returns (summary, raw_folds) where:
      summary[cls] = {"precision":(mean,std), "recall":(mean,std), "f1":(mean,std), "n":int}
      raw_folds[cls] = {"precision":[...k values], "recall":[...], "f1":[...]}
    raw_folds enables proper Wilcoxon signed-rank testing (Gap 4).
    """
    from sklearn.model_selection import StratifiedKFold
    from src.models import load_trained_baseline

    if eval_csv is None:
        train_df = pd.read_csv(TRAIN_CSV)
        val_df   = pd.read_csv(VAL_CSV)
        full_df  = pd.concat([train_df, val_df], ignore_index=True)
        full_eval_csv = DATA_DIR / "_kfold_full_eval.csv"
        full_df.to_csv(full_eval_csv, index=False)
        print(f"K-Fold: using train+val = {len(full_df)} samples")
    else:
        full_df       = pd.read_csv(eval_csv)
        full_eval_csv = Path(eval_csv)

    class_counts = full_df["label"].value_counts()
    foldable     = class_counts[class_counts >= k].index.tolist()
    unfoldable   = class_counts[class_counts <  k].index.tolist()
    print(f"  Foldable ({k}-fold): {sorted(foldable)}")
    print(f"  Unfoldable (n<{k}): {sorted(unfoldable)}")

    df_foldable = full_df[full_df["label"].isin(foldable)]
    model       = load_trained_baseline(model_name, augmented=(label == "augmented"))
    model.eval()

    skf          = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    fold_results = {cls: {"precision": [], "recall": [], "f1": []} for cls in foldable}

    for fold, (_, val_idx) in enumerate(skf.split(df_foldable, df_foldable["label"])):
        fold_val = df_foldable.iloc[val_idx]
        fold_csv = DATA_DIR / f"_kfold_fold_{fold}.csv"
        fold_val.to_csv(fold_csv, index=False)

        fold_ds  = GastroVisionDataset(fold_csv, split="val", mode="classifier")
        fold_ldr = DataLoader(fold_ds, batch_size=CLF_BATCH_SIZE,
                              shuffle=False, num_workers=2)

        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for xb, yb in fold_ldr:
                preds = model(xb.to(DEVICE)).argmax(dim=1)
                y_pred_list.append(preds.cpu().numpy())
                y_true_list.append(yb.numpy())

        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(NUM_CLASSES)),
            average=None, zero_division=0
        )
        for cls in foldable:
            fold_results[cls]["precision"].append(float(p[cls]))
            fold_results[cls]["recall"].append(float(r[cls]))
            fold_results[cls]["f1"].append(float(f[cls]))

        fold_csv.unlink(missing_ok=True)

    del model
    torch.cuda.empty_cache()
    if eval_csv is None:
        full_eval_csv.unlink(missing_ok=True)

    summary   = {}
    raw_folds = {}
    print(f"\nK-Fold Results — {label} (k={k}, model={model_name})")
    print(f"  {'Cls':>4} {'n':>5}  {'P mean±std':<18} {'R mean±std':<18} {'F1 mean±std':<18} {'Rare'}")
    print(f"  {'-'*75}")

    for cls in range(NUM_CLASSES):
        if cls not in fold_results:
            continue
        vals = fold_results[cls]
        p_m, p_s = float(np.mean(vals["precision"])), float(np.std(vals["precision"]))
        r_m, r_s = float(np.mean(vals["recall"])),    float(np.std(vals["recall"]))
        f_m, f_s = float(np.mean(vals["f1"])),        float(np.std(vals["f1"]))
        n        = int(class_counts.get(cls, 0))
        rare_tag = " ★" if cls in rare_classes else ""
        print(f"  {cls:>4} {n:>5}  {p_m:.3f}±{p_s:.3f}           "
              f"{r_m:.3f}±{r_s:.3f}           {f_m:.3f}±{f_s:.3f}{rare_tag}")
        summary[cls] = {
            "precision": (p_m, p_s),
            "recall":    (r_m, r_s),
            "f1":        (f_m, f_s),
            "n":         n,
        }
        raw_folds[cls] = vals   # lists of k raw values (for Wilcoxon)

    return summary, raw_folds


# ─────────────────────────────────────────────────────────────────────────────
# Gap 4 — Statistical significance: Wilcoxon signed-rank + McNemar
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_significance(raw_folds_s1: dict, raw_folds_s2: dict,
                                raw_folds_s3: dict,
                                rare_classes=RARE_CLASSES,
                                alpha: float = 0.05) -> dict:
    """
    Wilcoxon signed-rank tests on per-fold F1 scores.
    Compares: S2 vs S1, S3 vs S1, S3 vs S2 for each rare class.

    raw_folds_sX[cls]["f1"] must be a list of k floats (one per CV fold).
    Obtain from kfold_evaluate(...)[1].

    Returns nested dict: results[cls][comparison] = {stat, p_value, significant, effect_d}
    """
    comparisons = [
        ("S2 vs S1", raw_folds_s2, raw_folds_s1),
        ("S3 vs S1", raw_folds_s3, raw_folds_s1),
        ("S3 vs S2", raw_folds_s3, raw_folds_s2),
    ]

    print(f"\n{'='*70}")
    print(f"Statistical Significance (Wilcoxon signed-rank, α={alpha})")
    print(f"{'='*70}")
    print(f"  {'Class':<6} {'Comparison':<14} {'ΔF1 mean':>10} {'p-value':>10}  {'Sig?':>6}  Cohen d")
    print(f"  {'-'*60}")

    all_results: dict = {}

    for cls in rare_classes:
        all_results[cls] = {}
        for label, folds_a, folds_b in comparisons:
            if cls not in folds_a or cls not in folds_b:
                continue
            fa = np.array(folds_a[cls]["f1"])
            fb = np.array(folds_b[cls]["f1"])
            if len(fa) < 3 or len(fb) < 3:
                continue

            diff = fa - fb
            if (diff == 0).all():
                stat, p_val = 0.0, 1.0
            else:
                try:
                    stat, p_val = scipy_stats.wilcoxon(
                        fa, fb, alternative="two-sided", zero_method="wilcox"
                    )
                except Exception:
                    stat, p_val = 0.0, 1.0

            # Cohen's d for paired data
            d_mean = float(diff.mean())
            d_std  = float(diff.std() + 1e-8)
            cohend = d_mean / d_std

            sig = p_val < alpha
            all_results[cls][label] = {
                "stat":        float(stat),
                "p_value":     float(p_val),
                "significant": bool(sig),
                "delta_f1":    d_mean,
                "cohen_d":     float(cohend),
            }
            sig_str = "✅" if sig else "✗"
            print(f"  {cls:<6} {label:<14} {d_mean:>+10.4f} {p_val:>10.4f}  {sig_str:>6}  {cohend:+.3f}")

    return all_results


def mcnemar_test(y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray,
                 y_true:   np.ndarray) -> dict:
    """
    McNemar's test for pairwise model comparison.
    H0: models A and B make the same errors.
    Uses continuity correction. Returns p_value and contingency table.
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    a = int((correct_a  & correct_b).sum())   # both correct
    b = int((~correct_a & correct_b).sum())   # A wrong, B correct
    c = int((correct_a  & ~correct_b).sum())  # A correct, B wrong
    d = int((~correct_a & ~correct_b).sum())  # both wrong

    if b + c == 0:
        return {"p_value": 1.0, "a": a, "b": b, "c": c, "d": d,
                "significant": False, "note": "No discordant pairs"}

    # McNemar with continuity correction
    stat    = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "p_value":     p_value,
        "stat":        float(stat),
        "significant": p_value < 0.05,
        "a": a, "b": b, "c": c, "d": d,
        "note": f"A better: c={c}  B better: b={b}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gap 15 — LaTeX table output
# ─────────────────────────────────────────────────────────────────────────────

def results_to_latex(s1: dict, s2: dict, s3: dict,
                     out_path: Path,
                     rare_classes=RARE_CLASSES,
                     caption: str = "Per-strategy classification results.",
                     label:   str = "tab:results") -> str:
    """
    Converts eval JSON dicts (from evaluate_all / evaluate_on_test) into
    a LaTeX booktabs table. Writes to out_path and returns the string.

    Each dict: model_name → {acc, f1_mean, f1_rare, macro_auc (optional)}
    """
    rows = []
    strategies = [
        ("S1: Real only",      s1),
        ("S2: Heavy aug",      s2),
        ("S3: SD synthetic",   s3),
    ]

    for strat_name, data in strategies:
        if not data:
            continue
        for model, res in data.items():
            if model.startswith("_"):
                continue
            acc        = res.get("acc",          float("nan"))
            f1m        = res.get("f1_mean",       float("nan"))
            f1r        = res.get("f1_rare",       float("nan"))
            mauc       = res.get("macro_auc",     None)
            ece        = res.get("ece",           None)
            prauc      = res.get("rare_prauc",    None)   # Gap 24
            tag        = r"\textbf{" + model + r"}" if model == "ensemble" else model
            mauc_str   = f"{mauc:.4f}"  if mauc  is not None else "—"
            ece_str    = f"{ece:.4f}"   if ece   is not None else "—"
            prauc_str  = f"{prauc:.4f}" if prauc is not None else "—"
            rows.append(
                f"  {strat_name} & {tag} & {acc:.4f} & {f1m:.4f} & {f1r:.4f} "
                f"& {mauc_str} & {ece_str} & {prauc_str} \\\\"
            )
        rows.append(r"  \midrule")

    if rows and rows[-1].strip() == r"\midrule":
        rows.pop()   # remove trailing midrule

    header = "\n".join([
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + caption + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    Strategy & Model & Acc & Mean F1 & Rare F1 & Macro AUC & ECE & Rare PR-AUC \\",
        r"    \midrule",
    ])
    footer = "\n".join([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    table_rows = "\n".join(f"    {r}" for r in rows)
    latex      = header + "\n" + table_rows + "\n" + footer + "\n"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    print(f"  LaTeX table saved → {out_path}")
    return latex


# ─────────────────────────────────────────────────────────────────────────────
# Gap 19 — Data-efficiency curve (in evaluation.py: curve plotter)
# The experiment runner (subsampling + training loop) lives in pipeline.
# This function plots pre-computed efficiency results.
# ─────────────────────────────────────────────────────────────────────────────

def plot_data_efficiency_curve(results: dict, save_dir: Path,
                                title: str = "Data Efficiency") -> None:
    """
    Plot F1 (mean over rare classes) vs number of training samples per rare class.

    results: {n_samples (int) → {"f1_rare": float, "f1_mean": float, "acc": float}}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns      = sorted(results.keys())
        f1_rare = [results[n].get("f1_rare", 0.0) for n in ns]
        f1_mean = [results[n].get("f1_mean", 0.0) for n in ns]
        accs    = [results[n].get("acc",     0.0) for n in ns]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ns, f1_rare, "o-", color="#d65f5f", lw=2, label="Rare-class F1")
        ax.plot(ns, f1_mean, "s--", color="#4878cf", lw=2, label="Mean F1 (all)")
        ax.plot(ns, accs,    "^:",  color="#6acc65", lw=2, label="Accuracy")
        ax.set_xlabel("Training samples per rare class")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xscale("log")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / "data_efficiency_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Data-efficiency plot saved → {out}")
    except Exception as e:
        print(f"  Warning: could not save data-efficiency plot: {e}")

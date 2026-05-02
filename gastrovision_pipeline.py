#!/usr/bin/env python3
"""
gastrovision_pipeline.py
========================
GastroVision rare-class augmentation pipeline.
Gaps added in Phase 2:
  Gap  8  predict_with_confidence on ConfidenceEnsemble
  Gap  3  Grad-CAM grid visualisation
  Gap  4  Wilcoxon / McNemar (in evaluation.py)
  Gap  5  evaluate_on_test()
  Gap 11  MixUp / CutMix datasets + trainer
  Gap 12  plot_training_history()
  Gap 16  Adaptive synthetic ratio
  Gap 17  W&B / CSV experiment tracking (_log)
  Gap 18  DCGAN baseline (train_dcgan + generate_synthetic_gan)
  Gap 19  run_data_efficiency_experiment()
  Gap 20  ablate_lora_rank()
"""

import os, sys, gc, argparse, json, warnings, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_recall_fscore_support, roc_curve, auc,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import inception_v3
import timm
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import seaborn as sns

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed — hyperparameter tuning disabled.")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False
    print("Grad-CAM not available — XAI disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from diffusers import (
    StableDiffusionPipeline, DDPMScheduler,
    UNet2DConditionModel, AutoencoderKL,
)
from diffusers.optimization import get_scheduler as get_diffusers_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

warnings.filterwarnings("ignore")


# ==============================================================================
# SECTION 1 — Argument parsing
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GastroVision DDPM Augmentation Pipeline")

    # Paths
    p.add_argument("--data_dir",      default="/data")
    p.add_argument("--output_dir",    default="/output")
    p.add_argument("--image_root",    default="gastrovision_raw/Gastrovision")
    p.add_argument("--train_csv",     default="train.csv")
    p.add_argument("--val_csv",       default="val.csv")
    p.add_argument("--test_csv",      default="test.csv")
    p.add_argument("--aug_train_csv", default="train_aug.csv")
    p.add_argument("--synth_csv",     default="synthetic_train.csv")
    p.add_argument("--synth_dir",     default="synthetic")

    # Classifier
    p.add_argument("--img_size",         type=int,   default=224)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=1e-4)
    p.add_argument("--freeze_epochs",    type=int,   default=16)
    p.add_argument("--fine_tune_epochs", type=int,   default=24)
    p.add_argument("--gamma",            type=float, default=2.0)
    p.add_argument("--freeze_lr_mult",   type=float, default=10.0)

    # Diffusion / SD
    p.add_argument("--sd_model_id",        default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora_rank",          type=int,   default=32)
    p.add_argument("--lora_alpha",         type=int,   default=64)
    p.add_argument("--lora_dropout",       type=float, default=0.1)
    p.add_argument("--domain_adapt_steps", type=int,   default=15000)
    p.add_argument("--sd_batch_size",      type=int,   default=4)
    p.add_argument("--sd_grad_accum",      type=int,   default=4)
    p.add_argument("--sd_lr",             type=float, default=1e-4)
    p.add_argument("--ema_decay",          type=float, default=0.9999)
    p.add_argument("--ema_warmup_steps",   type=int,   default=100)
    p.add_argument("--samples_per_class",  type=int,   default=500)
    p.add_argument("--gen_steps",          type=int,   default=40)
    p.add_argument("--guidance_scale",     type=float, default=7.0)
    p.add_argument("--gen_batch_size",     type=int,   default=4)

    # Gap 16 — adaptive synthetic ratio
    p.add_argument("--synthetic_ratio", type=int, default=0,
                   help="If >0, generate synthetic_ratio x n_real images per rare class "
                        "(overrides samples_per_class). 0 = use samples_per_class.")

    # Gap 18 — GAN baseline
    p.add_argument("--gan_epochs",      type=int, default=200,
                   help="DCGAN training epochs per rare class.")
    p.add_argument("--gan_lr",          type=float, default=2e-4)
    p.add_argument("--gan_latent_dim",  type=int, default=128)
    p.add_argument("--gan_batch_size",  type=int, default=16)

    # Gap 20 — LoRA rank ablation
    p.add_argument("--lora_ablation_ranks", nargs="+", type=int,
                   default=[4, 8, 16, 32],
                   help="LoRA ranks to test in ablation study.")
    p.add_argument("--lora_ablation_steps", type=int, default=5000,
                   help="Training steps per rank in ablation (shorter than full run).")

    # Gap 19 — data efficiency
    p.add_argument("--efficiency_counts", nargs="+", type=int,
                   default=[5, 10, 20, 50],
                   help="Per-class sample counts for data-efficiency curve.")
    p.add_argument("--efficiency_model",  default="swin",
                   help="Model to use for data-efficiency experiment.")

    # Evaluation
    p.add_argument("--kfold_splits",         type=int, default=5)
    p.add_argument("--min_reliable_samples", type=int, default=10)
    p.add_argument("--seed",                 type=int, default=42)

    # Gap 17 — experiment tracking
    p.add_argument("--use_wandb", action="store_true",
                   help="Log metrics to Weights & Biases.")
    p.add_argument("--wandb_project", default="gastrovision",
                   help="W&B project name.")
    p.add_argument("--wandb_run",     default=None,
                   help="W&B run name (auto if not set).")

    # Execution modes
    p.add_argument("--skip_domain_adapt", action="store_true")
    p.add_argument("--skip_generation",   action="store_true")
    p.add_argument("--skip_training",     action="store_true")
    p.add_argument("--evaluate_only",     action="store_true")
    p.add_argument("--run_gan_baseline",  action="store_true",
                   help="Train DCGAN baseline and compare with SD (Gap 18).")
    p.add_argument("--run_lora_ablation", action="store_true",
                   help="Run LoRA rank ablation study (Gap 20).")
    p.add_argument("--run_data_efficiency", action="store_true",
                   help="Run data-efficiency curve experiment (Gap 19).")
    p.add_argument("--run_gradcam",       action="store_true",
                   help="Generate Grad-CAM visualisation grids (Gap 3).")

    # Gap 21 — multi-seed repeated runs
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Seeds for repeated runs. E.g. --seeds 42 123 456 789 101112. "
                        "All metrics reported as mean±std across seeds.")

    # Gap 23 — loss type ablation
    p.add_argument("--loss_type", default="focal",
                   choices=["focal", "weighted_ce", "balanced_softmax"],
                   help="Classifier loss function (focal | weighted_ce | balanced_softmax).")

    # Gap 25 — CV as default
    p.add_argument("--single_split", action="store_true",
                   help="Use fixed train/val split instead of 5-fold × cv_repeats CV.")
    p.add_argument("--cv_repeats", type=int, default=3,
                   help="Number of CV repetitions (5-fold × cv_repeats total fits).")

    # Gap 26 — confusion cost matrix
    p.add_argument("--cost_matrix_path", default=None,
                   help="Path to JSON cost matrix. Keys: 'true:pred'. "
                        "Omit for hardcoded medical default (ulcer→normal = 10).")

    # Gap 18 extension — mode-collapse guard
    p.add_argument("--gan_collapse_threshold", type=float, default=0.02,
                   help="D_loss below this for --gan_collapse_patience epochs → collapse declared.")
    p.add_argument("--gan_collapse_patience", type=int, default=10,
                   help="Consecutive epochs with D_loss < threshold before collapse is logged.")

    # Gap 20 alias
    p.add_argument("--lora_rank_list", nargs="+", type=int, default=None,
                   help="Alias for --lora_ablation_ranks.")

    p.add_argument("--tune",              action="store_true")
    p.add_argument("--tune_trials",       type=int, default=15)
    p.add_argument("--tune_epochs",       type=int, default=8)
    p.add_argument("--models", nargs="+",
                   default=["efficientnetv2_rw_s", "swin", "mobile",
                            "hybrid_cnn_transformer", "hybrid_cnn_transformer_v2"])
    p.add_argument("--min_free_disk_gb",  type=float, default=20.0)

    return p.parse_args()


# ==============================================================================
# SECTION 2 — Global config
# ==============================================================================

args = parse_args()

# Resolve lora_rank_list alias
if args.lora_rank_list is not None:
    args.lora_ablation_ranks = args.lora_rank_list

# Ensure seeds list is unique and sorted
args.seeds = sorted(set(args.seeds))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

DATA_DIR       = Path(args.data_dir)
OUTPUT_DIR     = Path(args.output_dir)
IMAGE_ROOT_DIR = DATA_DIR / args.image_root
SPLITS_DIR     = OUTPUT_DIR / "splits"
SYNTH_DIR      = OUTPUT_DIR / args.synth_dir
CKPT_DIR       = OUTPUT_DIR / "checkpoints"
RESULTS_DIR    = OUTPUT_DIR / "results"
LOGS_DIR       = OUTPUT_DIR / "logs"
GAN_DIR        = OUTPUT_DIR / "gan_checkpoints"   # Gap 18

for d in [SPLITS_DIR, SYNTH_DIR, CKPT_DIR, RESULTS_DIR, LOGS_DIR, GAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES   = None
RARE_CLASSES  = []
LABEL_MAP     = {}
REV_LABEL_MAP = {}

HPARAMS = {
    "efficientnetv2_rw_s": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    },
    "swin": {
        "lr": args.lr, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    },
    "mobile": {
        "lr": args.lr * 1.5, "freeze_epochs": args.freeze_epochs,
        "fine_tune_epochs": args.fine_tune_epochs, "batch_size": args.batch_size,
        "gamma": args.gamma, "freeze_lr_mult": args.freeze_lr_mult,
        "weight_decay": args.weight_decay,
    },
    "hybrid_cnn_transformer": {
        "lr": args.lr * 0.67, "freeze_epochs": max(1, args.freeze_epochs - 6),
        "fine_tune_epochs": args.fine_tune_epochs + 6,
        "batch_size": min(args.batch_size, 8),
        "gamma": args.gamma, "freeze_lr_mult": 5.0,
        "weight_decay": args.weight_decay,
    },
    "hybrid_cnn_transformer_v2": {
        "lr": args.lr * 0.67, "freeze_epochs": max(1, args.freeze_epochs - 8),
        "fine_tune_epochs": args.fine_tune_epochs,
        "batch_size": min(args.batch_size, 16),
        "gamma": args.gamma, "freeze_lr_mult": 5.0,
        "weight_decay": args.weight_decay,
    },
}

if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1e9 < 20:
    for k in HPARAMS:
        HPARAMS[k]["batch_size"] = min(HPARAMS[k]["batch_size"], 16)
    HPARAMS["swin"]["batch_size"]                  = 8
    HPARAMS["hybrid_cnn_transformer"]["batch_size"] = 4
    HPARAMS["hybrid_cnn_transformer_v2"]["batch_size"] = 8
    print("  RTX 2080 Ti detected — batch sizes capped for 11GB VRAM")


# ==============================================================================
# SECTION 3 — Prompt constants
# ==============================================================================

DOMAIN_PREFIX   = "endoscopy photo, circular vignette, specular highlights, pink mucosa: "
NEGATIVE_PROMPT = (
    "illustration, diagram, cartoon, drawing, text, watermark, "
    "x-ray, mri, ct scan, histology, microscopy, "
    "blurry, low quality, overexposed, noisy, "
    "natural scene, person, face, outdoor"
)

CLASS_MAP = {
    "Accessory tools": 0, "Angiectasia": 1,
    "Barretts esophagus": 2, "Barrett\u2019s esophagus": 2,
    "Barrett's esophagus": 2, "Blood in lumen": 3, "Cecum": 4,
    "Colon diverticula": 5, "Colon polyps": 6, "Colorectal cancer": 7,
    "Duodenal bulb": 8, "Dyed-lifted-polyps": 9, "Dyed-resection-margins": 10,
    "Erythema": 11, "Esophageal varices": 12, "Esophagitis": 13,
    "Gastric polyps": 14, "Gastroesophageal_junction_normal z-line": 15,
    "Ileocecal valve": 16, "Mucosal inflammation large bowel": 17,
    "Normal esophagus": 18,
    "Normal mucosa and vascular pattern in the large bowel": 19,
    "Normal stomach": 20, "Pylorus": 21, "Resected polyps": 22,
    "Resection margins": 23, "Retroflex rectum": 24,
    "Small bowel_terminal ileum": 25, "Ulcer": 26,
}

CLASS_PROMPTS = {
    0:  "metal endoscopic tools, forceps or snare visible, gastroscopy",
    1:  "angiectasia, tortuous red vessels, salmon mucosa, capsule endoscopy",
    2:  "Barrett's esophagus, salmon irregular patches, lower esophagus",
    3:  "blood in lumen, dark red pooling, gastric cavity",
    4:  "cecum, pale pink mucosa, appendiceal orifice, haustral folds",
    5:  "colon diverticula, dark circular openings in colonic wall",
    6:  "colon polyp, sessile or pedunculated lesion, pink mucosa",
    7:  "colorectal cancer, irregular friable mass, ulceration, colon",
    8:  "duodenal bulb, pale smooth mucosa, circular folds",
    9:  "dyed lifted polyp, blue submucosal injection, raised lesion",
    10: "dyed resection margins, blue mucosal edges, post-polypectomy",
    11: "gastric erythema, diffuse reddish mucosal discoloration",
    12: "esophageal varices, bluish bulging veins, longitudinal, esophagus",
    13: "esophagitis, erythematous mucosa, linear erosions, esophagus",
    14: "gastric polyp, smooth rounded lesion, gastric wall",
    15: "gastroesophageal junction, z-line, squamocolumnar border",
    16: "ileocecal valve, two lips visible, cecal mucosa",
    17: "mucosal inflammation, granular friable reddish colon, lost vascular pattern",
    18: "normal esophagus, smooth pale pink mucosa, longitudinal folds",
    19: "normal colon, smooth pink mucosa, clear vascular pattern, haustrae",
    20: "normal stomach, rugal folds, pink gastric mucosa, gastric pool",
    21: "pylorus, circular orifice, antral folds, gastroscopy",
    22: "resected polyp, post-polypectomy scar, cauterized flat defect",
    23: "resection margins, cauterized edges, whitish fibrinous border",
    24: "retroflex rectum, retroflexed view, anorectal junction",
    25: "terminal ileum, pale villous mucosa, fine texture, small bowel",
    26: "gastric ulcer, mucosal crater, white fibrinous base, erythematous rim",
}

FID_TRANSFORM = T.Compose([T.Resize((299, 299)), T.ToTensor()])


# ==============================================================================
# SECTION 4 — Dataset
# ==============================================================================

def create_splits():
    raw_dir = IMAGE_ROOT_DIR
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_folder in sorted(raw_dir.iterdir()):
        if not class_folder.is_dir(): continue
        class_name = class_folder.name
        if class_name not in CLASS_MAP:
            print(f"  WARNING: {repr(class_name)} not in CLASS_MAP — skipping"); continue
        original_label = CLASS_MAP[class_name]
        images = []
        for ext in ["*.png","*.jpg","*.jpeg","*.PNG","*.JPG","*.JPEG"]:
            images.extend(class_folder.glob(ext))
        for img_path in images:
            rows.append({"image_path": str(img_path.relative_to(raw_dir)),
                         "original_label": original_label, "class_name": class_name})
        print(f"  [{original_label:2d}] {class_name:<50} {len(images):>4} images")

    df = pd.DataFrame(rows)
    unique_labels = sorted(df["original_label"].unique())
    global LABEL_MAP, REV_LABEL_MAP, NUM_CLASSES
    LABEL_MAP     = {orig: i for i, orig in enumerate(unique_labels)}
    REV_LABEL_MAP = {i: orig for orig, i in LABEL_MAP.items()}
    NUM_CLASSES   = len(unique_labels)
    df["label"]   = df["original_label"].map(LABEL_MAP)

    train_rows, val_rows, test_rows = [], [], []
    for class_id, class_df in df.groupby("label"):
        class_df = class_df.sample(frac=1, random_state=args.seed)
        n = len(class_df)
        if n == 1:
            train_rows.append(class_df)
        elif n == 2:
            train_rows.append(class_df.iloc[[0]]); val_rows.append(class_df.iloc[[1]])
        elif n < 10:
            n_train = max(1, int(0.6 * n)); n_val = max(1, int(0.2 * n))
            train_rows.append(class_df.iloc[:n_train])
            v = class_df.iloc[n_train:n_train+n_val]; t = class_df.iloc[n_train+n_val:]
            if len(v): val_rows.append(v)
            if len(t): test_rows.append(t)
        else:
            tr, tmp = train_test_split(class_df, test_size=0.2, random_state=args.seed)
            v,  t   = train_test_split(tmp,      test_size=0.5, random_state=args.seed)
            train_rows.append(tr); val_rows.append(v); test_rows.append(t)

    train_df = pd.concat(train_rows, ignore_index=True)
    val_df   = pd.concat(val_rows,   ignore_index=True)
    test_df  = pd.concat(test_rows,  ignore_index=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    test_df.to_csv( SPLITS_DIR / "test.csv",  index=False)
    unreliable = sorted([c for c in df["original_label"].unique()
                         if len(df[df["original_label"]==c]) < 30])
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
    return train_df, val_df, test_df, unreliable


class GastroVisionDataset(Dataset):
    def __init__(self, csv_path, split="train", mode="classifier", synth_dir_name=None):
        self.split = split
        df = pd.read_csv(csv_path)
        if "label" not in df.columns and "original_label" in df.columns:
            df["label"] = df["original_label"].map(LABEL_MAP)
        if {"image_path","label"} - set(df.columns):
            raise ValueError("CSV missing image_path or label columns")
        self.imagepaths     = df["image_path"].tolist()
        self.labels         = df["label"].astype(int).tolist()
        self.class_names    = df["class_name"].tolist() if "class_name" in df.columns else None
        self.synth_dir_name = synth_dir_name or args.synth_dir

        if mode == "diffusion":
            norm = T.Normalize([0.5]*3, [0.5]*3)
        else:
            norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

        if split == "train" and mode == "classifier":
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2,hue=0.1),
                T.RandomAffine(degrees=0,translate=(0.1,0.1)),
                T.ToTensor(), norm,
            ])
        elif split == "train" and mode == "diffusion":
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(), T.RandomRotation(10),
                T.ToTensor(), norm,
            ])
        else:
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)), T.ToTensor(), norm,
            ])

    def __len__(self): return len(self.imagepaths)

    def __getitem__(self, idx):
        rel  = self.imagepaths[idx]
        path = (OUTPUT_DIR / rel if rel.startswith(self.synth_dir_name + "/")
                or (OUTPUT_DIR / rel).exists() and not (IMAGE_ROOT_DIR / rel).exists()
                else IMAGE_ROOT_DIR / rel)
        if not path.exists():
            alt = IMAGE_ROOT_DIR / rel
            path = alt if alt.exists() else path
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: corrupted {path}: {e}")
            return torch.zeros(3, args.img_size, args.img_size), self.labels[idx]
        return self.transform(img), int(self.labels[idx])


class HeavyAugDataset(GastroVisionDataset):
    def __init__(self, csv_path, split="train"):
        super().__init__(csv_path, split, "classifier")
        if split == "train":
            norm = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            self.transform = T.Compose([
                T.Resize((args.img_size, args.img_size)),
                T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.4,hue=0.15),
                T.RandomAffine(degrees=15,translate=(0.15,0.15),scale=(0.8,1.2),shear=10),
                T.RandomPerspective(distortion_scale=0.4, p=0.5),
                T.GaussianBlur(kernel_size=3, sigma=(0.1,1.5)),
                T.ToTensor(), norm,
            ])


# ── Gap 11 — MixUp / CutMix datasets ─────────────────────────────────────────

class MixUpDataset(GastroVisionDataset):
    """Applies alpha-parameterised MixUp (Zhang et al. ICLR 2018) at item level."""
    def __init__(self, csv_path, split="train", alpha: float = 0.4):
        super().__init__(csv_path, split, "classifier")
        self.alpha = alpha
        # Pre-load for quick second-sample access
        self._df   = pd.read_csv(csv_path)

    def __getitem__(self, idx):
        x1, y1 = super().__getitem__(idx)
        # Sample a random second image
        idx2    = np.random.randint(len(self))
        x2, y2  = super().__getitem__(idx2)
        lam     = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        x_mix   = lam * x1 + (1 - lam) * x2
        # Return soft labels as a tuple; loss must handle this
        return x_mix, (y1, y2, lam)


class CutMixDataset(GastroVisionDataset):
    """Applies CutMix (Yun et al. ICCV 2019)."""
    def __init__(self, csv_path, split="train", alpha: float = 1.0):
        super().__init__(csv_path, split, "classifier")
        self.alpha = alpha

    @staticmethod
    def _rand_bbox(size, lam):
        W, H  = size[-1], size[-2]
        cut_w = int(W * np.sqrt(1.0 - lam))
        cut_h = int(H * np.sqrt(1.0 - lam))
        cx    = np.random.randint(W)
        cy    = np.random.randint(H)
        x1    = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
        y1    = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
        return x1, y1, x2, y2

    def __getitem__(self, idx):
        x1, y1 = super().__getitem__(idx)
        idx2   = np.random.randint(len(self))
        x2, y2 = super().__getitem__(idx2)
        lam    = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        x1_out = x1.clone()
        bx1, by1, bx2, by2 = self._rand_bbox(x1.shape, lam)
        x1_out[:, by1:by2, bx1:bx2] = x2[:, by1:by2, bx1:bx2]
        actual_lam = 1 - (bx2-bx1)*(by2-by1) / (x1.shape[-1]*x1.shape[-2])
        return x1_out, (y1, y2, actual_lam)


def _mixup_loss(criterion, logits, targets):
    """Handles soft (y1, y2, lam) or hard integer targets transparently."""
    if isinstance(targets, (list, tuple)):
        y1, y2, lam = targets
        if isinstance(y1, torch.Tensor): y1 = y1.to(logits.device)
        if isinstance(y2, torch.Tensor): y2 = y2.to(logits.device)
        return lam * criterion(logits, y1) + (1 - lam) * criterion(logits, y2)
    return criterion(logits, targets)


def _mixup_collate(batch):
    """Custom collate that handles soft-label tuples."""
    xs  = torch.stack([b[0] for b in batch])
    y1s = torch.tensor([b[1][0] for b in batch]) if isinstance(batch[0][1], tuple) else torch.tensor([b[1] for b in batch])
    if isinstance(batch[0][1], tuple):
        y2s  = torch.tensor([b[1][1] for b in batch])
        lams = float(np.mean([b[1][2] for b in batch]))
        return xs, (y1s, y2s, lams)
    return xs, y1s


class GastroVisionSDDataset(Dataset):
    def __init__(self, csv_path, tokenizer, size=512):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((size, size)), T.RandomHorizontalFlip(),
            T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3),
        ])
        self.label_to_name = (
            dict(zip(self.df["label"].astype(int), self.df["class_name"]))
            if "class_name" in self.df.columns else {}
        )

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])
        pixel = self.transform(Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB"))
        prompt = CLASS_PROMPTS.get(REV_LABEL_MAP.get(label, label),
            "endoscopy photograph of gastrointestinal tissue, "
            "round endoscopic field with dark vignette border")
        tokens = self.tokenizer(
            prompt, padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        ).input_ids.squeeze(0)
        return {"pixel_values": pixel, "input_ids": tokens, "label": label}


def get_weighted_sampler(csv_path):
    df     = pd.read_csv(csv_path)
    labels = df["label"].astype(int).tolist()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w      = 1.0 / counts
    sw     = [w[l] for l in labels]
    return WeightedRandomSampler(sw, len(sw), replacement=True)


# ==============================================================================
# SECTION 5 — Loss
# ==============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.reduction = reduction

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum" else loss


# ── Gap 23 — WeightedCrossEntropyLoss ─────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard cross-entropy with inverse-frequency class weights.
    class_counts: 1-D tensor or array of per-class sample counts.
    """
    def __init__(self, class_counts, reduction="mean"):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        counts = counts.clamp(min=1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)   # normalise so mean weight = 1
        self.register_buffer("weights", weights)
        self.reduction = reduction

    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets,
                               weight=self.weights.to(logits.device),
                               reduction=self.reduction)


# ── Gap 23 — BalancedSoftmaxLoss ──────────────────────────────────────────────

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax loss (Ren et al. NeurIPS 2020):
      L = -log( n_y * softmax(z_y) / sum_j(n_j * softmax(z_j)) )
    Equivalent to adding log(n_j) as a prior bias to each class logit.
    class_counts: 1-D tensor or array of per-class sample counts.
    """
    def __init__(self, class_counts):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1.0)
        # log-prior bias: log(n_j)
        self.register_buffer("log_prior", torch.log(counts))

    def forward(self, logits, targets):
        # Add log-prior to adjust for class frequency
        adjusted = logits + self.log_prior.to(logits.device)
        return F.cross_entropy(adjusted, targets)


# ── Gap 23 — Loss factory ──────────────────────────────────────────────────────

def get_criterion(loss_type: str = "focal", class_counts=None, gamma: float = 2.0):
    """
    Returns the appropriate loss criterion.
    loss_type: 'focal' | 'weighted_ce' | 'balanced_softmax'
    class_counts: required for 'weighted_ce' and 'balanced_softmax'.
    """
    if loss_type == "weighted_ce":
        if class_counts is None:
            raise ValueError("WeightedCrossEntropyLoss requires class_counts.")
        return WeightedCrossEntropyLoss(class_counts)
    elif loss_type == "balanced_softmax":
        if class_counts is None:
            raise ValueError("BalancedSoftmaxLoss requires class_counts.")
        return BalancedSoftmaxLoss(class_counts)
    else:  # default: focal
        return FocalLoss(gamma=gamma)


# ==============================================================================
# SECTION 6 — Model definitions
# ==============================================================================

def get_effnetv2_s(num_classes):
    return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes)

def get_swin_transformer(num_classes):
    return timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)

def get_mobilenetv3(num_classes):
    return timm.create_model("tf_mobilenetv3_large_minimal_100", pretrained=True, num_classes=num_classes)


class CrossAttentionFusion(nn.Module):
    def __init__(self, cnn_dim, tfm_dim, num_heads=8, dropout=0.1):
        super().__init__()
        d = min(cnn_dim, tfm_dim)
        self.cp = nn.Linear(cnn_dim, d); self.tp = nn.Linear(tfm_dim, d)
        self.ca1 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.ca2 = nn.MultiheadAttention(d, num_heads, dropout=dropout, batch_first=True)
        self.n1  = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.out_dim = d * 2

    def forward(self, cf, tf):
        cq = self.cp(cf).unsqueeze(1); tq = self.tp(tf).unsqueeze(1)
        a1, _ = self.ca1(cq, tq, tq); a1 = self.n1(a1.squeeze(1) + cq.squeeze(1))
        a2, _ = self.ca2(tq, cq, cq); a2 = self.n2(a2.squeeze(1) + tq.squeeze(1))
        return torch.cat([a1, a2], dim=-1)


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.cnn    = timm.create_model("convnext_small",              pretrained=pretrained, num_classes=0)
        self.tfm    = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0)
        self.fusion = CrossAttentionFusion(self.cnn.num_features, self.tfm.num_features)
        fd = self.fusion.out_dim
        self.head   = nn.Sequential(
            nn.LayerNorm(fd), nn.Dropout(dropout),
            nn.Linear(fd, 512), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(512, num_classes),
        )

    def forward(self, x): return self.head(self.fusion(self.cnn(x), self.tfm(x)))

    def freeze_backbones(self):
        for p in self.cnn.parameters(): p.requires_grad = False
        for p in self.tfm.parameters(): p.requires_grad = False
        for p in self.fusion.parameters(): p.requires_grad = True
        for p in self.head.parameters(): p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


class HybridCNNTransformerV2(nn.Module):
    def __init__(self, num_classes, cnn_name="efficientnetv2_rw_s",
                 transformer_dim=512, depth=4, heads=8, mlp_dim=1024,
                 dropout=0.1, img_size=224):
        super().__init__()
        self.cnn = timm.create_model(cnn_name, pretrained=True, features_only=True)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            last  = self.cnn(dummy)[-1]
            cout  = last.shape[1]; self.n_tokens = last.shape[2] * last.shape[3]
        self.cnn_proj  = nn.Conv2d(cout, transformer_dim, 1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.cls_pos   = nn.Parameter(torch.randn(1, 1, transformer_dim))
        self.patch_pos = nn.Parameter(torch.randn(1, self.n_tokens, transformer_dim))
        enc = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(transformer_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(transformer_dim, num_classes))

    def forward(self, x):
        B = x.shape[0]
        proj    = self.cnn_proj(self.cnn(x)[-1])
        patches = proj.flatten(2).transpose(1, 2) + self.patch_pos
        cls     = self.cls_token.expand(B, -1, -1) + self.cls_pos
        tokens  = self.transformer(torch.cat([cls, patches], dim=1))
        return self.head(self.norm(tokens[:, 0]))

    def freeze_backbones(self):
        for p in self.cnn.parameters():         p.requires_grad = False
        for p in self.cnn_proj.parameters():    p.requires_grad = False
        for p in self.transformer.parameters(): p.requires_grad = False
        for p in self.head.parameters():        p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad = True


MODEL_REGISTRY = {
    "efficientnetv2_rw_s":       get_effnetv2_s,
    "swin":                      get_swin_transformer,
    "mobile":                    get_mobilenetv3,
    "hybrid_cnn_transformer":    lambda n: HybridCNNTransformer(n),
    "hybrid_cnn_transformer_v2": lambda n: HybridCNNTransformerV2(n),
}

def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](NUM_CLASSES).to(DEVICE)


def load_checkpoint(model_name, augmented=False, suffix_override=None):
    global NUM_CLASSES
    suffix = suffix_override if suffix_override else ("_aug" if augmented else "")
    path   = CKPT_DIR / f"sota_{model_name}{suffix}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=DEVICE)
    for key in state:
        if "classifier" in key and "weight" in key:
            NUM_CLASSES = state[key].shape[0]; break
    model = get_model(model_name)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_name} from {path}")
    return model


# ==============================================================================
# SECTION 7 — Gap 17: Experiment tracking (_log helper)
# ==============================================================================

_csv_log_path = LOGS_DIR / "metrics_log.csv"
_csv_log_initialized = False

def _log(step: int, metrics: dict, use_wandb: bool = False, prefix: str = "") -> None:
    """
    Log metrics to W&B (if available + requested) and always to a CSV file.
    prefix: e.g. "train/", "eval/", "diffusion/"
    """
    global _csv_log_initialized

    tagged = {f"{prefix}{k}": v for k, v in metrics.items()}
    tagged["step"] = step

    # W&B
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log(tagged, step=step)
        except Exception as e:
            pass  # silently skip W&B errors

    # CSV fallback (always)
    import csv
    row = {k: (f"{v:.6f}" if isinstance(v, float) else str(v)) for k, v in tagged.items()}
    write_header = not _csv_log_path.exists() or not _csv_log_initialized
    _csv_log_initialized = True
    with open(_csv_log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _init_wandb(run_name: str = None) -> None:
    if not WANDB_AVAILABLE:
        print("  W&B not installed — pip install wandb"); return
    wandb.init(
        project = args.wandb_project,
        name    = run_name or args.wandb_run,
        config  = vars(args),
        resume  = "allow",
    )
    print(f"  W&B run initialised: {wandb.run.name}")


# ==============================================================================
# SECTION 8 — Training engine
# ==============================================================================

def _freeze(model, model_name):
    if "hybrid" in model_name:
        model.freeze_backbones()
    else:
        for p in model.parameters(): p.requires_grad = False
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        if head is None: raise AttributeError(f"No head on {model_name}")
        for p in head.parameters(): p.requires_grad = True

def _unfreeze(model, model_name):
    if "hybrid" in model_name: model.unfreeze_all()
    else:
        for p in model.parameters(): p.requires_grad = True

def _eval_acc(model, loader):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            with autocast():
                preds = model(xb.to(DEVICE)).argmax(1)
            yp.append(preds.cpu().numpy()); yt.append(yb.numpy())
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    return float((yt == yp).mean()), yt, yp


# ── Gap 12 — Training-curve plotter ──────────────────────────────────────────

def plot_training_history(history: dict, model_name: str, save_dir: Path) -> None:
    """
    Plots train_loss and val_acc vs epoch with a phase boundary marker.
    history: {"train_loss": [...], "val_acc": [...], "phase": [...]}
    """
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        n_epochs  = len(history["train_loss"])
        phase_chg = next((i for i, p in enumerate(history["phase"]) if p == "finetune"), None)
        epochs    = list(range(1, n_epochs + 1))

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax2 = ax1.twinx()
        ax1.plot(epochs, history["train_loss"], color="#d65f5f", lw=2, label="Train Loss")
        ax2.plot(epochs, history["val_acc"],    color="#4878cf", lw=2, label="Val Acc")
        if phase_chg:
            ax1.axvline(phase_chg + 0.5, color="gray", linestyle="--", alpha=0.7,
                        label="Freeze → Fine-tune")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color="#d65f5f")
        ax2.set_ylabel("Val Accuracy", color="#4878cf")
        ax1.set_title(f"{model_name} — Training History")
        lines1, labs1 = ax1.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper right")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        out = save_dir / f"training_history_{model_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Training history saved → {out}")
    except Exception as e:
        print(f"  Warning: could not save training history: {e}")


def train_classifier(model_name, train_csv, val_csv, augmented=False):
    cfg    = HPARAMS[model_name]
    scaler = GradScaler()
    # Gap 23 — compute class counts for frequency-weighted losses
    _train_df_tmp = pd.read_csv(train_csv)
    _class_counts = np.bincount(_train_df_tmp["label"].astype(int).tolist(),
                                 minlength=NUM_CLASSES).tolist()
    crit = get_criterion(args.loss_type, _class_counts, gamma=cfg["gamma"])
    model   = get_model(model_name)
    history = {"train_loss": [], "val_acc": [], "phase": []}
    ckpt    = CKPT_DIR / f"sota_{model_name}{'_aug' if augmented else ''}.pt"
    suffix  = "_aug" if augmented else ""

    if args.use_wandb:
        _init_wandb(f"{model_name}{suffix}")

    train_ds = GastroVisionDataset(train_csv, "train", "classifier", synth_dir_name=args.synth_dir)
    val_ds   = GastroVisionDataset(val_csv,   "val",   "classifier", synth_dir_name=args.synth_dir)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=4, pin_memory=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # Phase 1
    _freeze(model, model_name)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0))
    print(f"\n{'='*60}\n[{model_name}] Phase 1: frozen ({cfg['freeze_epochs']} epochs)\n{'='*60}")
    global_step = 0
    for ep in range(cfg["freeze_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update()
            rl += loss.item(); global_step += 1
            _log(global_step, {"loss": loss.item()}, args.use_wandb, f"{model_name}{suffix}/train/")
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc); history["phase"].append("freeze")
        _log(ep, {"val_acc": acc, "train_loss": rl/len(tl)},
             args.use_wandb, f"{model_name}{suffix}/epoch/")
        print(f"  Ep {ep+1:2d}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    # Phase 2
    _unfreeze(model, model_name)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] Phase 2: fine-tune ({cfg['fine_tune_epochs']} epochs)\n{'='*60}")
    for ep in range(cfg["fine_tune_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            rl += loss.item(); global_step += 1
            _log(global_step, {"loss": loss.item()}, args.use_wandb, f"{model_name}{suffix}/train/")
        sch.step()
        acc, _, _ = _eval_acc(model, vl)
        history["train_loss"].append(rl / len(tl))
        history["val_acc"].append(acc); history["phase"].append("finetune")
        _log(ep + cfg["freeze_epochs"], {"val_acc": acc, "train_loss": rl/len(tl)},
             args.use_wandb, f"{model_name}{suffix}/epoch/")
        print(f"  Ep {ep+1:2d}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            with open(ckpt.with_suffix(".meta.json"), "w") as mf:
                json.dump({"num_classes": NUM_CLASSES}, mf)
            print(f"  Saved (val_acc={best_acc:.4f})")

    print(f"\n  {model_name} best val_acc: {best_acc:.4f}")
    plot_training_history(history, model_name + suffix, RESULTS_DIR)   # Gap 12
    if args.use_wandb and WANDB_AVAILABLE:
        try: wandb.finish()
        except: pass
    return history


def train_classifier_heavy_aug(model_name, train_csv, val_csv):
    cfg    = HPARAMS[model_name]
    scaler = GradScaler()
    model  = get_model(model_name)
    ckpt   = CKPT_DIR / f"sota_{model_name}_heavy.pt"
    history = {"train_loss": [], "val_acc": [], "phase": []}
    _train_df_tmp  = pd.read_csv(train_csv)
    _class_counts  = np.bincount(_train_df_tmp["label"].astype(int).tolist(),
                                   minlength=NUM_CLASSES).tolist()
    crit = get_criterion(args.loss_type, _class_counts, gamma=cfg["gamma"])

    train_ds = HeavyAugDataset(train_csv, "train")
    val_ds   = GastroVisionDataset(val_csv, "val", "classifier", synth_dir_name=args.synth_dir)
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=2, pin_memory=True, persistent_workers=True)
    vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=2, pin_memory=True, persistent_workers=True)

    _freeze(model, model_name)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0))
    print(f"\n{'='*60}\n[{model_name}] Heavy Aug — Phase 1\n{'='*60}")
    for ep in range(cfg["freeze_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update(); rl += loss.item()
        acc = _eval_acc(model, vl)[0]
        history["train_loss"].append(rl/len(tl)); history["val_acc"].append(acc); history["phase"].append("freeze")
        print(f"  Ep {ep+1}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    _unfreeze(model, model_name)
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] Heavy Aug — Phase 2\n{'='*60}")
    for ep in range(cfg["fine_tune_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast(): loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); rl += loss.item()
        sch.step()
        acc = _eval_acc(model, vl)[0]
        history["train_loss"].append(rl/len(tl)); history["val_acc"].append(acc); history["phase"].append("finetune")
        print(f"  Ep {ep+1}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            with open(ckpt.with_suffix(".meta.json"), "w") as mf:
                json.dump({"num_classes": NUM_CLASSES}, mf)
            print(f"  Saved (val_acc={best_acc:.4f})")

    print(f"  {model_name} heavy aug best val_acc: {best_acc:.4f}")
    plot_training_history(history, f"{model_name}_heavy", RESULTS_DIR)
    return best_acc


# ── Gap 11 — MixUp / CutMix trainer ──────────────────────────────────────────

def train_classifier_mixup(model_name, train_csv, val_csv,
                            mode="mixup", alpha=0.4):
    """Strategy S4: MixUp or CutMix augmentation."""
    cfg    = HPARAMS[model_name]
    scaler = GradScaler()
    model  = get_model(model_name)
    _train_df_tmp = pd.read_csv(train_csv)
    _class_counts = np.bincount(_train_df_tmp["label"].astype(int).tolist(),
                                 minlength=NUM_CLASSES).tolist()
    # MixUp/CutMix always uses focal loss (soft labels incompatible with weighted variants)
    crit  = FocalLoss(gamma=cfg["gamma"])
    label = "mixup" if mode == "mixup" else "cutmix"
    ckpt   = CKPT_DIR / f"sota_{model_name}_{label}.pt"
    history = {"train_loss": [], "val_acc": [], "phase": []}

    DS = MixUpDataset if mode == "mixup" else CutMixDataset
    train_ds = DS(train_csv, "train", alpha=alpha)
    val_ds   = GastroVisionDataset(val_csv, "val", "classifier")
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=2, pin_memory=True, collate_fn=_mixup_collate)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=2, pin_memory=True)

    _freeze(model, model_name)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0))
    print(f"\n{'='*60}\n[{model_name}] {label} Phase 1\n{'='*60}")
    for ep in range(cfg["freeze_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                logits = model(xb)
                loss   = _mixup_loss(crit, logits,
                                     (yb[0].to(DEVICE), yb[1].to(DEVICE), yb[2])
                                     if isinstance(yb, (list,tuple)) else yb.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(opt); scaler.update(); rl += loss.item()
        acc = _eval_acc(model, vl)[0]
        history["train_loss"].append(rl/len(tl)); history["val_acc"].append(acc); history["phase"].append("freeze")
        print(f"  Ep {ep+1}/{cfg['freeze_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")

    _unfreeze(model, model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay",0.01))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
    best_acc = 0.0
    print(f"\n{'='*60}\n[{model_name}] {label} Phase 2\n{'='*60}")
    for ep in range(cfg["fine_tune_epochs"]):
        model.train(); rl = 0.0
        for xb, yb in tl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                logits = model(xb)
                loss   = _mixup_loss(crit, logits,
                                     (yb[0].to(DEVICE), yb[1].to(DEVICE), yb[2])
                                     if isinstance(yb, (list,tuple)) else yb.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); rl += loss.item()
        sch.step()
        acc = _eval_acc(model, vl)[0]
        history["train_loss"].append(rl/len(tl)); history["val_acc"].append(acc); history["phase"].append("finetune")
        print(f"  Ep {ep+1}/{cfg['fine_tune_epochs']}  loss={rl/len(tl):.4f}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt)
            print(f"  Saved (val_acc={best_acc:.4f})")

    print(f"  {model_name} {label} best val_acc: {best_acc:.4f}")
    plot_training_history(history, f"{model_name}_{label}", RESULTS_DIR)
    return best_acc


def tune_classifier(model_name, train_csv, val_csv, n_trials=15, tune_epochs=8):
    if not OPTUNA_AVAILABLE:
        print(f"  Optuna not available — skipping tuning for {model_name}"); return

    def objective(trial):
        lr             = trial.suggest_float("lr",             1e-5, 5e-4, log=True)
        gamma          = trial.suggest_float("gamma",          0.5,  3.0)
        freeze_lr_mult = trial.suggest_float("freeze_lr_mult", 2.0,  15.0)
        weight_decay   = trial.suggest_float("weight_decay",   1e-5, 1e-2, log=True)
        batch_size     = trial.suggest_categorical("batch_size", [8, 16])

        train_ds = GastroVisionDataset(train_csv, "train", "classifier", synth_dir_name=args.synth_dir)
        val_ds   = GastroVisionDataset(val_csv,   "val",   "classifier", synth_dir_name=args.synth_dir)
        tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        vl = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model  = get_model(model_name)
        crit   = FocalLoss(gamma=gamma)
        scaler = GradScaler()

        _freeze(model, model_name)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr * freeze_lr_mult)
        for _ in range(min(3, tune_epochs // 2)):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast(): loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                scaler.step(opt); scaler.update()

        _unfreeze(model, model_name)
        opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tune_epochs)
        best_acc = 0.0
        for ep in range(tune_epochs):
            model.train()
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                with autocast(): loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sch.step()
            acc = _eval_acc(model, vl)[0]
            best_acc = max(best_acc, acc)
            trial.report(acc, ep)
            if trial.should_prune():
                del model; torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del model; torch.cuda.empty_cache()
        return best_acc

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=4, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial.params
    print(f"  Best val_acc: {study.best_value:.4f}")
    HPARAMS[model_name].update({k: best[k] for k in best})
    hparams_path = OUTPUT_DIR / "best_hparams.json"
    with open(hparams_path, "w") as f: json.dump(HPARAMS, f, indent=2)
    return study


# ==============================================================================
# SECTION 9 — EMA + SNR helpers (unchanged)
# ==============================================================================

class EMAModel:
    def __init__(self, model, decay=0.9999, update_after_step=100):
        self.decay = decay; self.update_after_step = update_after_step; self.step_count = 0
        self.shadow = {n: p.detach().cpu().clone()
                       for n, p in model.named_parameters() if p.requires_grad}

    def step(self, model):
        self.step_count += 1
        decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        if self.step_count < self.update_after_step:
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    self.shadow[n] = p.detach().cpu().clone()
            return
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow and p.requires_grad:
                    s = self.shadow[n].to(p.device)
                    s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
                    self.shadow[n] = s.cpu()

    def copy_to(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow and p.requires_grad:
                p.data.copy_(self.shadow[n].to(p.device))

    def restore(self, model, orig):
        for n, p in model.named_parameters():
            if n in orig and p.requires_grad:
                p.data.copy_(orig[n].to(p.device))

    def state_dict(self):
        return {"shadow": self.shadow, "step_count": self.step_count, "decay": self.decay}

    def load_state_dict(self, s):
        self.shadow = s["shadow"]; self.step_count = s["step_count"]; self.decay = s.get("decay", self.decay)

    def save_adapter(self, model, path):
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        orig = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        try:
            self.copy_to(model); model.save_pretrained(path)
            print(f"  EMA adapter saved → {path}")
        finally:
            self.restore(model, orig)


def _snr_weights(scheduler, t, device, gamma=5.0):
    ac  = scheduler.alphas_cumprod.to(device)
    snr = (ac[t] ** 0.5 / ((1 - ac[t]) ** 0.5 + 1e-8)) ** 2
    return (torch.clamp(snr, max=gamma) / (snr + 1e-8)).detach()


# ==============================================================================
# SECTION 10 — Domain adaptation (SD LoRA)
# ==============================================================================

def domain_adapt_sd():
    train_csv = SPLITS_DIR / args.train_csv
    print("Loading SD components...")
    vram_gb     = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    offload_cpu = vram_gb < 20
    print(f"  GPU VRAM: {vram_gb:.0f}GB  CPU offload: {offload_cpu}")

    tokenizer    = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet")
    noise_sched  = DDPMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")

    if offload_cpu:
        unet = unet.to(DEVICE); text_encoder = text_encoder.cpu(); vae = vae.cpu()
    else:
        text_encoder = text_encoder.to(DEVICE); vae = vae.to(DEVICE); unet = unet.to(DEVICE)

    vae.requires_grad_(False); text_encoder.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["to_q","to_k","to_v","to_out.0","proj_in","proj_out"],
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()

    ema     = EMAModel(unet, decay=args.ema_decay, update_after_step=args.ema_warmup_steps)
    dataset = GastroVisionSDDataset(train_csv, tokenizer)
    loader  = DataLoader(dataset, batch_size=args.sd_batch_size, shuffle=True,
                         num_workers=4, pin_memory=True, drop_last=True)
    opt     = torch.optim.AdamW(unet.parameters(), lr=args.sd_lr, weight_decay=1e-4)
    lrsched = get_diffusers_scheduler("cosine", optimizer=opt,
                                       num_warmup_steps=500,
                                       num_training_steps=args.domain_adapt_steps)
    scaler  = GradScaler()

    resume_path = CKPT_DIR / "resume_sd_lora.pt"
    step = 0; losses = []
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=DEVICE)
        unet.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optimizer"])
        lrsched.load_state_dict(ck["scheduler"]); step = ck["global_step"]
        losses = ck.get("losses", [])
        if "ema" in ck: ema.load_state_dict(ck["ema"])
        print(f"Resumed at step {step}/{args.domain_adapt_steps}")

    unet.train(); opt.zero_grad(); it = iter(loader); rl = 0.0
    while step < args.domain_adapt_steps:
        try: batch = next(it)
        except StopIteration: it = iter(loader); batch = next(it)

        pv = batch["pixel_values"].to(DEVICE)
        ii = batch["input_ids"].to(DEVICE)

        vae_on_gpu = next(vae.parameters()).device == DEVICE
        if not vae_on_gpu and not offload_cpu: vae.to(DEVICE)
        with torch.no_grad():
            lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
        if offload_cpu: vae.cpu(); torch.cuda.empty_cache()

        noise = torch.randn_like(lat)
        t     = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],), device=DEVICE).long()
        w     = _snr_weights(noise_sched, t, DEVICE)
        nl    = noise_sched.add_noise(lat, noise, t)

        te_on_gpu = next(text_encoder.parameters()).device == DEVICE
        if not te_on_gpu and not offload_cpu: text_encoder.to(DEVICE)
        with torch.no_grad(): hs = text_encoder(ii)[0]
        if offload_cpu: text_encoder.cpu(); torch.cuda.empty_cache()

        with autocast():
            pred = unet(nl, t, hs).sample
            lps  = F.mse_loss(pred, noise, reduction="none").mean(dim=[1,2,3])
            loss = (lps * w).mean() / args.sd_grad_accum

        scaler.scale(loss).backward()
        rl += loss.item() * args.sd_grad_accum

        if (step + 1) % args.sd_grad_accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            scaler.step(opt); scaler.update(); lrsched.step(); opt.zero_grad()
            ema.step(unet)

        step += 1
        if step % 100 == 0:
            avg = rl / 100; losses.append(avg); rl = 0.0
            lr_now = opt.param_groups[0]["lr"]
            print(f"  Step {step:5d}/{args.domain_adapt_steps}  loss={avg:.4f}  lr={lr_now:.2e}")
            _log(step, {"loss": avg, "lr": lr_now}, args.use_wandb, "sd/")
        if step % 500 == 0:
            torch.save({"state_dict": unet.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": lrsched.state_dict(), "global_step": step,
                        "losses": losses, "ema": ema.state_dict()}, resume_path)

    torch.save(unet.state_dict(), CKPT_DIR / "sd_gastrovision_lora.pt")
    unet.save_pretrained(CKPT_DIR / "sd_gastrovision_lora_adapter")
    ema.save_adapter(unet, CKPT_DIR / "sd_gastrovision_lora_ema_adapter")
    final = losses[-1] if losses else float("nan")
    print(f"\nDomain adaptation done — final loss: {final:.4f}")
    if final > 0.08:
        print(f"  Loss > 0.08 — consider more steps (--domain_adapt_steps {args.domain_adapt_steps+5000})")

    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        sx = [i * 100 for i in range(1, len(losses)+1)]
        ax.plot(sx, losses, color="#4878cf", lw=1.5)
        if len(losses) > 10:
            w = max(5, len(losses)//20)
            sm = np.convolve(losses, np.ones(w)/w, mode="valid")
            ax.plot(sx[w-1:], sm, color="#d65f5f", lw=2.0, alpha=0.8)
        ax.axhline(0.05, color="#6acc65", linestyle="--", alpha=0.7, label="Target 0.05")
        ax.axhline(0.08, color="#f0a500", linestyle="--", alpha=0.7, label="Acceptable 0.08")
        ax.set_xlabel("Step"); ax.set_ylabel("SNR-weighted MSE Loss")
        ax.set_title("SD LoRA Domain Adaptation"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sd_loss.png", dpi=150, bbox_inches="tight"); plt.close()
    except Exception as e:
        print(f"  Warning: could not save loss plot: {e}")

    del unet, vae, text_encoder, tokenizer; torch.cuda.empty_cache(); gc.collect()


# ==============================================================================
# SECTION 11 — Synthetic image generation (Gap 16: adaptive ratio)
# ==============================================================================

def _postprocess(img, sharpen=1.4, contrast=1.15):
    return ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(sharpen)).enhance(contrast)


def generate_synthetic(real_df: pd.DataFrame = None):
    """
    Gap 16: target count per class is adaptive when --synthetic_ratio > 0:
      target = max(samples_per_class_floor, n_real * synthetic_ratio)
    Otherwise falls back to args.samples_per_class (original behaviour).
    """
    free_gb = shutil.disk_usage(OUTPUT_DIR).free / (1024**3)
    if free_gb < args.min_free_disk_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free, "
                           f"minimum {args.min_free_disk_gb} GB.")
    print(f"Disk space check passed: {free_gb:.1f} GB free")

    ema_path = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
    raw_path = CKPT_DIR / "sd_gastrovision_lora_adapter"
    adapter  = ema_path if ema_path.exists() else raw_path
    if not adapter.exists():
        raise FileNotFoundError("No LoRA adapter found. Run domain adaptation first.")

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id, torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter)
    pipe.unet.eval(); pipe.enable_attention_slicing()

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if vram_gb < 20:
        try: pipe.enable_sequential_cpu_offload()
        except Exception as e: print(f"  CPU offload unavailable: {e}")
    else:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass

    if real_df is None:
        real_df = pd.read_csv(SPLITS_DIR / args.train_csv)
    l2n = (dict(zip(real_df["label"].astype(int), real_df["class_name"]))
           if "class_name" in real_df.columns else {})

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for cls in RARE_CLASSES:
        cls_name = l2n.get(cls, f"class_{cls}")
        cls_dir  = SYNTH_DIR / str(cls)
        cls_dir.mkdir(parents=True, exist_ok=True)
        original_label = REV_LABEL_MAP.get(cls, cls)
        prompt = DOMAIN_PREFIX + CLASS_PROMPTS.get(original_label,
                 f"endoscopy photo, {cls_name}, circular vignette")

        # Gap 16 — adaptive target
        n_real = len(real_df[real_df["label"] == cls])
        if args.synthetic_ratio > 0:
            target = max(args.samples_per_class, n_real * args.synthetic_ratio)
        else:
            target = args.samples_per_class

        tokens_n = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.shape[1]
        if tokens_n > 77:
            print(f"  Prompt too long ({tokens_n} tokens > 77) — will be truncated")

        existing = sorted(cls_dir.glob("synth_*.png"))
        for p in existing:
            rows.append({"image_path": str(p.relative_to(OUTPUT_DIR)),
                         "label": cls, "class_name": cls_name, "source": "sd_ema"})
        if len(existing) >= target:
            print(f"Class {cls}: {len(existing)}/{target} images done — skipping"); continue

        start = len(existing); idx = start
        print(f"\nClass {cls} ({cls_name}): generating {target - start} images (target={target})")
        while idx < target:
            n = min(args.gen_batch_size, target - idx)
            torch.cuda.empty_cache(); gc.collect()
            gens = [torch.Generator(device=DEVICE).manual_seed(args.seed + cls*100000 + idx + i)
                    for i in range(n)]
            with torch.no_grad():
                imgs = pipe(prompt=[prompt]*n, negative_prompt=[NEGATIVE_PROMPT]*n,
                            num_inference_steps=args.gen_steps, guidance_scale=args.guidance_scale,
                            height=512, width=512, generator=gens).images
            for img in imgs:
                img  = _postprocess(img.resize((args.img_size, args.img_size), Image.LANCZOS))
                path = cls_dir / f"synth_{idx:05d}.png"; img.save(path)
                rows.append({"image_path": str(path.relative_to(OUTPUT_DIR)),
                             "label": cls, "class_name": cls_name, "source": "sd_ema"})
                idx += 1
            if idx % 100 == 0 or idx >= target:
                print(f"  {idx}/{target}")
        print(f"Class {cls} done")

    del pipe; torch.cuda.empty_cache(); gc.collect()
    synth_df = pd.DataFrame(rows)
    synth_df.to_csv(SYNTH_DIR / "synthetic_train.csv", index=False)
    print(f"\n{len(synth_df)} synthetic images saved → {SYNTH_DIR / 'synthetic_train.csv'}")
    return synth_df


# ==============================================================================
# SECTION 11B — FID helpers (kept for backward compatibility)
# ==============================================================================

def _fid_features(df, root_dir, model, hook_list):
    feats = []
    for _, row in df.iterrows():
        try:
            img    = Image.open(root_dir / row["image_path"]).convert("RGB")
            tensor = FID_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            hook_list.clear()
            with torch.no_grad(): _ = model(tensor)
            if hook_list: feats.append(hook_list[0].flatten())
        except Exception: continue
    return np.array(feats) if feats else None


def _frechet(r, s):
    mr, ms = r.mean(0), s.mean(0)
    sr = np.cov(r, rowvar=False) + 1e-6 * np.eye(r.shape[1])
    ss = np.cov(s, rowvar=False) + 1e-6 * np.eye(s.shape[1])
    d  = mr - ms; cov = sqrtm(sr @ ss)
    if np.iscomplexobj(cov): cov = cov.real
    return float(d @ d + np.trace(sr) + np.trace(ss) - 2 * np.trace(cov))


def _kid(r, s):
    from sklearn.metrics.pairwise import polynomial_kernel
    n   = min(len(r), len(s), 500)
    rng = np.random.default_rng(args.seed)
    r   = r[rng.choice(len(r), n, replace=False)]
    s   = s[rng.choice(len(s), n, replace=False)]
    g   = 1.0 / r.shape[1]
    krr = polynomial_kernel(r, r, degree=3, gamma=g, coef0=1)
    kss = polynomial_kernel(s, s, degree=3, gamma=g, coef0=1)
    krs = polynomial_kernel(r, s, degree=3, gamma=g, coef0=1)
    np.fill_diagonal(krr, 0); np.fill_diagonal(kss, 0)
    return float((krr.sum()/(n*(n-1)) + kss.sum()/(n*(n-1)) - 2*krs.mean()) * 1000)


def compute_fid(real_df, synth_df):
    print("\nComputing FID / KID...")
    inc = inception_v3(pretrained=True, aux_logits=True, transform_input=False).to(DEVICE)
    inc.fc = nn.Identity(); inc.AuxLogits = None; inc.eval()
    hook_list = []
    def hook(m, i, o): hook_list.append(o.detach().flatten(1).cpu().numpy())
    h = inc.avgpool.register_forward_hook(hook)
    real_pooled  = real_df[real_df["label"].isin(RARE_CLASSES)]
    synth_pooled = synth_df[synth_df["label"].isin(RARE_CLASSES)]
    fr = _fid_features(real_pooled,  IMAGE_ROOT_DIR, inc, hook_list)
    fs = _fid_features(synth_pooled, OUTPUT_DIR,      inc, hook_list)
    h.remove(); del inc; torch.cuda.empty_cache()
    if fr is None or fs is None:
        print("  FID: insufficient features"); return None, None
    fid = _frechet(fr, fs); kid = _kid(fr, fs)
    print(f"  Pooled FID      = {fid:.2f}  (n_real={len(fr)}, n_synth={len(fs)})")
    print(f"  Pooled KID×1000 = {kid:.3f}")
    return fid, kid


# ==============================================================================
# SECTION 11C — Confidence-weighted ensemble (Gap 8: predict_with_confidence)
# ==============================================================================

class ConfidenceEnsemble:
    def __init__(self, model_names, suffix=""):
        self.models = {}; self.suffix = suffix
        for name in model_names:
            ckpt = CKPT_DIR / f"sota_{name}{suffix}.pt"
            if not ckpt.exists():
                print(f"  Ensemble: skipping {name} — {ckpt.name} not found"); continue
            try:
                m = get_model(name)
                m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
                m.eval(); self.models[name] = m
                print(f"  Ensemble: loaded {name}")
            except Exception as e:
                print(f"  Ensemble: failed to load {name}: {e}")
        if not self.models:
            raise RuntimeError(f"Ensemble: no models loaded for suffix='{suffix}'.")
        print(f"  Ensemble ready: {len(self.models)} models [{', '.join(self.models.keys())}]")

    def predict(self, x):
        x          = x.to(DEVICE)
        probs_list = []
        with torch.no_grad():
            for m in self.models.values():
                with autocast(): probs_list.append(F.softmax(m(x), dim=1))
        stacked        = torch.stack(probs_list, dim=0)              # (M, B, C)
        confidences    = stacked.max(dim=2).values.permute(1, 0)     # (B, M)
        weights        = confidences / confidences.sum(dim=1, keepdim=True)
        ensemble_probs = (stacked * weights.permute(1, 0).unsqueeze(-1)).sum(dim=0)
        return ensemble_probs.argmax(dim=1), ensemble_probs

    # ── Gap 8 ─────────────────────────────────────────────────────────────────
    def predict_with_confidence(self, x):
        """Returns (preds, probs, confidence) — 3-tuple expected by evaluation.py."""
        preds, probs = self.predict(x)
        confidence   = probs.max(dim=1).values
        return preds, probs, confidence


def eval_ensemble(ensemble, loader):
    yt_list, yp_list, pr_list = [], [], []
    for xb, yb in loader:
        preds, probs = ensemble.predict(xb)
        yt_list.append(yb.numpy())
        yp_list.append(preds.cpu().numpy())
        pr_list.append(probs.cpu().numpy())
    yt = np.concatenate(yt_list); yp = np.concatenate(yp_list); pr = np.concatenate(pr_list)
    return float((yt == yp).mean()), yt, yp, pr


# ==============================================================================
# SECTION 11D — Gap 3: Grad-CAM visualisation grid
# ==============================================================================

def _get_gradcam_target_layer(model, model_name: str):
    """Return the last convolutional/attention layer for each architecture."""
    if "hybrid_cnn_transformer_v2" in model_name:
        # Last stage of EfficientNetV2-S feature extractor
        stages = list(model.cnn.children())
        return stages[-1] if stages else None
    elif "hybrid_cnn_transformer" in model_name:
        # Last ConvNeXt stage
        return model.cnn.stages[-1]
    elif "swin" in model_name:
        return model.layers[-1].blocks[-1].norm1
    elif "efficientnetv2" in model_name:
        return model.blocks[-1]
    elif "mobile" in model_name:
        return model.blocks[-1]
    return None


def generate_gradcam_grid(model, model_name: str, val_loader,
                           rare_classes, save_dir: Path,
                           n_per_class: int = 4) -> None:
    """
    Gap 3: Generates a Grad-CAM heatmap grid for each rare class.
    Saves one PNG per class showing real images + overlaid activation maps.
    """
    if not GRADCAM_AVAILABLE:
        print("  Grad-CAM not available — install pytorch-grad-cam"); return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_layer = _get_gradcam_target_layer(model, model_name)
    if target_layer is None:
        print(f"  Could not identify target layer for {model_name} — skipping Grad-CAM"); return

    cam = GradCAM(model=model, target_layers=[target_layer])
    model.eval()

    # Collect images per rare class from the val_loader
    cls_images: dict = {c: [] for c in rare_classes}
    cls_labels_seen: dict = {c: 0 for c in rare_classes}

    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])

    for xb, yb in val_loader:
        for img, lbl in zip(xb, yb):
            c = int(lbl.item())
            if c in cls_images and cls_labels_seen[c] < n_per_class:
                cls_images[c].append(img)
                cls_labels_seen[c] += 1
        if all(v >= n_per_class for v in cls_labels_seen.values()):
            break

    for cls, imgs in cls_images.items():
        if not imgs:
            continue
        rows_vis = []
        for img_t in imgs:
            input_t = img_t.unsqueeze(0).to(DEVICE)
            targets = [ClassifierOutputTarget(cls)]
            try:
                grayscale_cam = cam(input_tensor=input_t, targets=targets)[0]
                # Denormalise for display
                rgb = (img_t * std[:, None, None] + mean[:, None, None]).clamp(0, 1)
                rgb_np = rgb.permute(1, 2, 0).numpy()
                overlay = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
                rows_vis.append(np.concatenate([
                    (rgb_np * 255).astype(np.uint8),
                    overlay
                ], axis=1))
            except Exception as e:
                print(f"  Grad-CAM failed for cls {cls}: {e}")

        if not rows_vis:
            continue
        try:
            grid = np.concatenate(rows_vis, axis=0)
            out  = save_dir / f"gradcam_cls{cls:02d}_{model_name}.png"
            Image.fromarray(grid).save(out)
            print(f"  Grad-CAM grid saved → {out}")
        except Exception as e:
            print(f"  Warning: could not save Grad-CAM grid: {e}")

    del cam


# ==============================================================================
# SECTION 11E — Gap 18: DCGAN baseline
# ==============================================================================

class _DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128, img_size=64, n_classes=None):
        super().__init__()
        self.img_size = img_size
        # Optional class conditioning via embedding
        self.embed = nn.Embedding(n_classes, latent_dim) if n_classes else None
        in_dim = latent_dim * (2 if n_classes else 1)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64,   3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )  # output: (3, 64, 64)

    def forward(self, z, c=None):
        if self.embed is not None and c is not None:
            z = torch.cat([z, self.embed(c)], dim=1)
        return self.net(z.view(z.size(0), -1, 1, 1))


class _DCGANDiscriminator(nn.Module):
    def __init__(self, n_classes=None):
        super().__init__()
        # Class conditioning via label map projected to spatial channel
        self.embed = nn.Embedding(n_classes, 64 * 64) if n_classes else None
        in_ch = 4 if n_classes else 3
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64,  4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64,   128,  4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128,  256,  4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256,  512,  4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512,    1,  4, 1, 0, bias=False),
        )

    def forward(self, x, c=None):
        if self.embed is not None and c is not None:
            emb = self.embed(c).view(c.size(0), 1, 64, 64)
            x   = torch.cat([x, emb], dim=1)
        return self.net(x).view(-1)


def train_dcgan(train_csv: str, rare_classes=None,
                epochs: int = None, latent_dim: int = None,
                gan_lr: float = None, gan_batch: int = None) -> None:
    """
    Gap 18: Trains a conditional DCGAN per rare class.
    Checkpoints saved to GAN_DIR/cls_{c}/generator.pt.
    """
    if rare_classes is None: rare_classes = RARE_CLASSES
    epochs     = epochs     or args.gan_epochs
    latent_dim = latent_dim or args.gan_latent_dim
    gan_lr     = gan_lr     or args.gan_lr
    gan_batch  = gan_batch  or args.gan_batch_size

    transform = T.Compose([
        T.Resize((64, 64)), T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    criterion = nn.BCEWithLogitsLoss()

    df = pd.read_csv(train_csv)

    for cls in rare_classes:
        cls_df = df[df["label"] == cls]
        if len(cls_df) < 2:
            print(f"  DCGAN: class {cls} has <2 samples — skipping"); continue

        out_dir = GAN_DIR / f"cls_{cls}"; out_dir.mkdir(parents=True, exist_ok=True)
        gen_ckpt = out_dir / "generator.pt"
        if gen_ckpt.exists():
            print(f"  DCGAN class {cls}: checkpoint exists — skipping"); continue

        print(f"\n  DCGAN training class {cls} ({len(cls_df)} real images, {epochs} epochs)")

        # Build dataset from real images of this class
        class_imgs = []
        for _, row in cls_df.iterrows():
            try:
                img = Image.open(IMAGE_ROOT_DIR / row["image_path"]).convert("RGB")
                class_imgs.append(transform(img))
            except Exception: continue

        if len(class_imgs) < 2:
            print(f"  DCGAN class {cls}: could not load images"); continue

        ds     = torch.utils.data.TensorDataset(torch.stack(class_imgs))
        loader = DataLoader(ds, batch_size=min(gan_batch, len(class_imgs)),
                            shuffle=True, drop_last=False)

        G = _DCGANGenerator(latent_dim=latent_dim).to(DEVICE)
        D = _DCGANDiscriminator().to(DEVICE)
        opt_G = torch.optim.Adam(G.parameters(), lr=gan_lr, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(D.parameters(), lr=gan_lr, betas=(0.5, 0.999))

        G.train(); D.train()
        # Gap 18 extension — mode-collapse guard
        collapse_counter = 0
        collapsed        = False
        threshold  = getattr(args, "gan_collapse_threshold", 0.02)
        patience   = getattr(args, "gan_collapse_patience",  10)
        collapse_log = out_dir / "collapse.json"

        for ep in range(epochs):
            g_losses, d_losses = [], []
            for (real_imgs,) in loader:
                real_imgs = real_imgs.to(DEVICE)
                bs = real_imgs.size(0)
                real_lbl = torch.ones(bs, device=DEVICE)
                fake_lbl = torch.zeros(bs, device=DEVICE)

                # Train D
                z    = torch.randn(bs, latent_dim, device=DEVICE)
                fake = G(z).detach()
                d_loss = (criterion(D(real_imgs), real_lbl) +
                          criterion(D(fake),      fake_lbl)) * 0.5
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()

                # Train G
                z    = torch.randn(bs, latent_dim, device=DEVICE)
                fake = G(z)
                g_loss = criterion(D(fake), real_lbl)
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()

                g_losses.append(g_loss.item()); d_losses.append(d_loss.item())

            ep_d = float(np.mean(d_losses))
            ep_g = float(np.mean(g_losses))

            if (ep + 1) % 50 == 0 or ep == 0:
                print(f"    Ep {ep+1}/{epochs}  G={ep_g:.4f}  D={ep_d:.4f}")
                _log(ep, {"g_loss": ep_g, "d_loss": ep_d},
                     args.use_wandb, f"dcgan/cls{cls}/")

            # Mode-collapse detection: D_loss near 0 means D can't distinguish real/fake
            if ep_d < threshold:
                collapse_counter += 1
            else:
                collapse_counter = 0
            if collapse_counter >= patience:
                msg = (f"  DCGAN class {cls}: MODE COLLAPSE detected at epoch {ep+1} "
                       f"(D_loss={ep_d:.5f} < {threshold} for {patience} consecutive epochs). "
                       f"Saving partial checkpoint and skipping generation.")
                print(msg)
                with open(collapse_log, "w") as cf:
                    json.dump({"cls": cls, "epoch": ep + 1, "d_loss": ep_d,
                               "g_loss": ep_g, "status": "mode_collapse"}, cf, indent=2)
                collapsed = True
                break

        if not collapsed:
            torch.save(G.state_dict(), gen_ckpt)
            print(f"  DCGAN class {cls} generator saved → {gen_ckpt}")
        else:
            # Save partial state for inspection but mark as collapsed
            torch.save(G.state_dict(), out_dir / "generator_collapsed.pt")
        del G, D; torch.cuda.empty_cache()


def generate_synthetic_gan(rare_classes=None,
                            samples_per_class: int = None,
                            latent_dim: int = None) -> pd.DataFrame:
    """
    Gap 18: Loads saved DCGAN generators and synthesises images.
    Returns a DataFrame compatible with the augmented training CSV.
    """
    if rare_classes is None: rare_classes = RARE_CLASSES
    n_synth    = samples_per_class or args.samples_per_class
    latent_dim = latent_dim        or args.gan_latent_dim

    df_rows = []
    real_df = pd.read_csv(SPLITS_DIR / args.train_csv)
    l2n     = (dict(zip(real_df["label"].astype(int), real_df["class_name"]))
               if "class_name" in real_df.columns else {})
    gan_synth_dir = OUTPUT_DIR / "synthetic_gan"
    gan_synth_dir.mkdir(parents=True, exist_ok=True)

    for cls in rare_classes:
        gen_ckpt = GAN_DIR / f"cls_{cls}" / "generator.pt"
        if not gen_ckpt.exists():
            print(f"  GAN: no checkpoint for class {cls} — skipping"); continue

        G = _DCGANGenerator(latent_dim=latent_dim).to(DEVICE)
        G.load_state_dict(torch.load(gen_ckpt, map_location=DEVICE))
        G.eval()

        cls_name = l2n.get(cls, f"class_{cls}")
        cls_dir  = gan_synth_dir / str(cls); cls_dir.mkdir(parents=True, exist_ok=True)

        # Un-normalise: [-1,1] → [0,255]
        denorm = T.Compose([
            T.Normalize([-1.0]*3, [2.0]*3),  # x = (x - (-1)) / 2
        ])

        with torch.no_grad():
            for i in range(n_synth):
                z    = torch.randn(1, latent_dim, device=DEVICE)
                fake = G(z).squeeze(0).cpu()
                img  = T.ToPILImage()(fake.clamp(-1, 1) * 0.5 + 0.5)
                img  = img.resize((args.img_size, args.img_size), Image.LANCZOS)
                path = cls_dir / f"gan_{i:05d}.png"; img.save(path)
                df_rows.append({"image_path": str(path.relative_to(OUTPUT_DIR)),
                                 "label": cls, "class_name": cls_name, "source": "dcgan"})

        del G; torch.cuda.empty_cache()
        print(f"  GAN generated {n_synth} images for class {cls}")

    gan_csv = gan_synth_dir / "synthetic_train_gan.csv"
    out_df  = pd.DataFrame(df_rows)
    out_df.to_csv(gan_csv, index=False)
    print(f"  GAN synthetic CSV → {gan_csv}")
    return out_df


# ==============================================================================
# SECTION 11F — Gap 5: evaluate_on_test
# ==============================================================================

def evaluate_on_test(augmented: bool = False, test_csv_path=None) -> dict:
    """
    Gap 5: Evaluates best checkpoints on the held-out test set.
    Must be called only once — after all training is complete.
    """
    if test_csv_path is None:
        test_csv_path = SPLITS_DIR / args.test_csv

    print("\n" + "="*65)
    print(f"TEST SET EVALUATION  ({'augmented' if augmented else 'baseline'})")
    print("="*65)

    test_ds  = GastroVisionDataset(test_csv_path, "val", "classifier",
                                    synth_dir_name=args.synth_dir)
    test_ldr = DataLoader(test_ds, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    results  = {}

    for name in args.models:
        try:
            model = load_checkpoint(name, augmented)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}"); continue

        acc, yt, yp = _eval_acc(model, test_ldr)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
        )
        rare_f1 = float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean())
        print(f"\n  {name} TEST: acc={acc:.4f}  mean_f1={f1.mean():.4f}  rare_f1={rare_f1:.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))
        results[name] = {
            "acc": acc, "f1": f1.tolist(),
            "f1_mean": float(f1.mean()), "f1_rare": rare_f1,
        }
        _log(0, {"test_acc": acc, "test_f1_mean": float(f1.mean()), "test_f1_rare": rare_f1},
             args.use_wandb, f"{name}/test/")
        del model; torch.cuda.empty_cache()

    suffix = "_aug" if augmented else ""
    with open(RESULTS_DIR / f"test_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTest results saved → {RESULTS_DIR / f'test_results{suffix}.json'}")
    return results


# ==============================================================================
# SECTION 11G — Gap 19: Data-efficiency experiment
# ==============================================================================

def run_data_efficiency_experiment(model_name: str, train_csv: str, val_csv: str,
                                    rare_classes=None,
                                    sample_counts=None,
                                    synth_csv: str = None) -> dict:
    """
    Gap 19 (extended): Trains model with subsampled rare-class data.
    When synth_csv is provided, also runs S3 (real subsampled + SD synthetic)
    and produces a dual-line plot (S1 vs S3).
    Returns {n_samples → {"s1": {...}, "s3": {...}}} or {"s1": {...}} only.
    """
    if rare_classes   is None: rare_classes  = RARE_CLASSES
    if sample_counts  is None: sample_counts = args.efficiency_counts

    full_df = pd.read_csv(train_csv)
    val_df  = pd.read_csv(val_csv)
    results = {}

    for n in sorted(sample_counts):
        print(f"\n{'='*60}")
        print(f"Data efficiency: {n} samples per rare class")
        print(f"{'='*60}")

        # Subsample rare classes; keep all common class images
        sampled_rows = []
        for cls, grp in full_df.groupby("label"):
            if cls in rare_classes:
                take = min(n, len(grp))
                sampled_rows.append(grp.sample(take, random_state=args.seed))
            else:
                sampled_rows.append(grp)

        sub_df  = pd.concat(sampled_rows, ignore_index=True)
        sub_csv = SPLITS_DIR / f"_efficiency_{n}.csv"
        sub_df.to_csv(sub_csv, index=False)
        print(f"  Subsampled: {len(sub_df)} rows ({n} per rare class)")

        # Train a fresh model (shortened schedule)
        orig_cfg = {k: HPARAMS[model_name][k] for k in HPARAMS[model_name]}
        HPARAMS[model_name]["freeze_epochs"]    = max(2, orig_cfg["freeze_epochs"] // 3)
        HPARAMS[model_name]["fine_tune_epochs"] = max(4, orig_cfg["fine_tune_epochs"] // 3)

        try:
            model = get_model(model_name)
            cfg   = HPARAMS[model_name]
            crit  = FocalLoss(gamma=cfg["gamma"])
            scaler = GradScaler()
            train_ds = GastroVisionDataset(sub_csv, "train", "classifier")
            val_ds   = GastroVisionDataset(val_csv, "val",   "classifier")
            tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
            vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

            _freeze(model, model_name)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0))
            for _ in range(cfg["freeze_epochs"]):
                model.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    with autocast(): loss = crit(model(xb), yb)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                    scaler.step(opt); scaler.update()

            _unfreeze(model, model_name)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg.get("weight_decay", 0.01))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
            best_acc = 0.0; best_state = None
            for _ in range(cfg["fine_tune_epochs"]):
                model.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    with autocast(): loss = crit(model(xb), yb)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                sch.step()
                acc, _, _ = _eval_acc(model, vl)
                if acc > best_acc:
                    best_acc  = acc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Final evaluation
            if best_state:
                model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
            acc, yt, yp = _eval_acc(model, vl)
            _, _, f1, _ = precision_recall_fscore_support(
                yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
            )
            rare_f1 = float(f1[[c for c in rare_classes if c < NUM_CLASSES]].mean())
            s1_entry = {"acc": float(acc), "f1_mean": float(f1.mean()), "f1_rare": rare_f1}
            results[n] = {"s1": s1_entry}
            print(f"  S1  n={n}  acc={acc:.4f}  mean_f1={f1.mean():.4f}  rare_f1={rare_f1:.4f}")
            _log(n, s1_entry, args.use_wandb, f"efficiency/s1/{model_name}/")

            # Gap 19 extension — S3: also train on real subsampled + SD synthetic
            if synth_csv is not None:
                try:
                    synth_df_full = pd.read_csv(synth_csv)
                    # Filter synthetic to only the rare classes present in subsample
                    synth_rare = synth_df_full[synth_df_full["label"].isin(rare_classes)]
                    s3_df  = pd.concat([sub_df, synth_rare], ignore_index=True)
                    s3_csv = SPLITS_DIR / f"_efficiency_s3_{n}.csv"
                    s3_df.to_csv(s3_csv, index=False)

                    model_s3  = get_model(model_name)
                    cfg_s3    = HPARAMS[model_name]
                    crit_s3   = FocalLoss(gamma=cfg_s3["gamma"])
                    scaler_s3 = GradScaler()
                    tl_s3 = DataLoader(GastroVisionDataset(s3_csv, "train", "classifier"),
                                       batch_size=cfg_s3["batch_size"], shuffle=True, num_workers=2)
                    _freeze(model_s3, model_name)
                    opt_s3 = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model_s3.parameters()),
                        lr=cfg_s3["lr"] * cfg_s3.get("freeze_lr_mult", 10.0))
                    for _ in range(cfg_s3["freeze_epochs"]):
                        model_s3.train()
                        for xb, yb in tl_s3:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            opt_s3.zero_grad()
                            with autocast(): loss_s3 = crit_s3(model_s3(xb), yb)
                            scaler_s3.scale(loss_s3).backward(); scaler_s3.unscale_(opt_s3)
                            torch.nn.utils.clip_grad_norm_(
                                filter(lambda p: p.requires_grad, model_s3.parameters()), 1.0)
                            scaler_s3.step(opt_s3); scaler_s3.update()
                    _unfreeze(model_s3, model_name)
                    opt_s3 = torch.optim.AdamW(model_s3.parameters(), lr=cfg_s3["lr"],
                                               weight_decay=cfg_s3.get("weight_decay", 0.01))
                    sch_s3 = torch.optim.lr_scheduler.CosineAnnealingLR(
                        opt_s3, T_max=cfg_s3["fine_tune_epochs"])
                    best_s3 = 0.0; best_state_s3 = None
                    for _ in range(cfg_s3["fine_tune_epochs"]):
                        model_s3.train()
                        for xb, yb in tl_s3:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            opt_s3.zero_grad()
                            with autocast(): loss_s3 = crit_s3(model_s3(xb), yb)
                            scaler_s3.scale(loss_s3).backward(); scaler_s3.unscale_(opt_s3)
                            torch.nn.utils.clip_grad_norm_(model_s3.parameters(), 1.0)
                            scaler_s3.step(opt_s3); scaler_s3.update()
                        sch_s3.step()
                        a3, _, _ = _eval_acc(model_s3, vl)
                        if a3 > best_s3:
                            best_s3 = a3
                            best_state_s3 = {k: v.cpu().clone()
                                              for k, v in model_s3.state_dict().items()}
                    if best_state_s3:
                        model_s3.load_state_dict({k: v.to(DEVICE) for k, v in best_state_s3.items()})
                    a3, yt3, yp3 = _eval_acc(model_s3, vl)
                    _, _, f1_3, _ = precision_recall_fscore_support(
                        yt3, yp3, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
                    rf1_3 = float(f1_3[[c for c in rare_classes if c < NUM_CLASSES]].mean())
                    s3_entry = {"acc": float(a3), "f1_mean": float(f1_3.mean()), "f1_rare": rf1_3}
                    results[n]["s3"] = s3_entry
                    print(f"  S3  n={n}  acc={a3:.4f}  mean_f1={f1_3.mean():.4f}  rare_f1={rf1_3:.4f}")
                    _log(n, s3_entry, args.use_wandb, f"efficiency/s3/{model_name}/")
                    s3_csv.unlink(missing_ok=True)
                    del model_s3; torch.cuda.empty_cache()
                except Exception as e3:
                    print(f"  S3 efficiency n={n} failed: {e3}")
                    results[n]["s3"] = {"acc": 0.0, "f1_mean": 0.0, "f1_rare": 0.0}

        except Exception as e:
            print(f"  Data efficiency n={n} failed: {e}")
            results[n] = {"s1": {"acc": 0.0, "f1_mean": 0.0, "f1_rare": 0.0}}
        finally:
            HPARAMS[model_name].update(orig_cfg)
            sub_csv.unlink(missing_ok=True)
            try: del model; torch.cuda.empty_cache()
            except: pass

    # Save + dual-line plot (S1 vs S3)
    with open(RESULTS_DIR / f"data_efficiency_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    try:
        ns = sorted(results.keys())
        has_s3 = any("s3" in results[n] for n in ns)
        fig, ax = plt.subplots(figsize=(10, 5))
        # S1 lines
        ax.plot(ns, [results[n]["s1"]["f1_rare"]  for n in ns], "o-",  color="#d65f5f", lw=2,
                label="S1 Rare-class F1")
        ax.plot(ns, [results[n]["s1"]["f1_mean"]  for n in ns], "o--", color="#4878cf", lw=2,
                label="S1 Mean F1", alpha=0.7)
        if has_s3:
            ax.plot(ns, [results[n].get("s3", {}).get("f1_rare", 0) for n in ns],
                    "s-",  color="#e07b39", lw=2, label="S3 Rare-class F1")
            ax.plot(ns, [results[n].get("s3", {}).get("f1_mean", 0) for n in ns],
                    "s--", color="#6aa3d5", lw=2, label="S3 Mean F1", alpha=0.7)
        ax.set_xlabel("Training samples per rare class"); ax.set_ylabel("Score")
        ax.set_title(f"Data Efficiency — {model_name} {'(S1 vs S3)' if has_s3 else '(S1 only)'}")
        ax.set_xscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"data_efficiency_{model_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Data efficiency plot saved → {RESULTS_DIR / f'data_efficiency_{model_name}.png'}")
    except Exception as e:
        print(f"  Warning: could not save data efficiency plot: {e}")

    return results


# ==============================================================================
# SECTION 11H — Gap 20: LoRA rank ablation
# ==============================================================================

def ablate_lora_rank(train_csv: str, val_csv: str,
                     ranks=None, adapt_steps: int = None) -> dict:
    """
    Gap 20: Trains SD LoRA with each specified rank, generates a small
    validation batch per rare class, computes FID/KID, and returns results.
    ranks defaults to args.lora_ablation_ranks (e.g. [4, 8, 16, 32]).
    """
    if ranks       is None: ranks       = args.lora_ablation_ranks
    if adapt_steps is None: adapt_steps = args.lora_ablation_steps

    results = {}

    # Load shared components once
    tokenizer    = CLIPTokenizer.from_pretrained(args.sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.sd_model_id, subfolder="text_encoder").to(DEVICE)
    vae          = AutoencoderKL.from_pretrained(args.sd_model_id, subfolder="vae").to(DEVICE)
    noise_sched  = DDPMScheduler.from_pretrained(args.sd_model_id, subfolder="scheduler")
    text_encoder.requires_grad_(False); vae.requires_grad_(False)

    real_df    = pd.read_csv(train_csv)
    n_validate = 20   # generate this many images per class for FID/KID estimation

    for rank in ranks:
        print(f"\n{'='*60}")
        print(f"LoRA rank ablation: rank={rank}  steps={adapt_steps}")
        print(f"{'='*60}")

        rank_dir = CKPT_DIR / f"lora_rank_{rank}"; rank_dir.mkdir(exist_ok=True)
        adapter_path = rank_dir / "adapter"

        if not adapter_path.exists():
            unet = UNet2DConditionModel.from_pretrained(args.sd_model_id, subfolder="unet").to(DEVICE)
            lora_cfg = LoraConfig(
                r=rank, lora_alpha=rank * 2, lora_dropout=args.lora_dropout,
                target_modules=["to_q","to_k","to_v","to_out.0"],
            )
            unet = get_peft_model(unet, lora_cfg)

            dataset = GastroVisionSDDataset(train_csv, tokenizer)
            loader  = DataLoader(dataset, batch_size=args.sd_batch_size,
                                  shuffle=True, num_workers=2, drop_last=True)
            opt     = torch.optim.AdamW(unet.parameters(), lr=args.sd_lr)
            scaler  = GradScaler()
            unet.train(); opt.zero_grad()
            it = iter(loader); step = 0; rl = 0.0
            while step < adapt_steps:
                try: batch = next(it)
                except StopIteration: it = iter(loader); batch = next(it)
                pv = batch["pixel_values"].to(DEVICE)
                ii = batch["input_ids"].to(DEVICE)
                with torch.no_grad():
                    lat = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                    hs  = text_encoder(ii)[0]
                noise = torch.randn_like(lat)
                t     = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (lat.shape[0],), device=DEVICE).long()
                nl    = noise_sched.add_noise(lat, noise, t)
                with autocast():
                    pred = unet(nl, t, hs).sample
                    loss = F.mse_loss(pred, noise) / args.sd_grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % args.sd_grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad()
                rl += loss.item() * args.sd_grad_accum; step += 1
                if step % 200 == 0:
                    print(f"  rank={rank} step {step}/{adapt_steps} loss={rl/200:.4f}"); rl = 0.0

            unet.save_pretrained(adapter_path)
            del unet; torch.cuda.empty_cache(); gc.collect()

        # Generate a small validation set and compute FID/KID
        pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_model_id, torch_dtype=torch.float16, safety_checker=None
        ).to(DEVICE)
        pipe.unet = PeftModel.from_pretrained(pipe.unet, adapter_path)
        pipe.unet.eval(); pipe.enable_attention_slicing()

        val_synth_dir = rank_dir / "val_synth"; val_synth_dir.mkdir(exist_ok=True)
        synth_rows = []
        for cls in RARE_CLASSES:
            orig_lbl = REV_LABEL_MAP.get(cls, cls)
            prompt   = DOMAIN_PREFIX + CLASS_PROMPTS.get(orig_lbl, "endoscopy photo")
            gen_dir  = val_synth_dir / str(cls); gen_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                imgs = pipe(prompt=[prompt]*n_validate, negative_prompt=[NEGATIVE_PROMPT]*n_validate,
                            num_inference_steps=20, guidance_scale=args.guidance_scale,
                            height=512, width=512).images
            for i, img in enumerate(imgs):
                p = gen_dir / f"val_{i:04d}.png"
                img.resize((args.img_size, args.img_size), Image.LANCZOS).save(p)
                synth_rows.append({"image_path": str(p.relative_to(OUTPUT_DIR)),
                                    "label": cls, "class_name": f"cls_{cls}"})
        del pipe; torch.cuda.empty_cache(); gc.collect()

        synth_df = pd.DataFrame(synth_rows)
        fid, kid = compute_fid(real_df, synth_df)
        results[rank] = {"fid": fid, "kid": kid, "adapt_steps": adapt_steps}
        print(f"  rank={rank}  FID={fid:.2f if fid else 'N/A'}  KID×1000={kid:.3f if kid else 'N/A'}")
        _log(rank, {"fid": fid or -1, "kid": kid or -1}, args.use_wandb, "lora_ablation/")

    del text_encoder, vae, tokenizer; torch.cuda.empty_cache()

    with open(RESULTS_DIR / "lora_rank_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    try:
        valid = {r: v for r, v in results.items() if v["kid"] is not None}
        if valid:
            rs = sorted(valid.keys()); kids = [valid[r]["kid"] for r in rs]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(rs, kids, "o-", color="#4878cf", lw=2, markersize=8)
            for r, k in zip(rs, kids):
                ax.annotate(f"{k:.3f}", (r, k), textcoords="offset points", xytext=(0, 8), ha="center")
            ax.set_xlabel("LoRA Rank"); ax.set_ylabel("KID × 1000 (lower is better)")
            ax.set_title("LoRA Rank Ablation — Generation Quality")
            ax.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(RESULTS_DIR / "lora_rank_ablation.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  LoRA ablation plot saved → {RESULTS_DIR / 'lora_rank_ablation.png'}")
    except Exception as e:
        print(f"  Warning: could not save ablation plot: {e}")

    return results


# ==============================================================================
# SECTION 11I — Evaluation helpers (unchanged core + new evaluate_heavy_aug)
# ==============================================================================

def evaluate_all(augmented=False):
    print("\n" + "="*65)
    print(f"Evaluation ({'augmented' if augmented else 'baseline'})")
    print("="*65)

    val_ds  = GastroVisionDataset(SPLITS_DIR / args.val_csv, "val", "classifier",
                                   synth_dir_name=args.synth_dir)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    results = {}

    for name in args.models:
        try:
            model = load_checkpoint(name, augmented)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}"); continue

        acc, yt, yp = _eval_acc(model, val_ldr)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
        )
        print(f"\n{name}: acc={acc:.4f}  mean_f1={f1.mean():.4f}")
        print(classification_report(yt, yp, digits=4, zero_division=0))
        results[name] = {"acc": acc, "f1": f1.tolist(), "f1_mean": float(f1.mean()),
                         "f1_rare": float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean())}
        _log(0, {"val_acc": acc, "val_f1_mean": float(f1.mean())},
             args.use_wandb, f"{name}/val/")
        del model; torch.cuda.empty_cache()

    suffix = "_aug" if augmented else ""
    if len(results) >= 2:
        try:
            ensemble = ConfidenceEnsemble(args.models, suffix=suffix)
            acc_e, yt_e, yp_e, pr_e = eval_ensemble(ensemble, val_ldr)
            p_e, r_e, f1_e, _ = precision_recall_fscore_support(
                yt_e, yp_e, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
            )
            print(f"\nEnsemble: acc={acc_e:.4f}  mean_f1={f1_e.mean():.4f}")
            results["ensemble"] = {
                "acc": acc_e, "f1": f1_e.tolist(),
                "f1_mean": float(f1_e.mean()),
                "f1_rare": float(f1_e[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean()),
                "n_models": len(ensemble.models),
                "models":   list(ensemble.models.keys()),
            }
            cm = confusion_matrix(yt_e, yp_e)
            fig, ax = plt.subplots(figsize=(16, 14))
            sns.heatmap(cm.astype(float)/(cm.sum(axis=1,keepdims=True)+1e-8),
                        annot=True, fmt=".2f", cmap="Blues", ax=ax)
            ax.set_title(f"Ensemble — normalised CM ({'aug' if augmented else 'baseline'})")
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / f"confusion_matrix_ensemble{suffix}.png",
                        dpi=150, bbox_inches="tight"); plt.close()
            del ensemble; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Ensemble failed: {e}")

    if results:
        fig, ax = plt.subplots(figsize=(max(12, NUM_CLASSES*2), 5))
        x = np.arange(NUM_CLASSES); w = 0.8 / max(len(results), 1)
        cols = ["#4878cf","#6acc65","#d65f5f","#f0a500","#b47cc7"]
        for i, (nm, res) in enumerate(results.items()):
            ax.bar(x+i*w, res["f1"], w, label=nm, color=cols[i%len(cols)], alpha=0.85)
        for cls in RARE_CLASSES:
            if cls < NUM_CLASSES: ax.axvspan(cls-0.4, cls+0.4, alpha=0.07, color="yellow")
        ax.set_xlabel("Class"); ax.set_ylabel("F1")
        ax.set_title(f"Per-class F1 — {'augmented' if augmented else 'baseline'} (yellow=rare)")
        ax.legend(); ax.set_ylim(0, 1.15); plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"f1_per_class{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

    if augmented:
        synth_csv = SYNTH_DIR / "synthetic_train.csv"
        if synth_csv.exists():
            real_df  = pd.read_csv(SPLITS_DIR / args.train_csv)
            synth_df = pd.read_csv(synth_csv)
            fid, kid = compute_fid(real_df, synth_df)
            results["_fid_pooled"] = fid; results["_kid_pooled"] = kid

    with open(RESULTS_DIR / f"eval_results{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_DIR / f'eval_results{suffix}.json'}")
    print(f"\n{'Model':<35} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}")
    print("-"*60)
    for nm, res in results.items():
        if nm.startswith("_"): continue
        print(f"  {nm:<33} {res['acc']:>8.4f}  {res['f1_mean']:>8.4f}  {res['f1_rare']:>8.4f}")
    return results


def evaluate_heavy_aug():
    print("\n" + "="*65 + "\nEvaluation (S2: heavy traditional augmentation)\n" + "="*65)
    val_ds  = GastroVisionDataset(SPLITS_DIR / args.val_csv, "val", "classifier",
                                   synth_dir_name=args.synth_dir)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    results = {}
    for name in args.models:
        ckpt = CKPT_DIR / f"sota_{name}_heavy.pt"
        if not ckpt.exists():
            print(f"  Skipping {name}: no heavy aug checkpoint"); continue
        try:
            model = get_model(name)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE)); model.eval()
        except Exception as e:
            print(f"  Skipping {name}: {e}"); continue
        acc, yt, yp = _eval_acc(model, val_ldr)
        _, _, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
        )
        print(f"\n{name} (heavy aug): acc={acc:.4f}  mean_f1={f1.mean():.4f}")
        results[name] = {"acc": acc, "f1": f1.tolist(), "f1_mean": float(f1.mean()),
                         "f1_rare": float(f1[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean())}
        del model; torch.cuda.empty_cache()

    if len(results) >= 2:
        try:
            ensemble = ConfidenceEnsemble(args.models, suffix="_heavy")
            acc_e, yt_e, yp_e, _ = eval_ensemble(ensemble, val_ldr)
            _, _, f1_e, _ = precision_recall_fscore_support(
                yt_e, yp_e, labels=list(range(NUM_CLASSES)), average=None, zero_division=0
            )
            results["ensemble"] = {"acc": acc_e, "f1": f1_e.tolist(),
                                    "f1_mean": float(f1_e.mean()),
                                    "f1_rare": float(f1_e[[c for c in RARE_CLASSES if c < NUM_CLASSES]].mean()),
                                    "n_models": len(ensemble.models)}
            del ensemble; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Heavy aug ensemble failed: {e}")

    with open(RESULTS_DIR / "eval_results_heavy.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


# ==============================================================================
# SECTION 12A — Gap 21/25/26 helper functions
# ==============================================================================

def _set_global_seed(seed: int) -> None:
    """Re-seed all RNGs for a reproducible run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_seed_results(seed_results: dict) -> dict:
    """
    Gap 21: Given {seed → {strategy → {model → {metric → value}}}},
    compute mean ± std across seeds for every metric.
    Saves aggregate JSON to RESULTS_DIR/seed_aggregate.json.
    Returns the aggregate dict.
    """
    strategies = list(next(iter(seed_results.values())).keys())
    aggregate  = {}
    for strat in strategies:
        aggregate[strat] = {}
        # Collect all model names across seeds
        model_names = set()
        for sr in seed_results.values():
            if strat in sr:
                model_names.update(sr[strat].keys())
        for model in model_names:
            all_metrics: dict = {}
            for sr in seed_results.values():
                if strat not in sr or model not in sr[strat]: continue
                for metric, val in sr[strat][model].items():
                    if isinstance(val, (int, float)):
                        all_metrics.setdefault(metric, []).append(val)
            if not all_metrics: continue
            aggregate[strat][model] = {
                metric: {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                    "n":    len(vals),
                }
                for metric, vals in all_metrics.items()
            }
    with open(RESULTS_DIR / "seed_aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n{'='*65}\nSEED AGGREGATE ({len(seed_results)} seeds)\n{'='*65}")
    for strat, models in aggregate.items():
        print(f"\n  [{strat}]")
        print(f"  {'Model':<33} {'Acc mean±std':>16}  {'Mean F1 mean±std':>18}  {'Rare F1 mean±std':>18}")
        print("  " + "-"*88)
        for model, metrics in models.items():
            a   = metrics.get("acc",    {})
            mf  = metrics.get("f1_mean",{})
            rf  = metrics.get("f1_rare",{})
            print(f"  {model:<33} "
                  f"{a.get('mean',0):.4f}±{a.get('std',0):.4f}  "
                  f"{mf.get('mean',0):.4f}±{mf.get('std',0):.4f}  "
                  f"{rf.get('mean',0):.4f}±{rf.get('std',0):.4f}")
    return aggregate


def run_repeated_kfold_eval(model_name: str, full_csv: str,
                             rare_classes=None, k: int = 5,
                             n_repeats: int = 3) -> dict:
    """
    Gap 25: Repeated stratified k-fold evaluation (k × n_repeats total fits).
    Returns summary dict with mean ± std across all folds × repeats.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold
    if rare_classes is None: rare_classes = RARE_CLASSES

    df     = pd.read_csv(full_csv)
    labels = df["label"].astype(int).tolist()
    cfg    = HPARAMS[model_name]

    rskf   = RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats,
                                      random_state=args.seed)
    fold_metrics: dict = {"acc": [], "f1_mean": [], "f1_rare": []}
    raw_folds: dict    = {c: {"f1": []} for c in range(NUM_CLASSES)}

    for rep_fold, (train_idx, val_idx) in enumerate(rskf.split(df, labels)):
        print(f"\n  [CV] fold {rep_fold+1}/{k*n_repeats}  "
              f"(model={model_name})")
        train_sub = df.iloc[train_idx].reset_index(drop=True)
        val_sub   = df.iloc[val_idx].reset_index(drop=True)
        tmp_tr = SPLITS_DIR / f"_cv_train_{rep_fold}.csv"
        tmp_vl = SPLITS_DIR / f"_cv_val_{rep_fold}.csv"
        train_sub.to_csv(tmp_tr, index=False)
        val_sub.to_csv(  tmp_vl, index=False)

        try:
            model  = get_model(model_name)
            scaler = GradScaler()
            counts = np.bincount(train_sub["label"].astype(int).tolist(),
                                  minlength=NUM_CLASSES).tolist()
            crit   = get_criterion(args.loss_type, counts, gamma=cfg["gamma"])
            train_ds = GastroVisionDataset(tmp_tr, "train", "classifier")
            val_ds   = GastroVisionDataset(tmp_vl, "val",   "classifier")
            tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
            vl = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

            _freeze(model, model_name)
            opt = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["lr"] * cfg.get("freeze_lr_mult", 10.0))
            for _ in range(cfg["freeze_epochs"]):
                model.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    with autocast(): loss = crit(model(xb), yb)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                    scaler.step(opt); scaler.update()

            _unfreeze(model, model_name)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg.get("weight_decay", 0.01))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["fine_tune_epochs"])
            best_acc = 0.0; best_state = None
            for _ in range(cfg["fine_tune_epochs"]):
                model.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad()
                    with autocast(): loss = crit(model(xb), yb)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                sch.step()
                a, _, _ = _eval_acc(model, vl)
                if a > best_acc:
                    best_acc  = a
                    best_state = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}

            if best_state:
                model.load_state_dict({k2: v.to(DEVICE) for k2, v in best_state.items()})
            acc, yt, yp = _eval_acc(model, vl)
            _, _, f1, _ = precision_recall_fscore_support(
                yt, yp, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
            rare_f1 = float(f1[[c for c in rare_classes if c < NUM_CLASSES]].mean())
            fold_metrics["acc"].append(float(acc))
            fold_metrics["f1_mean"].append(float(f1.mean()))
            fold_metrics["f1_rare"].append(rare_f1)
            for c in range(NUM_CLASSES):
                raw_folds[c]["f1"].append(float(f1[c]) if c < len(f1) else 0.0)
            print(f"    acc={acc:.4f}  mean_f1={f1.mean():.4f}  rare_f1={rare_f1:.4f}")
        except Exception as e:
            print(f"  [CV] fold {rep_fold+1} failed: {e}")
        finally:
            tmp_tr.unlink(missing_ok=True); tmp_vl.unlink(missing_ok=True)
            try: del model; torch.cuda.empty_cache()
            except: pass

    summary = {
        metric: {
            "mean": float(np.mean(vals)) if vals else 0.0,
            "std":  float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "folds": vals,
        }
        for metric, vals in fold_metrics.items()
    }
    out_path = RESULTS_DIR / f"cv_results_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw_folds": raw_folds}, f, indent=2)
    print(f"\n  CV {k}×{n_repeats}: acc={summary['acc']['mean']:.4f}±{summary['acc']['std']:.4f}  "
          f"rare_f1={summary['f1_rare']['mean']:.4f}±{summary['f1_rare']['std']:.4f}")
    return summary, raw_folds


# ── Gap 26 — Confusion cost analysis ──────────────────────────────────────────

_DEFAULT_COST_MATRIX = {
    # (true_class, pred_class): cost
    # Medical priority: misclassifying ulcer (26) as normal stomach (20) = 10
    # misclassifying colorectal cancer (7) as normal colon (19) = 10
    # misclassifying esophageal varices (12) as normal esophagus (18) = 8
    # All other errors default to 1
}
_DEFAULT_COST_DEFAULTS = {
    "26:20": 10.0, "7:19": 10.0, "12:18": 8.0,
    "3:20":  7.0,  "7:4":  6.0,  "26:20": 10.0,
}


def load_cost_matrix() -> dict:
    """
    Gap 26: Returns {(true, pred): cost} dict.
    Uses args.cost_matrix_path if provided, else hardcoded medical default.
    """
    if args.cost_matrix_path:
        try:
            with open(args.cost_matrix_path) as f:
                raw = json.load(f)
            cost_matrix = {}
            for key, cost in raw.items():
                t, p = map(int, key.split(":"))
                cost_matrix[(t, p)] = float(cost)
            print(f"  Cost matrix loaded from {args.cost_matrix_path} "
                  f"({len(cost_matrix)} entries)")
            return cost_matrix
        except Exception as e:
            print(f"  Warning: could not load cost matrix: {e}. Using default.")

    # Hardcoded medical default
    cost_matrix = {}
    for key, cost in _DEFAULT_COST_DEFAULTS.items():
        t, p = map(int, key.split(":"))
        cost_matrix[(t, p)] = cost
    print(f"  Using hardcoded medical cost matrix ({len(cost_matrix)} high-cost pairs).")
    return cost_matrix


def compute_confusion_cost(y_true, y_pred, cost_matrix: dict,
                            default_error_cost: float = 1.0) -> dict:
    """
    Gap 26: Computes total weighted misclassification cost.
    Returns {"total_cost", "n_errors", "mean_cost_per_error", "cost_breakdown"}.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    total_cost    = 0.0
    cost_breakdown: dict = {}
    n_errors      = 0
    for t, p in zip(y_true, y_pred):
        if t == p: continue
        n_errors += 1
        cost = cost_matrix.get((int(t), int(p)), default_error_cost)
        total_cost += cost
        key = f"{int(t)}:{int(p)}"
        cost_breakdown[key] = cost_breakdown.get(key, 0.0) + cost
    return {
        "total_cost":           round(total_cost, 4),
        "n_errors":             n_errors,
        "mean_cost_per_error":  round(total_cost / max(n_errors, 1), 4),
        "cost_breakdown":       cost_breakdown,
    }


# ==============================================================================
# SECTION 12 — Main
# ==============================================================================

def main():
    global NUM_CLASSES, RARE_CLASSES

    print("=" * 65 + "\nGastroVision DDPM Augmentation Pipeline\n" + "=" * 65)
    for k, v in [("data_dir", DATA_DIR), ("output_dir", OUTPUT_DIR),
                  ("models", args.models), ("lora_rank", args.lora_rank),
                  ("samples_per_class", args.samples_per_class),
                  ("synthetic_ratio", args.synthetic_ratio),
                  ("loss_type", args.loss_type),
                  ("seeds", args.seeds),
                  ("single_split", args.single_split),
                  ("use_wandb", args.use_wandb)]:
        print(f"  {k:<22} {v}")

    if args.use_wandb:
        _init_wandb("pipeline_main")

    # ── Step 1: Create / load data splits (done once, seed-independent) ────────
    train_csv = SPLITS_DIR / args.train_csv
    val_csv   = SPLITS_DIR / args.val_csv
    test_csv  = SPLITS_DIR / args.test_csv

    if not train_csv.exists():
        print("Creating splits...")
        train_df, val_df, test_df, rare = create_splits()
        RARE_CLASSES = rare
    else:
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        test_df  = pd.read_csv(test_csv)
        if "original_label" in train_df.columns:
            original_labels = sorted(
                set(train_df["original_label"].unique()) |
                set(val_df["original_label"].unique()) |
                set(test_df["original_label"].unique())
            )
            rare_orig = [c for c in original_labels
                         if len(train_df[train_df["original_label"] == c]) < 30]
        else:
            original_labels = sorted(train_df["label"].unique())
            rare_orig = [c for c in original_labels
                         if len(train_df[train_df["label"] == c]) < 30]
        global LABEL_MAP, REV_LABEL_MAP
        LABEL_MAP     = {orig: i for i, orig in enumerate(original_labels)}
        REV_LABEL_MAP = {i: orig for orig, i in LABEL_MAP.items()}
        NUM_CLASSES   = len(original_labels)
        need_remap    = False
        if "original_label" in train_df.columns:
            train_df["label"] = train_df["original_label"].map(LABEL_MAP)
            val_df["label"]   = val_df["original_label"].map(LABEL_MAP)
            test_df["label"]  = test_df["original_label"].map(LABEL_MAP)
            need_remap = True
        else:
            current = sorted(train_df["label"].unique())
            if current != list(range(NUM_CLASSES)):
                train_df["label"] = train_df["label"].map(LABEL_MAP)
                val_df["label"]   = val_df["label"].map(LABEL_MAP)
                test_df["label"]  = test_df["label"].map(LABEL_MAP)
                need_remap = True
        if need_remap:
            train_df.to_csv(train_csv, index=False); val_df.to_csv(val_csv, index=False)
            test_df.to_csv(test_csv, index=False)
        RARE_CLASSES = sorted([LABEL_MAP[c] for c in rare_orig])
        print(f"Loaded splits. RARE_CLASSES={RARE_CLASSES}")
    print(f"NUM_CLASSES={NUM_CLASSES}")

    if args.evaluate_only:
        evaluate_all(augmented=False)
        evaluate_heavy_aug()
        evaluate_all(augmented=True)
        evaluate_on_test(augmented=False)
        evaluate_on_test(augmented=True)
        return

    # ── Load best hyperparams if available ──────────────────────────────────────
    hparams_path = OUTPUT_DIR / "best_hparams.json"
    if hparams_path.exists():
        with open(hparams_path) as f: saved = json.load(f)
        for k, v in saved.items():
            if k in HPARAMS: HPARAMS[k].update(v)

    # ── Cost matrix (Gap 26) — loaded once, reused across seeds ─────────────
    cost_matrix = load_cost_matrix()

    # ══════════════════════════════════════════════════════════════════════════
    # Gap 21 — Multi-seed outer loop
    # ══════════════════════════════════════════════════════════════════════════
    seed_results: dict = {}   # {seed: {"s1": {model: metrics}, "s2": ..., "s3": ...}}

    for seed in args.seeds:
        print(f"\n{'#'*65}\n# SEED {seed}  ({args.seeds.index(seed)+1}/{len(args.seeds)})\n{'#'*65}")
        _set_global_seed(seed)

        # ── Step 2: Train S1 (real only) ──────────────────────────────────────
        if not args.skip_training:
            print("\n" + "="*65 + f"\n[seed={seed}] Step 2: S1 — real-data training\n" + "="*65)
            for name in args.models:
                if args.tune:
                    tune_classifier(name, train_csv, val_csv,
                                     n_trials=args.tune_trials, tune_epochs=args.tune_epochs)
                train_classifier(name, train_csv, val_csv, augmented=False)

        # ── Step 2B: Train S2 (heavy aug) ─────────────────────────────────────
        if not args.skip_training:
            print("\n" + "="*65 + f"\n[seed={seed}] Step 2B: S2 — heavy augmentation\n" + "="*65)
            for name in args.models:
                train_classifier_heavy_aug(name, train_csv, val_csv)

        # ── Step 2C: Train S4 (MixUp / CutMix) ─────────────────────────────
        if not args.skip_training:
            print("\n" + "="*65 + f"\n[seed={seed}] Step 2C: S4 — MixUp / CutMix\n" + "="*65)
            for name in args.models:
                for aug_mode in ["mixup", "cutmix"]:
                    train_classifier_mixup(name, train_csv, val_csv, mode=aug_mode)

        # ── Step 3: SD domain adaptation (seed-independent; skip if done) ─────
        if not args.skip_domain_adapt:
            ema_adapter = CKPT_DIR / "sd_gastrovision_lora_ema_adapter"
            if ema_adapter.exists():
                print(f"\n[seed={seed}] Step 3: EMA adapter exists — skipping")
            else:
                print("\n" + "="*65 + f"\n[seed={seed}] Step 3: SD LoRA domain adaptation\n" + "="*65)
                domain_adapt_sd()

        # ── Step 3B: LoRA rank ablation (Gap 20) ───────────────────────────
        if args.run_lora_ablation and seed == args.seeds[0]:
            # Run only once (not seed-dependent)
            print("\n" + "="*65 + "\nStep 3B: LoRA rank ablation (Gap 20)\n" + "="*65)
            ablate_lora_rank(train_csv, val_csv)

        # ── Step 4: Generate synthetic images for S3 ───────────────────────
        synth_csv_path = SYNTH_DIR / "synthetic_train.csv"
        if not args.skip_generation:
            already_done = (RARE_CLASSES and all(
                len(list((SYNTH_DIR / str(cls)).glob("synth_*.png"))) >= args.samples_per_class
                for cls in RARE_CLASSES
            ))
            if already_done and synth_csv_path.exists():
                print(f"\n[seed={seed}] Step 4: Generation complete — skipping")
                synth_df = pd.read_csv(synth_csv_path)
            else:
                print("\n" + "="*65 + f"\n[seed={seed}] Step 4: SD synthetic generation\n" + "="*65)
                synth_df = generate_synthetic(real_df=train_df)
        else:
            synth_df = pd.read_csv(synth_csv_path) if synth_csv_path.exists() else None

        # ── Step 4B: GAN baseline (Gap 18 extension → S5) ─────────────────
        gan_synth_df = None
        if args.run_gan_baseline:
            print("\n" + "="*65 + f"\n[seed={seed}] Step 4B: DCGAN baseline (Gap 18)\n" + "="*65)
            train_dcgan(str(train_csv), rare_classes=RARE_CLASSES)
            gan_synth_df = generate_synthetic_gan(rare_classes=RARE_CLASSES)

        # ── Step 5: Build augmented CSVs and retrain S3 (+ S5 if GAN ok) ──
        aug_csv = SPLITS_DIR / args.aug_train_csv
        if not args.skip_training and synth_df is not None:
            print("\n" + "="*65 + f"\n[seed={seed}] Step 5: S3 augmented retraining\n" + "="*65)
            def _norm(p, base=None): p = Path(p); return p.resolve() if p.is_absolute() else p
            val_abs   = set(_norm(p) for p in pd.read_csv(val_csv)["image_path"])
            test_abs  = set(_norm(p) for p in pd.read_csv(test_csv)["image_path"])
            synth_abs = set(_norm(p) for p in synth_df["image_path"])
            overlap   = synth_abs & (val_abs | test_abs)
            if overlap:
                print(f"  Leakage detected — removing {len(overlap)} images")
                synth_df = synth_df[~synth_df["image_path"].isin([str(p) for p in overlap])]
            else:
                print("  No leakage detected")
            aug_df = pd.concat([train_df[["image_path","label","class_name"]], synth_df],
                                ignore_index=True)
            aug_df.to_csv(aug_csv, index=False)
            print(f"  Augmented: {len(train_df)} real + {len(synth_df)} synthetic = {len(aug_df)}")
            for name in args.models:
                train_classifier(name, aug_csv, val_csv, augmented=True)

            # S5: real + GAN synthetic (Gap 18 extension)
            if gan_synth_df is not None and not gan_synth_df.empty:
                s5_df = pd.concat([train_df[["image_path","label","class_name"]],
                                   gan_synth_df], ignore_index=True)
                s5_csv = SPLITS_DIR / "train_s5_gan.csv"
                s5_df.to_csv(s5_csv, index=False)
                print(f"\n  S5: {len(train_df)} real + {len(gan_synth_df)} GAN = {len(s5_df)}")
                for name in args.models:
                    # reuse train_classifier with a dedicated ckpt name
                    orig_aug_name = args.aug_train_csv
                    args.aug_train_csv = "train_s5_gan.csv"
                    train_classifier(name, s5_csv, val_csv, augmented=True)
                    args.aug_train_csv = orig_aug_name

        # ── Step 6 / Gap 25: Evaluation (CV or single-split) ───────────────
        print("\n" + "="*65 + f"\n[seed={seed}] Step 6: Evaluation\n" + "="*65)

        if args.single_split:
            # Original fixed-split evaluation
            s1_res = evaluate_all(augmented=False)
            s2_res = evaluate_heavy_aug() if any(
                (CKPT_DIR / f"sota_{n}_heavy.pt").exists() for n in args.models) else {}
            s3_res = evaluate_all(augmented=True) if (
                not args.skip_training and synth_df is not None) else {}
        else:
            # Gap 25 — CV as default: 5-fold × cv_repeats (using full train CSV)
            print(f"  Running {args.kfold_splits}-fold × {args.cv_repeats} repeated CV per model …")
            full_for_cv = train_csv   # CV uses full training CSV; folds provide the val split
            s1_res = {}; s2_res = {}; s3_res = {}
            for name in args.models:
                print(f"\n  === CV for {name} (S1) ===")
                try:
                    cv_sum, _ = run_repeated_kfold_eval(
                        name, str(full_for_cv), RARE_CLASSES,
                        k=args.kfold_splits, n_repeats=args.cv_repeats)
                    s1_res[name] = {
                        "acc":    cv_sum["acc"]["mean"],
                        "f1_mean":cv_sum["f1_mean"]["mean"],
                        "f1_rare":cv_sum["f1_rare"]["mean"],
                        "acc_std":    cv_sum["acc"]["std"],
                        "f1_mean_std":cv_sum["f1_mean"]["std"],
                        "f1_rare_std":cv_sum["f1_rare"]["std"],
                    }
                except Exception as e:
                    print(f"  CV failed for {name}: {e}")
            # Save CV summary
            with open(RESULTS_DIR / f"cv_s1_seed{seed}.json", "w") as f:
                json.dump(s1_res, f, indent=2)

        # ── Step 7: Held-out test set (Gap 5) ─────────────────────────────
        print("\n" + "="*65 + f"\n[seed={seed}] Step 7: Held-out test evaluation\n" + "="*65)
        test_s1 = evaluate_on_test(augmented=False)
        test_s3 = {}
        if not args.skip_training and synth_df is not None:
            test_s3 = evaluate_on_test(augmented=True)

        # ── Step 7B: Confusion cost analysis (Gap 26) ────────────────────
        print(f"\n[seed={seed}] Step 7B: Confusion cost analysis (Gap 26)")
        cost_results: dict = {}
        val_ds_eval  = GastroVisionDataset(val_csv, "val", "classifier",
                                            synth_dir_name=args.synth_dir)
        val_ldr_eval = DataLoader(val_ds_eval, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4, pin_memory=True)
        for name in args.models:
            for suffix, label in [("", "S1"), ("_aug", "S3"), ("_heavy", "S2")]:
                ckpt = CKPT_DIR / f"sota_{name}{suffix}.pt"
                if not ckpt.exists(): continue
                try:
                    m = get_model(name)
                    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
                    _, yt, yp = _eval_acc(m, val_ldr_eval)
                    cost_info = compute_confusion_cost(yt, yp, cost_matrix)
                    cost_results[f"{name}{suffix}"] = {"strategy": label, **cost_info}
                    print(f"  {label} {name}: total_cost={cost_info['total_cost']:.1f}  "
                          f"mean_per_error={cost_info['mean_cost_per_error']:.3f}")
                    del m; torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  Cost analysis {name}{suffix}: {e}")
        with open(RESULTS_DIR / f"confusion_cost_seed{seed}.json", "w") as f:
            json.dump(cost_results, f, indent=2)

        # ── Collect per-seed results ───────────────────────────────────────
        seed_results[seed] = {
            "s1": test_s1 or s1_res,
            "s2": s2_res,
            "s3": test_s3 or s3_res,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Post-seed: aggregate mean±std (Gap 21)
    # ══════════════════════════════════════════════════════════════════════════
    if len(args.seeds) > 1:
        _aggregate_seed_results(seed_results)

    # ── Step 8: Grad-CAM — Gap 3 (once, after all seeds) ──────────────────
    if args.run_gradcam and GRADCAM_AVAILABLE:
        print("\n" + "="*65 + "\nStep 8: Grad-CAM visualisation (Gap 3)\n" + "="*65)
        _set_global_seed(args.seeds[0])
        val_ds_g  = GastroVisionDataset(val_csv, "val", "classifier")
        val_ldr_g = DataLoader(val_ds_g, batch_size=args.batch_size, shuffle=False, num_workers=2)
        for name in args.models:
            try:
                model = load_checkpoint(name, augmented=False)
                generate_gradcam_grid(model, name, val_ldr_g, RARE_CLASSES,
                                       RESULTS_DIR / "gradcam")
                del model; torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Grad-CAM {name} failed: {e}")

    # ── Step 9: Data-efficiency curve — Gap 19 extension (S1 + S3) ────────
    if args.run_data_efficiency:
        print("\n" + "="*65 + "\nStep 9: Data-efficiency curve (Gap 19)\n" + "="*65)
        _set_global_seed(args.seeds[0])
        synth_csv_for_eff = str(synth_csv_path) if synth_csv_path.exists() else None
        run_data_efficiency_experiment(
            args.efficiency_model, str(train_csv), str(val_csv),
            rare_classes=RARE_CLASSES, sample_counts=args.efficiency_counts,
            synth_csv=synth_csv_for_eff,
        )

    # ── Step 10: Strategy comparison summary + LaTeX table ────────────────
    print("\n" + "="*65 + "\nSTRATEGY COMPARISON SUMMARY\n" + "="*65)
    try:
        def _safe_load(path):
            return json.load(open(path)) if Path(path).exists() else {}

        s1 = _safe_load(RESULTS_DIR / "test_results.json")
        s2 = _safe_load(RESULTS_DIR / "eval_results_heavy.json")
        s3 = _safe_load(RESULTS_DIR / "test_results_aug.json")

        print(f"\n{'Strategy':<22} {'Model':<33} {'Acc':>8}  {'Mean F1':>8}  {'Rare F1':>8}")
        print("-"*82)
        for strategy, data in [("S1: Real only", s1), ("S2: Heavy aug", s2),
                                ("S3: SD synthetic", s3)]:
            if not data: continue
            for nm, res in data.items():
                if nm.startswith("_"): continue
                marker = "  ◄" if nm == "ensemble" else ""
                print(f"  {strategy:<20} {nm:<33} {res['acc']:>8.4f}  "
                      f"{res['f1_mean']:>8.4f}  {res['f1_rare']:>8.4f}{marker}")
            print()

        # LaTeX table — Gap 15
        try:
            from evaluation import results_to_latex
            results_to_latex(s1, s2, s3, RESULTS_DIR / "table_results.tex",
                              rare_classes=RARE_CLASSES)
        except Exception as e:
            print(f"  LaTeX table: {e}")

        # Print confusion cost summary across all strategies
        print("\n  CONFUSION COST SUMMARY (lower = better)")
        print(f"  {'Model+Strategy':<40} {'Total cost':>12}  {'Mean/error':>12}")
        print("  " + "-"*66)
        all_cost_files = sorted(RESULTS_DIR.glob("confusion_cost_seed*.json"))
        merged_costs: dict = {}
        for cf in all_cost_files:
            try:
                cc = json.load(open(cf))
                for k2, v2 in cc.items():
                    merged_costs.setdefault(k2, []).append(v2.get("total_cost", 0))
            except Exception: pass
        for key, costs in merged_costs.items():
            print(f"  {key:<40} {np.mean(costs):>12.2f}  (n={len(costs)} seeds)")

    except Exception as e:
        print(f"  Could not print comparison table: {e}")

    if args.use_wandb and WANDB_AVAILABLE:
        try: wandb.finish()
        except: pass

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

"""
msfit training script for WBCBench 2026.

This file retains the broader experimental framework used during development,
but the publication-facing recipe is the multi-stage DINOBloom-base pipeline
invoked by `msfit/reproduce_best_submission.sh`.
"""

import os
# Must be set before any CUDA/torch import so the allocator picks it up
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import sys
import json
import random
import math
import warnings
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score
)

try:
    from .modeling import DinoBloomMultiScale, detect_pathology_fm
except ImportError:
    from modeling import DinoBloomMultiScale, detect_pathology_fm

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
NUM_CLASSES = 13
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for name, idx in LABEL_TO_IDX.items()}

# Morphology-defined confusion pairs: (true_class_idx, confuser_class_idx)
# These come from WBC biology, not from tuning on validation
DEFAULT_CONFUSION_PAIRS = [
    (2, 11, 0.3),   # BNE confused as SNE — band vs segmented nucleus
    (12, 4, 0.3),   # VLY confused as LY — reactive vs normal lymphocyte
    (5, 7, 0.25),   # MMY confused as MY — maturation stage neighbors
    (10, 7, 0.25),  # PMY confused as MY — maturation stage neighbors
    (7, 5, 0.2),    # MY confused as MMY — reverse direction (less common)
    (7, 10, 0.2),   # MY confused as PMY — reverse direction
]


def hf_login(token=None):
    """Login to HuggingFace for gated model access."""
    try:
        from huggingface_hub import login as hf_hub_login
        if token:
            hf_hub_login(token=token)
            print("  HuggingFace: logged in with provided token")
        else:
            try:
                from kaggle_secrets import UserSecretsClient
                secrets = UserSecretsClient()
                hf_token = secrets.get_secret("HF_TOKEN")
                hf_hub_login(token=hf_token)
                print("  HuggingFace: logged in via Kaggle secret")
            except Exception:
                env_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
                if env_token:
                    hf_hub_login(token=env_token)
                    print("  HuggingFace: logged in via env variable")
                else:
                    print("  WARNING: No HF token found. Gated models will fail.")
    except ImportError:
        print("  WARNING: huggingface_hub not installed")


# External dataset class mappings
MLL23_CLASS_MAP = {
    'basophil': 'BA', 'promyelocyte': 'PMY', 'myelocyte': 'MY',
    'metamyelocyte': 'MMY', 'neutrophil_band': 'BNE',
    'lymphocyte_reactive': 'VLY', 'lymphocyte_neoplastic': 'PLY', 'plasma_cell': 'PC',
}

MULTI_FOCUS_CLASS_MAP = {
    'basophil': 'BA', 'promyelocyte': 'PMY', 'myelocyte': 'MY',
    'metamyelocyte': 'MMY', 'band_neutrophil': 'BNE', 'abnormal_lymphocyte': 'VLY',
}

CHULA15_CLASS_MAP = {
    'BNE': 'BNE', 'Metamyelocyte': 'MMY', 'Myelocyte': 'MY',
    'Promyelocyte': 'PMY', 'Atypical Lymphocyte': 'VLY',
    'Lymphoblast': 'PLY', 'Basophil': 'BA', 'Eosinophil': 'EO',
}

KUOPTOFIL_CLASS_MAP = {
    'Band Neutrophil': 'BNE', 'Basophil': 'BA', 'Blast': 'BL',
    'Eosinophil': 'EO', 'Metamyelocyte': 'MMY',
    'Monocyte': 'MO', 'Myelocyte': 'MY', 'Reactive Lymphocyte': 'VLY',
}


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='WBC Classification Training')

    # Data paths
    parser.add_argument('--data-root', type=str, nargs='+',
                        default=['/kaggle/input/wbcbench-2026'],
                        help='WBCBench data root(s)')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working',
                        help='Output directory for checkpoints')

    # Pretrained model
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (e.g., V5 best model)')
    parser.add_argument('--backbone-only', action='store_true', default=False,
                        help='Load only backbone weights from checkpoint (ignore head). '
                             'Use for Step 2: load fine-tuned backbone, train fresh head.')
    parser.add_argument('--fresh-optimizer', action='store_true', default=False,
                        help='Do not resume optimizer momentum from checkpoint. '
                             'Use when changing data regime significantly.')

    # External data
    parser.add_argument('--external-mll23', type=str, default=None,
                        help='Path to MLL23 dataset')
    parser.add_argument('--external-multifocus', type=str, default=None,
                        help='Path to Multi-focus Korea dataset')
    parser.add_argument('--external-chula15', type=str, default=None,
                        help='Path to Chula-15 dataset')
    parser.add_argument('--external-kuoptofil', type=str, default=None,
                        help='Path to KU-Optofil PBC dataset root')
    parser.add_argument('--external-cellwiki', type=str, nargs='*', default=None,
                        help='Paths to CellWiki folders: PLY MMY PMY PMY_AML VLY BNE')
    parser.add_argument('--external-classes', type=str, nargs='*', default=None,
                        help='Only include these classes from external datasets (e.g., PLY BNE)')

    # Pseudo labels
    parser.add_argument('--pseudo-labels', type=str, default=None,
                        help='Path to pseudo labels CSV')
    parser.add_argument('--pseudo-min-confidence', type=float, default=0.8,
                        help='Flat confidence threshold for pseudo labels. '
                             'Set to 0.0 to include all pseudo labels (let focal loss handle imbalance). '
                             'Default 0.8 is a reasonable balance between coverage and label quality.')

    # Training data composition
    parser.add_argument('--use-eval-in-train', action='store_true', default=False,
                        help='Add eval set to training data (final submission mode — no validation)')

    # Model
    parser.add_argument('--backbone', type=str, default='dinobloom_base',
                        help='Backbone: dinobloom_base/large/giant, uni2-h, uni, virchow2')
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--multi-scale', action='store_true', default=True,
                        help='Use multi-scale CLS fusion (blocks 3,7,11)')
    parser.add_argument('--no-multi-scale', action='store_false', dest='multi_scale',
                        help='Disable multi-scale (V5-style single-scale)')
    parser.add_argument('--block-indices', type=int, nargs='+', default=[3, 7, 11],
                        help='Block indices for multi-scale (0-indexed)')
    parser.add_argument('--freeze-backbone', action='store_true', default=True)
    parser.add_argument('--no-freeze-backbone', action='store_false', dest='freeze_backbone')
    parser.add_argument('--unfreeze-last-n-blocks', type=int, default=0,
                        help='Partial backbone unfreeze: freeze all, then unfreeze last N transformer '
                             'blocks + final norm. 0 = disabled (use --freeze/--no-freeze-backbone). '
                             'For giant (40 blocks): 20 ≈ 566M params, 15 ≈ 425M params.')

    # Head
    parser.add_argument('--head', type=str, default='cosine', choices=['cosine', 'linear', 'mlp'])
    parser.add_argument('--cosine-scale', type=float, default=30.0)
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--mlp-hidden-dim', type=int, default=512)
    parser.add_argument('--mlp-dropout', type=float, default=0.20)

    # Loss
    parser.add_argument('--loss', type=str, default='ldam', choices=['ldam', 'ce', 'focal'])
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--ldam-max-margin', type=float, default=0.5)
    parser.add_argument('--ldam-scale', type=float, default=30.0)
    parser.add_argument('--confusion-loss', action='store_true', default=True,
                        help='Add pairwise confusion penalty to LDAM')
    parser.add_argument('--no-confusion-loss', action='store_false', dest='confusion_loss')
    parser.add_argument('--confusion-weight', type=float, default=0.5,
                        help='Weight for confusion penalty term')
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # DRW
    parser.add_argument('--drw', action='store_true', default=True)
    parser.add_argument('--no-drw', action='store_false', dest='drw')
    parser.add_argument('--drw-start', type=float, default=0.5,
                        help='Fraction of training before activating DRW')

    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5, help='Head learning rate')
    parser.add_argument('--lr-backbone', type=float, default=3e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--grad-checkpoint', action='store_true', default=False,
                        help='Enable gradient checkpointing on backbone (saves memory, slower)')

    # Sampler
    parser.add_argument('--sampler', type=str, default='sqrt',
                        choices=['none', 'sqrt', 'balanced', 'effective', 'tail_quota'])
    parser.add_argument('--tail-classes', type=str, nargs='+',
                        default=['PLY', 'BNE', 'VLY', 'PC', 'PMY', 'MMY', 'MY'],
                        help='Classes treated as tail for quota sampler')
    parser.add_argument('--tail-frac', type=float, default=0.30,
                        help='Fraction of each batch reserved for tail classes')

    # Augmentation
    parser.add_argument('--mixup-prob', type=float, default=0.3)
    parser.add_argument('--mixup-alpha', type=float, default=0.3)
    parser.add_argument('--cutmix-prob', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)

    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Prefer deterministic CUDA/cuDNN execution for more repeatable reruns. '
                             'Slower; not the original fast training setting.')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', action='store_false', dest='amp')

    # TTA (for validation)
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--no-tta', action='store_false', dest='tta')
    parser.add_argument('--tta-views', type=int, default=8)

    # EMA
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Exponential Moving Average of model weights. '
                             'EMA weights are used for val and saved as checkpoint.')
    parser.add_argument('--ema-decay', type=float, default=0.9998,
                        help='EMA decay rate (0.9998 = standard, lower = faster adaptation)')

    # Layer-wise LR decay
    parser.add_argument('--llrd-decay', type=float, default=1.0,
                        help='Layer-wise LR decay multiplier per ViT block. '
                             '1.0 = disabled (uniform backbone LR). '
                             '0.85 = standard for ViT-L/24 fine-tuning.')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'muon'],
                        help='Optimizer backend. muon uses Muon for backbone 2D+ weights '
                             'and AdamW for everything else.')
    parser.add_argument('--muon-momentum', type=float, default=0.95,
                        help='Momentum for Muon updates')
    parser.add_argument('--muon-ns-steps', type=int, default=5,
                        help='Newton-Schulz iterations for Muon orthogonalization')

    return parser.parse_args()


# =============================================================================
# SEED & DEVICE
# =============================================================================

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


# =============================================================================
# DATA LOADING
# =============================================================================

def find_data_paths(data_roots):
    """Find WBCBench data files across one or more data roots."""
    paths = {}
    for root in data_roots:
        root = Path(root)
        for name in ['phase1_label.csv', 'phase2_train.csv', 'phase2_eval.csv', 'phase2_test.csv']:
            p = root / name
            if p.exists() and name not in paths:
                paths[name] = p
        for name in ['phase1', 'phase2']:
            d = root / name
            if d.exists() and name not in paths:
                paths[name] = d
    return paths


def load_wbcbench(data_paths):
    """Load WBCBench phase1 + phase2 train data."""
    samples = []

    # Phase 1
    if 'phase1_label.csv' in data_paths and 'phase1' in data_paths:
        df = pd.read_csv(data_paths['phase1_label.csv'])
        for _, row in df.iterrows():
            samples.append({
                'img_path': str(data_paths['phase1'] / row['ID']),
                'labels': row['labels'],
                'source': 'wbcbench',
            })

    # Phase 2 train
    if 'phase2_train.csv' in data_paths and 'phase2' in data_paths:
        df = pd.read_csv(data_paths['phase2_train.csv'])
        for _, row in df.iterrows():
            samples.append({
                'img_path': str(data_paths['phase2'] / 'train' / row['ID']),
                'labels': row['labels'],
                'source': 'wbcbench',
            })

    return pd.DataFrame(samples)


def load_wbcbench_eval(data_paths):
    """Load WBCBench phase2 eval data."""
    if 'phase2_eval.csv' not in data_paths or 'phase2' not in data_paths:
        return pd.DataFrame()
    df = pd.read_csv(data_paths['phase2_eval.csv'])
    df['img_path'] = df['ID'].apply(lambda x: str(data_paths['phase2'] / 'eval' / x))
    df['source'] = 'wbcbench_eval'
    df['label_idx'] = df['labels'].map(LABEL_TO_IDX)
    return df


def load_wbcbench_test(data_paths):
    """Load WBCBench phase2 test data (no labels)."""
    if 'phase2_test.csv' not in data_paths or 'phase2' not in data_paths:
        return pd.DataFrame()
    df = pd.read_csv(data_paths['phase2_test.csv'])
    df['img_path'] = df['ID'].apply(lambda x: str(data_paths['phase2'] / 'test' / x))
    return df


def load_mll23(path):
    """Load MLL23 external dataset."""
    if path is None:
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])
    root = Path(path)
    if not root.exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    samples = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        wbc_class = MLL23_CLASS_MAP.get(folder.name.lower())
        if wbc_class is None:
            continue
        nested = folder / folder.name
        search = nested if nested.exists() else folder
        for ext in ('*.TIF', '*.tif', '*.jpg', '*.png'):
            for img in search.glob(ext):
                samples.append({'img_path': str(img), 'labels': wbc_class, 'source': 'mll23'})
    return pd.DataFrame(samples)


def load_multifocus(path):
    """Load Multi-focus Korea dataset (middle plane only)."""
    if path is None:
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])
    root = Path(path)
    labels_csv = root / 'labels.csv'
    if not labels_csv.exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    df = pd.read_csv(labels_csv)
    df['labels'] = df['label'].map(MULTI_FOCUS_CLASS_MAP)
    df = df.dropna(subset=['labels'])
    df['img_path'] = df['img_num'].apply(lambda x: str(root / f"{x}_5.jpg"))
    df['source'] = 'multifocus'
    return df[['img_path', 'labels', 'source']]


def load_chula15(path):
    """Load Chula-15 dataset."""
    if path is None:
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])
    
    root = Path(path)
    labels_csv = root / 'labels.csv'
    
    if not labels_csv.exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    df = pd.read_csv(labels_csv)
    df['labels'] = df['label'].map(CHULA15_CLASS_MAP)
    df = df.dropna(subset=['labels'])
    
    # FIX: Images are in 'images/' subdirectory, not directly in root
    df['img_path'] = df['name'].apply(lambda x: str(root / 'images' / x))
    
    df['source'] = 'chula15'
    return df[['img_path', 'labels', 'source']]


def load_kuoptofil(path, splits=('train', 'val')):
    """
    Load KU-Optofil PBC dataset.

    Handles the nested dataset/dataset/ structure on Kaggle:
        root/dataset/dataset/{train,val,test}/{Class Name}/*.jpg

    Args:
        path: root path (e.g., /kaggle/input/datasets/antonymgitau/ku-optofil-pbc)
        splits: which splits to load (default: train + val, not test)
    """
    if path is None:
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])
    root = Path(path)
    if not root.exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    samples = []
    # Search all possible nestings for the split directories
    search_prefixes = ['', 'dataset', 'dataset/dataset']
    for prefix in search_prefixes:
        for split in splits:
            split_path = root / prefix / split if prefix else root / split
            if not split_path.exists():
                continue
            for class_folder in split_path.iterdir():
                if not class_folder.is_dir():
                    continue
                wbc_class = KUOPTOFIL_CLASS_MAP.get(class_folder.name)
                if wbc_class is None:
                    continue
                for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp'):
                    for img in class_folder.glob(ext):
                        samples.append({
                            'img_path': str(img),
                            'labels': wbc_class,
                            'source': 'kuoptofil',
                        })
            if samples:
                print(f"  KU-Optofil found at: {split_path}")

    df = pd.DataFrame(samples)
    if len(df) > 0:
        print(f"KU-Optofil loaded: {len(df)} samples from splits {splits}")
        print(f"  Per-class: {df['labels'].value_counts().to_dict()}")
    else:
        print(f"WARNING: KU-Optofil found 0 images at {root}")
        print(f"  Searched: {[str(root/p) for p in search_prefixes]}")
    return df


def load_cellwiki_folder(root, label, source):
    """Load a CellWiki-style folder (all images = same class)."""
    root = Path(root)
    if not root.exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    samples = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'):
        for img in root.glob(ext):
            samples.append({'img_path': str(img), 'labels': label, 'source': source})
    return pd.DataFrame(samples)


def load_pseudo_labels(path, test_img_dir, min_confidence=None):
    """Load pseudo labels. If min_confidence is set, uses a flat threshold (0.0 = all).
    Otherwise uses per-class thresholds and head-class caps."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=['img_path', 'labels', 'source'])

    df = pd.read_csv(path)

    if min_confidence is not None:
        # Flat threshold — let focal loss handle imbalance
        df_out = df[df['confidence'] >= min_confidence].copy() if min_confidence > 0.0 else df.copy()
    else:
        # Per-class thresholds: lower for tail, higher for head
        thresholds = {
            'SNE': 0.95, 'LY': 0.95, 'MO': 0.92, 'EO': 0.92, 'BL': 0.90,
            'BA': 0.80, 'MY': 0.80, 'MMY': 0.70, 'PMY': 0.60, 'PC': 0.55,
            'VLY': 0.60, 'BNE': 0.50, 'PLY': 0.40,
        }
        caps = {'SNE': 10, 'LY': 800, 'MO': 500, 'EO': 400, 'BL': 400}

        filtered = []
        for cls, thresh in thresholds.items():
            cls_df = df[df['labels'] == cls].copy()
            cls_df = cls_df[cls_df['confidence'] >= thresh]
            if cls in caps and len(cls_df) > caps[cls]:
                cls_df = cls_df.nlargest(caps[cls], 'confidence')
            filtered.append(cls_df)
        df_out = pd.concat(filtered, ignore_index=True)

    df_out['img_path'] = df_out['ID'].apply(lambda x: str(Path(test_img_dir) / x))
    df_out['source'] = 'pseudo'

    print(f"Pseudo labels: {len(df_out)}/{len(df)} selected")
    for cls in sorted(df_out['labels'].unique()):
        n = len(df_out[df_out['labels'] == cls])
        if n > 0:
            print(f"  {cls}: {n}")

    return df_out[['img_path', 'labels', 'source']]


# =============================================================================
# AUGMENTATION
# =============================================================================

class SimpleStainNormalize:
    """Channel-wise percentile normalization for stain consistency."""
    def __init__(self, p_low=1, p_high=99):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, img: Image.Image):
        arr = np.array(img).astype(np.float32)
        for c in range(3):
            lo = np.percentile(arr[..., c], self.p_low)
            hi = np.percentile(arr[..., c], self.p_high)
            arr[..., c] = np.clip((arr[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


class RandomJPEG:
    def __init__(self, quality=(30, 100), p=0.3):
        self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        q = random.randint(*self.quality)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownUp:
    def __init__(self, min_scale=0.7, p=0.25):
        self.min_scale = min_scale
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        s = random.uniform(self.min_scale, 1.0)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR)
        return img.resize((w, h), Image.BILINEAR)


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.4), p=0.35):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return TF.adjust_gamma(img, random.uniform(*self.gamma_range))


class RandomNoise(nn.Module):
    def __init__(self, std_range=(0.0, 0.04), p=0.25):
        super().__init__()
        self.std_range = std_range
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return x + torch.randn_like(x) * random.uniform(*self.std_range)


def get_train_transforms(img_size=384):
    return transforms.Compose([
        SimpleStainNormalize(),
        transforms.RandomResizedCrop(
            img_size, scale=(0.6, 1.0), ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180, fill=0),
        transforms.ColorJitter(
            brightness=(0.65, 1.35), contrast=(0.7, 1.25),
            saturation=(0.7, 1.25), hue=(-0.10, 0.10),
        ),
        RandomGamma((0.7, 1.4), p=0.35),
        RandomJPEG((30, 100), p=0.25),
        RandomDownUp(0.8, p=0.20),
        transforms.RandomApply(
            [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.15
        ),
        transforms.ToTensor(),
        RandomNoise((0.0, 0.03), p=0.20),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


def get_val_transforms(img_size=384):
    return transforms.Compose([
        SimpleStainNormalize(),
        transforms.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC, antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


def get_tta_transforms(img_size=384, n_augments=8):
    base = [
        SimpleStainNormalize(),
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BICUBIC, antialias=True),
    ]
    to_tensor = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]

    tta = [transforms.Compose(base + to_tensor)]  # identity
    tta.append(transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + to_tensor))
    tta.append(transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)] + to_tensor))
    tta.append(transforms.Compose(
        base + [transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomVerticalFlip(p=1.0)] + to_tensor
    ))
    for angle in [90, 180, 270]:
        if len(tta) >= n_augments:
            break
        tta.append(transforms.Compose(
            base + [transforms.Lambda(lambda x, a=angle: TF.rotate(x, a))] + to_tensor
        ))
    if len(tta) < n_augments:
        zoom = int(img_size * 1.1)
        tta.append(transforms.Compose([
            SimpleStainNormalize(),
            transforms.Resize((zoom, zoom),
                              interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]))
    return tta[:n_augments]


# =============================================================================
# DATASET
# =============================================================================

class WBCDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False, img_size=384):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            if not hasattr(self, '_warn_count'):
                self._warn_count = 0
            self._warn_count += 1
            if self._warn_count <= 5:
                print(f"WARNING: Failed to load {img_path}: {e}")
            if self._warn_count == 5:
                print("WARNING: Suppressing further image load warnings...")
            image = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, row['ID']
        return image, int(row['label_idx'])


# =============================================================================
# LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) with class-balanced alpha."""
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, cosine_scale=1.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.cosine_scale = cosine_scale
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits.float() * self.cosine_scale, targets, reduction='none',
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()


class ModelEMA:
    """
    Exponential Moving Average of model weights.

    shadow = decay * shadow + (1 - decay) * current_param

    Maintains a smoothed copy of all trainable weights. Use for validation
    and checkpoint saving — the EMA model generalises better than the raw
    model, especially for small/imbalanced datasets.

    Typical decay: 0.9998 for ~5k steps per epoch; 0.999 for smaller runs.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().float().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data.float(), alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model weights → EMA weights (for validation / checkpoint save)."""
        self._backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.float() for k, v in state.items()}


class ConfusionAwareLDAMLoss(nn.Module):
    """
    LDAM Loss + Pairwise Confusion Penalty.

    Standard LDAM pushes tail-class margins. The confusion penalty adds
    targeted angular repulsion between morphologically confusable pairs.

    confusion_pairs: list of (true_class_idx, confuser_class_idx, pair_margin)
    """
    def __init__(self, class_counts, max_margin=0.5, scale=30.0,
                 weight=None, label_smoothing=0.0,
                 confusion_pairs=None, confusion_weight=0.5):
        super().__init__()

        class_counts = np.array(class_counts, dtype=np.float32)
        class_counts = np.maximum(class_counts, 1.0)  # guard against zero-count classes
        margins = max_margin / np.power(class_counts, 0.25)
        margins = margins * (max_margin / margins.max())
        self.register_buffer('margins', torch.from_numpy(margins))
        self.scale = scale
        self.label_smoothing = label_smoothing

        if weight is not None:
            self.register_buffer('weight', torch.from_numpy(np.array(weight, dtype=np.float32)))
        else:
            self.weight = None

        self.confusion_pairs = confusion_pairs or DEFAULT_CONFUSION_PAIRS
        self.confusion_weight = confusion_weight
        self.drw_active = False

    def set_drw(self, active: bool):
        self.drw_active = active
        if active:
            print("DRW activated")

    def forward(self, logits, targets):
        logits_f32 = logits.float()

        # Standard LDAM: subtract margin from target logit, then scale
        batch_margins = self.margins[targets]
        idx = torch.arange(logits_f32.size(0), device=logits_f32.device)
        logits_margin = logits_f32.clone()
        logits_margin[idx, targets] = logits_margin[idx, targets] - batch_margins
        logits_margin = logits_margin * self.scale

        w = self.weight if self.drw_active else None
        ldam_loss = F.cross_entropy(
            logits_margin, targets, weight=w, label_smoothing=self.label_smoothing
        )

        # Pairwise confusion penalty (operates on raw cosine logits, not scaled)
        if self.confusion_weight > 0 and self.confusion_pairs:
            confusion_loss = torch.tensor(0.0, device=logits.device)
            n_active = 0
            for true_cls, confuser_cls, pair_margin in self.confusion_pairs:
                mask = (targets == true_cls)
                if mask.sum() == 0:
                    continue
                true_logits = logits_f32[mask, true_cls]
                confuser_logits = logits_f32[mask, confuser_cls]
                # Hinge: penalize when confuser is within pair_margin of true
                violation = F.relu(confuser_logits - true_logits + pair_margin)
                confusion_loss = confusion_loss + violation.mean()
                n_active += 1

            if n_active > 0:
                confusion_loss = confusion_loss / n_active
                return ldam_loss + self.confusion_weight * confusion_loss

        return ldam_loss


def compute_class_weights(class_counts, beta=0.9999):
    """Effective number class weights (Cui et al., CVPR 2019)."""
    counts = np.maximum(np.array(class_counts, dtype=np.float32), 1.0)
    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective
    weights = weights / weights.sum() * len(weights)
    return weights


# =============================================================================
# SAMPLER
# =============================================================================

def create_balanced_sampler(df, strategy='sqrt'):
    counts = df['labels'].value_counts().to_dict()
    if strategy == 'balanced':
        weights = {c: 1.0 / n for c, n in counts.items()}
    elif strategy == 'sqrt':
        weights = {c: 1.0 / np.sqrt(n) for c, n in counts.items()}
    elif strategy == 'effective':
        beta = 0.9999
        weights = {c: (1 - beta) / (1 - beta ** n) for c, n in counts.items()}
    else:
        weights = {c: 1.0 for c in counts}

    total = sum(weights.values())
    weights = {c: w / total for c, w in weights.items()}
    sample_weights = df['labels'].map(weights).values
    return WeightedRandomSampler(sample_weights, len(df), replacement=True)


class TailQuotaBatchSampler:
    """
    Batch sampler that guarantees a fixed fraction of each batch comes from
    tail classes. Within the tail quota, classes are sampled equally so that
    extreme tails (PLY=11) get the same per-batch representation as moderate
    tails (BNE=2999).

    Args:
        labels: array of integer class labels for the full dataset
        batch_size: samples per batch
        tail_idx_set: set of class indices considered "tail"
        tail_frac: fraction of each batch reserved for tail classes (default 0.30)
        num_batches: batches per epoch (default: len(dataset) // batch_size)
        seed: random seed
    """
    def __init__(self, labels, batch_size, tail_idx_set, tail_frac=0.30,
                 num_batches=None, seed=42):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = batch_size
        self.tail_frac = tail_frac
        self.seed = seed
        self.epoch = 0

        mask_tail = np.isin(self.labels, list(tail_idx_set))
        self.tail_pool = np.where(mask_tail)[0]
        self.head_pool = np.where(~mask_tail)[0]

        if len(self.tail_pool) == 0 or len(self.head_pool) == 0:
            raise ValueError("TailQuotaBatchSampler needs both tail and non-tail samples")

        self.n_tail = max(1, int(round(batch_size * tail_frac)))
        self.n_head = batch_size - self.n_tail
        self.num_batches = num_batches or (len(self.labels) // batch_size)

        # Within tail: equal probability per CLASS (not per sample)
        # so PLY(11 samples) gets same batch share as BNE(2999 samples)
        tail_labels = self.labels[self.tail_pool]
        unique_tail = np.unique(tail_labels)
        p_tail = np.zeros(len(self.tail_pool), dtype=np.float64)
        for cls in unique_tail:
            cls_mask = (tail_labels == cls)
            p_tail[cls_mask] = 1.0 / (cls_mask.sum() * len(unique_tail))
        self.p_tail = p_tail / p_tail.sum()

        # Within head: uniform
        self.p_head = np.ones(len(self.head_pool), dtype=np.float64) / len(self.head_pool)

        print(f"TailQuotaBatchSampler: {self.n_tail} tail + {self.n_head} head per batch, "
              f"tail pool={len(self.tail_pool)}, head pool={len(self.head_pool)}, "
              f"tail classes={len(unique_tail)}, batches/epoch={self.num_batches}")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.num_batches):
            idx_tail = rng.choice(self.tail_pool, size=self.n_tail,
                                  replace=True, p=self.p_tail)
            idx_head = rng.choice(self.head_pool, size=self.n_head,
                                  replace=True, p=self.p_head)
            batch = np.concatenate([idx_tail, idx_head])
            rng.shuffle(batch)
            yield batch.tolist()


# =============================================================================
# MIXUP / CUTMIX
# =============================================================================

RARE_CLASSES_IDX = {LABEL_TO_IDX[c] for c in ['PC', 'PLY', 'PMY', 'BA', 'BNE', 'MMY', 'VLY']}


def mixup(images, labels, alpha=0.3, protect_rare=True):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    if protect_rare:
        rare_mask = torch.tensor([l.item() in RARE_CLASSES_IDX for l in labels], device=images.device)
        if rare_mask.any():
            lam = max(lam, 0.85)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], lam


def cutmix(images, labels, alpha=1.0, protect_rare=True):
    if alpha <= 0:
        return images, labels, labels, 1.0
    # Skip cutmix entirely if rare classes are in the batch
    if protect_rare:
        has_rare = any(l.item() in RARE_CLASSES_IDX for l in labels)
        if has_rare:
            return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    _, _, H, W = images.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(0, W), np.random.randint(0, H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    return images, labels, labels[index], lam


# =============================================================================
# TRAINING / VALIDATION
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler,
                    device, epoch, args, ema=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    optimizers = list(optimizer) if isinstance(optimizer, (list, tuple)) else [optimizer]
    schedulers = list(scheduler) if isinstance(scheduler, (list, tuple)) else [scheduler]

    for opt in optimizers:
        opt.zero_grad()

    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch + 1}/{args.epochs}")

    for batch_idx, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)

        use_mix = random.random() < args.mixup_prob
        use_cut = random.random() < args.cutmix_prob and not use_mix

        if use_mix:
            images, labels_a, labels_b, lam = mixup(images, labels, args.mixup_alpha)
        elif use_cut:
            images, labels_a, labels_b, lam = cutmix(images, labels, args.cutmix_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        with autocast(enabled=args.amp):
            out = model(images)
            logits = out['logits']
            if use_mix or use_cut:
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)
            loss = loss / args.accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            if scaler:
                for opt in optimizers:
                    scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()
            for sch in schedulers:
                sch.step()
            if ema is not None:
                ema.update(model)
            for opt in optimizers:
                opt.zero_grad()

        running_loss += loss.item() * args.accumulation_steps

        if not (use_mix or use_cut):
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})

    epoch_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro') if all_labels else 0.0
    return epoch_loss, f1


@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_logits = [], [], []

    for images, labels in tqdm(loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)
        with autocast(enabled=args.amp):
            out = model(images)
            logits = out['logits']
            loss = criterion(logits, labels)
        running_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.cpu())

    epoch_loss = running_loss / len(loader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    bacc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, f1, bacc, all_preds, all_labels, torch.cat(all_logits)


@torch.no_grad()
def predict_tta(model, df, tta_transforms, device, args):
    """TTA prediction on validation set."""
    model.eval()
    all_logits = []
    for i, tfm in enumerate(tta_transforms):
        print(f"  TTA {i + 1}/{len(tta_transforms)}...")
        ds = WBCDataset(df, transform=tfm, is_test=False, img_size=args.img_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        logits_list = []
        for images, _ in loader:
            images = images.to(device)
            with autocast(enabled=args.amp):
                out = model(images)
            logits_list.append(out['logits'].cpu())
        all_logits.append(torch.cat(logits_list))
    mean_logits = torch.stack(all_logits).mean(0)
    return mean_logits.argmax(1).numpy(), mean_logits.numpy()


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.stop = False

    def __call__(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer: momentum + matrix-update orthogonalization.
    Intended for 2D+ weight tensors only.
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, ns_steps=5, eps=1e-8):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps
        )
        super().__init__(params, defaults)

    @staticmethod
    def _as_matrix(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)

    @staticmethod
    def _orthogonalize_newton_schulz(m: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
        # Normalize first for numerical stability (ensures spectral norm <= 1).
        x = m / (m.norm() + eps)
        rows, cols = x.shape
        transposed = False
        if rows > cols:
            x = x.t()
            transposed = True

        # Newton-Schulz iteration directly on X for polar decomposition.
        # X_{k+1} = (3*X_k - X_k @ X_k^T @ X_k) / 2  converges to the orthogonal factor.
        for _ in range(steps):
            a = x @ x.t()
            x = (3.0 * x - a @ x) * 0.5

        if transposed:
            x = x.t()
        return x

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    # Muon is for matrix-like tensors only.
                    continue

                g = p.grad.detach()
                if g.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p, dtype=torch.float32)
                m = state['momentum_buffer']
                m.mul_(momentum).add_(g.float())

                m2d = self._as_matrix(m)
                update2d = self._orthogonalize_newton_schulz(m2d, ns_steps, eps)
                update = update2d.reshape_as(p).to(dtype=p.dtype)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Seed: {args.seed} (deterministic={args.deterministic})")
    print(f"Multi-scale: {args.multi_scale} (blocks: {args.block_indices})")
    print(f"Confusion loss: {args.confusion_loss} (weight: {args.confusion_weight})")

    # ---- Load data ----
    data_paths = find_data_paths(args.data_root)
    print(f"Data paths found: {list(data_paths.keys())}")

    df_train_wbc = load_wbcbench(data_paths)
    df_val = load_wbcbench_eval(data_paths)
    df_test = load_wbcbench_test(data_paths)
    print(f"WBC train: {len(df_train_wbc)}, val: {len(df_val)}, test: {len(df_test)}")

    # External data
    ext_dfs = []
    ext_dfs.append(load_mll23(args.external_mll23))
    ext_dfs.append(load_multifocus(args.external_multifocus))
    ext_dfs.append(load_chula15(args.external_chula15))
    ext_dfs.append(load_kuoptofil(args.external_kuoptofil))

    # CellWiki folders: PLY MMY PMY PMY_AML VLY BNE
    cellwiki_map = [('PLY', 'ply_ext'), ('MMY', 'mmy_ext'), ('PMY', 'pmy_ext'),
                    ('PMY', 'pmy_aml'), ('VLY', 'vly_ext'), ('BNE', 'bne_ext')]
    if args.external_cellwiki:
        for path, (label, source) in zip(args.external_cellwiki, cellwiki_map):
            ext_dfs.append(load_cellwiki_folder(path, label, source))

    non_empty = [d for d in ext_dfs if len(d) > 0]
    df_external = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame(columns=['img_path', 'labels', 'source'])

    # Filter external data to specific classes if requested
    if args.external_classes is not None and len(df_external) > 0:
        before = len(df_external)
        df_external = df_external[df_external['labels'].isin(args.external_classes)]
        print(f"External class filter {args.external_classes}: {len(df_external)}/{before} kept")

    print(f"External data: {len(df_external)}")
    if len(df_external) > 0:
        print(f"  Per-class: {df_external['labels'].value_counts().to_dict()}")

    # Pseudo labels
    test_img_dir = data_paths.get('phase2', Path('/kaggle/input/wbcbench-2026/phase2'))
    test_img_dir = test_img_dir / 'test' if not str(test_img_dir).endswith('test') else test_img_dir
    df_pseudo = load_pseudo_labels(args.pseudo_labels, str(test_img_dir),
                                   min_confidence=args.pseudo_min_confidence)

    # Combine all training data
    all_train = [df_train_wbc]
    if args.use_eval_in_train and len(df_val) > 0:
        # Add eval set to training (final submission mode)
        eval_for_train = df_val[['img_path', 'labels', 'source']].copy()
        all_train.append(eval_for_train)
        print(f"FINAL MODE: Adding {len(eval_for_train)} eval samples to training")
    if len(df_external) > 0:
        all_train.append(df_external[['img_path', 'labels', 'source']])
    if len(df_pseudo) > 0:
        all_train.append(df_pseudo[['img_path', 'labels', 'source']])

    df_train = pd.concat(all_train, ignore_index=True)
    df_train['ID'] = df_train['img_path'].apply(lambda p: Path(p).name)
    df_train['label_idx'] = df_train['labels'].map(LABEL_TO_IDX)
    df_train = df_train.dropna(subset=['label_idx'])
    df_train['label_idx'] = df_train['label_idx'].astype(int)

    print(f"\nTotal train: {len(df_train)}")
    print(f"Class distribution:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {len(df_train[df_train['labels'] == cls])}")

    # ---- Datasets & loaders ----
    train_ds = WBCDataset(df_train, get_train_transforms(args.img_size),
                          is_test=False, img_size=args.img_size)
    val_ds = WBCDataset(df_val, get_val_transforms(args.img_size),
                        is_test=False, img_size=args.img_size)

    if args.sampler == 'tail_quota':
        tail_idx_set = {LABEL_TO_IDX[c] for c in args.tail_classes if c in LABEL_TO_IDX}
        batch_sampler = TailQuotaBatchSampler(
            labels=df_train['label_idx'].values,
            batch_size=args.batch_size,
            tail_idx_set=tail_idx_set,
            tail_frac=args.tail_frac,
            num_batches=math.ceil(len(df_train) / args.batch_size),
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    elif args.sampler != 'none':
        sampler = create_balanced_sampler(df_train, args.sampler)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)

    if not args.use_eval_in_train:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        val_loader = None

    # ---- Pathology FM auto-config ----
    fm_key, _ = detect_pathology_fm(args.backbone)
    if fm_key is not None:
        hf_login()
        args.multi_scale = False  # pathology FMs don't support multi-scale
        if not any('no-freeze' in a for a in sys.argv):
            args.freeze_backbone = True
            print(f"  Auto-freezing backbone (pathology FM: {fm_key})")

    # ---- Model ----
    model = DinoBloomMultiScale(
        model_name=args.backbone,
        num_classes=NUM_CLASSES,
        img_size=args.img_size,
        pretrained=True,
        use_cosine=(args.head == 'cosine'),
        use_mlp=(args.head == 'mlp'),
        cosine_scale=args.cosine_scale,
        dropout=args.dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        multi_scale=args.multi_scale,
        block_indices=tuple(args.block_indices),
    ).to(device)

    # Load pretrained checkpoint
    _ckpt_start_f1 = 0.0  # used below to seed best_score so we never overwrite a better model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        _ckpt_start_f1 = ckpt.get('best_f1', 0.0)
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

        # Remap old Sequential head format (V4-style: classifier.0/2 → norm/classifier)
        if any(k.startswith('classifier.0.') or k.startswith('classifier.2.') for k in state):
            remapped = {}
            for k, v in state.items():
                if k == 'classifier.0.weight':
                    remapped['norm.weight'] = v
                elif k == 'classifier.0.bias':
                    remapped['norm.bias'] = v
                elif k == 'classifier.2.weight':
                    remapped['classifier.weight'] = v
                elif k == 'classifier.2.bias':
                    remapped['classifier.bias'] = v
                else:
                    remapped[k] = v
            state = remapped
            print("  Remapped old Sequential head format (V4) → current format")

        # Handle cross-head loading (cosine ↔ linear)
        # classifier.scale exists in cosine but not linear — safe to drop
        # classifier.bias exists in linear but not cosine — will init to zero
        benign_unexpected = {'classifier.scale', 'dropout.weight', 'dropout.bias'}
        benign_missing = {'classifier.bias', 'classifier.scale'}

        if args.backbone_only or args.multi_scale:
            # Load backbone only (fresh head)
            bb_state = {k.replace('backbone.', ''): v
                        for k, v in state.items() if k.startswith('backbone.')}
            if bb_state:
                missing, unexpected = model.backbone.load_state_dict(bb_state, strict=False)
                print(f"  Loaded {len(bb_state)} backbone keys "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
                print(f"  Head randomly initialized ({model.feat_dim}-dim)")
            else:
                print("  No backbone keys found, using pretrained weights")
        else:
            # Load FULL state dict (backbone + head)
            missing, unexpected = model.load_state_dict(state, strict=False)

            # Categorize results
            bb_missing = [k for k in missing if k.startswith('backbone.')]
            head_missing = [k for k in missing if not k.startswith('backbone.')]
            real_head_missing = [k for k in head_missing if k not in benign_missing]
            real_unexpected = [k for k in unexpected if k not in benign_unexpected]

            head_keys = [k for k in state if not k.startswith('backbone.')]
            head_loaded = [k for k in head_keys if k not in unexpected]

            print(f"  Backbone: {len(state) - len(head_keys)} keys loaded"
                  + (f" ({len(bb_missing)} missing)" if bb_missing else ""))
            print(f"  Head: {head_loaded}")
            if head_missing:
                for k in head_missing:
                    note = " (ok, defaults to zero)" if k in benign_missing else " ← PROBLEM"
                    print(f"    missing: {k}{note}")
            if unexpected:
                for k in unexpected:
                    note = " (ok, ignored)" if k in benign_unexpected else " ← PROBLEM"
                    print(f"    skipped: {k}{note}")
            if real_head_missing or real_unexpected:
                print(f"  WARNING: head may not have loaded correctly!")

    # Freeze backbone
    if args.freeze_backbone or args.unfreeze_last_n_blocks > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        if args.freeze_backbone and args.unfreeze_last_n_blocks == 0:
            print("Backbone frozen")

    # Partial unfreeze: selectively re-enable last N blocks + final norm
    if args.unfreeze_last_n_blocks > 0:
        if not hasattr(model.backbone, 'blocks'):
            raise ValueError("--unfreeze-last-n-blocks requires a ViT backbone with .blocks")
        n_blocks = len(model.backbone.blocks)
        first_unfrozen = max(0, n_blocks - args.unfreeze_last_n_blocks)
        bb_unfrozen = 0
        for i in range(first_unfrozen, n_blocks):
            for p in model.backbone.blocks[i].parameters():
                p.requires_grad = True
                bb_unfrozen += p.numel()
        # Also unfreeze the final LayerNorm (sits between blocks and CLS pool)
        if hasattr(model.backbone, 'norm'):
            for p in model.backbone.norm.parameters():
                p.requires_grad = True
                bb_unfrozen += p.numel()
        print(f"Partial backbone unfreeze: last {args.unfreeze_last_n_blocks}/{n_blocks} blocks "
              f"(blocks {first_unfrozen}–{n_blocks-1}) + norm  "
              f"[{bb_unfrozen/1e6:.1f}M backbone params unfrozen]")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    if args.grad_checkpoint:
        if hasattr(model.backbone, 'set_grad_checkpointing'):
            model.backbone.set_grad_checkpointing(True)
            print("Gradient checkpointing enabled on backbone")
        else:
            print("WARNING: backbone does not support set_grad_checkpointing")

    # ---- EMA ----
    ema = None
    if args.ema:
        ema = ModelEMA(model, decay=args.ema_decay)
        # Try to restore accumulated EMA state from a previous run (not fresh stages)
        if (args.checkpoint and Path(args.checkpoint).exists()
                and not args.fresh_optimizer):
            _ckpt_ema = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            _ema_sd = _ckpt_ema.get('ema_state_dict')
            if _ema_sd is not None:
                ema.load_state_dict(_ema_sd)
                print(f"EMA: restored from checkpoint (decay={args.ema_decay})")
            else:
                print(f"EMA: initialized from model weights (decay={args.ema_decay})")
            del _ckpt_ema
        else:
            print(f"EMA: initialized (decay={args.ema_decay})")

    # ---- Loss ----
    class_counts = [len(df_train[df_train['labels'] == cls]) for cls in CLASS_NAMES]
    class_weights = compute_class_weights(class_counts)

    if args.loss == 'ldam':
        criterion = ConfusionAwareLDAMLoss(
            class_counts=class_counts,
            max_margin=args.ldam_max_margin,
            scale=args.ldam_scale,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
            confusion_pairs=DEFAULT_CONFUSION_PAIRS if args.confusion_loss else [],
            confusion_weight=args.confusion_weight,
        ).to(device)
    elif args.loss == 'focal':
        # Sqrt-inverse-frequency alpha (same as V4)
        counts = np.maximum(np.array(class_counts, dtype=np.float32), 1.0)
        alpha = 1.0 / np.sqrt(counts)
        alpha = alpha / alpha.sum() * len(alpha)  # normalize to sum=num_classes
        criterion = FocalLoss(
            alpha=alpha,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            cosine_scale=args.cosine_scale if args.head == 'cosine' else 1.0,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, device=device, dtype=torch.float32),
            label_smoothing=args.label_smoothing,
        )

    print(f"Loss: {args.loss}" +
          (f" (gamma={args.focal_gamma})" if args.loss == 'focal' else "") +
          (f" + confusion penalty (w={args.confusion_weight})" if args.confusion_loss and args.loss == 'ldam' else ""))

    # ---- Optimizer & Scheduler ----
    # Exclude biases and norm params from weight decay — standard ViT fine-tuning practice.
    # Applying weight decay to LayerNorm scale/bias or any bias term is harmful.
    no_decay_suffixes = ('bias', 'norm.weight', 'norm.bias',
                         'LayerNorm.weight', 'LayerNorm.bias')

    def is_no_decay(name):
        return any(name.endswith(s) for s in no_decay_suffixes)

    optimizer = None
    scheduler = None
    optimizers_for_train = None
    schedulers_for_train = None

    if args.optimizer == 'muon' and args.freeze_backbone:
        print("Optimizer=muon requested, but backbone is frozen; falling back to AdamW.")

    if args.optimizer == 'muon' and not args.freeze_backbone:
        if args.llrd_decay < 1.0:
            print("NOTE: LLRD is disabled for Muon mode (using uniform backbone Muon LR).")

        muon_params = []
        bb1d_decay = []
        bb1d_nodecay = []
        hd_decay = []
        hd_nodecay = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('backbone.') and p.ndim >= 2:
                muon_params.append(p)
            else:
                if name.startswith('backbone.'):
                    if is_no_decay(name):
                        bb1d_nodecay.append(p)
                    else:
                        bb1d_decay.append(p)
                else:
                    if is_no_decay(name):
                        hd_nodecay.append(p)
                    else:
                        hd_decay.append(p)

        if len(muon_params) == 0:
            print("WARNING: no backbone 2D params found for Muon; falling back to AdamW.")
            args.optimizer = 'adamw'
        else:
            optimizer_muon = Muon(
                [{'params': muon_params, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay}],
                lr=args.lr_backbone,
                momentum=args.muon_momentum,
                weight_decay=args.weight_decay,
                ns_steps=args.muon_ns_steps,
            )
            optimizer_adamw = torch.optim.AdamW([
                {'params': bb1d_decay,   'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
                {'params': bb1d_nodecay, 'lr': args.lr_backbone, 'weight_decay': 0.0},
                {'params': hd_decay,     'lr': args.lr,          'weight_decay': args.weight_decay},
                {'params': hd_nodecay,   'lr': args.lr,          'weight_decay': 0.0},
            ], foreach=False)

            steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
            scheduler_muon = OneCycleLR(
                optimizer_muon, max_lr=[args.lr_backbone], epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.warmup_epochs / args.epochs,
                anneal_strategy='cos', div_factor=25, final_div_factor=100,
            )
            scheduler_adamw = OneCycleLR(
                optimizer_adamw, max_lr=[args.lr_backbone, args.lr_backbone, args.lr, args.lr], epochs=args.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=args.warmup_epochs / args.epochs,
                anneal_strategy='cos', div_factor=25, final_div_factor=100,
            )

            optimizers_for_train = [optimizer_muon, optimizer_adamw]
            schedulers_for_train = [scheduler_muon, scheduler_adamw]
            n_bb1d = len(bb1d_decay) + len(bb1d_nodecay)
            n_head = len(hd_decay) + len(hd_nodecay)
            print(f"Muon mode: backbone 2D params={len(muon_params)} (Muon lr={args.lr_backbone:.2e}), "
                  f"backbone 1D params={n_bb1d} (AdamW lr={args.lr_backbone:.2e}), "
                  f"head params={n_head} (AdamW lr={args.lr:.2e})")

    if args.optimizer == 'adamw' and not args.freeze_backbone and args.llrd_decay < 1.0:
        # LLRD: per-block learning rates for deep ViT fine-tuning.
        # block i (0-indexed) → lr_backbone * llrd_decay^(n_blocks - i)
        # stem (patch_embed, pos/cls tokens) → lr_backbone * llrd_decay^(n_blocks+1) [smallest]
        # backbone.norm → full lr_backbone [same as last block's parent]
        n_blocks = len(model.backbone.blocks)

        def get_llrd_lr(name: str) -> float:
            if not name.startswith('backbone.'):
                return args.lr  # head
            if name.startswith('backbone.blocks.'):
                blk = int(name.split('.')[2])
                return args.lr_backbone * (args.llrd_decay ** (n_blocks - blk))
            if name.startswith('backbone.norm'):
                return args.lr_backbone  # final backbone norm — full LR
            return args.lr_backbone * (args.llrd_decay ** (n_blocks + 1))  # stem

        _llrd_groups: Dict[Tuple[float, float], list] = defaultdict(list)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lr_g = round(get_llrd_lr(name), 14)
            wd_g = 0.0 if is_no_decay(name) else args.weight_decay
            _llrd_groups[(lr_g, wd_g)].append(param)
        param_groups = [
            {'params': ps, 'lr': lr_g, 'weight_decay': wd_g}
            for (lr_g, wd_g), ps in sorted(_llrd_groups.items())
        ]
        optimizer = torch.optim.AdamW(param_groups, foreach=False)
        max_lr = [g['lr'] for g in param_groups]
        _bb_lrs = [g['lr'] for g in param_groups if g['lr'] <= args.lr_backbone * 1.01]
        print(f"LLRD ({n_blocks} blocks, decay={args.llrd_decay:.2f}): "
              f"backbone LR [{min(_bb_lrs):.2e} — {args.lr_backbone:.2e}], "
              f"head={args.lr}, {len(param_groups)} groups")
    elif args.optimizer == 'adamw' and not args.freeze_backbone:
        # Four groups: backbone/head × decay/no-decay
        bb_decay    = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith('backbone.') and not is_no_decay(n)]
        bb_nodecay  = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith('backbone.') and is_no_decay(n)]
        hd_decay    = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone.') and not is_no_decay(n)]
        hd_nodecay  = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone.') and is_no_decay(n)]
        param_groups = [
            {'params': bb_decay,   'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
            {'params': bb_nodecay, 'lr': args.lr_backbone, 'weight_decay': 0.0},
            {'params': hd_decay,   'lr': args.lr,          'weight_decay': args.weight_decay},
            {'params': hd_nodecay, 'lr': args.lr,          'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, foreach=False)
        max_lr = [args.lr_backbone, args.lr_backbone, args.lr, args.lr]
        print(f"Differential LR: backbone={args.lr_backbone}, head={args.lr} "
              f"(no-decay: {len(bb_nodecay)+len(hd_nodecay)} params)")
    elif args.optimizer == 'adamw':
        decay_p   = [p for n, p in model.named_parameters()
                     if p.requires_grad and not is_no_decay(n)]
        nodecay_p = [p for n, p in model.named_parameters()
                     if p.requires_grad and is_no_decay(n)]
        optimizer = torch.optim.AdamW([
            {'params': decay_p,   'weight_decay': args.weight_decay},
            {'params': nodecay_p, 'weight_decay': 0.0},
        ], lr=args.lr, foreach=False)
        max_lr = [args.lr, args.lr]

    if optimizers_for_train is None:
        steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_steps)
        scheduler = OneCycleLR(
            optimizer, max_lr=max_lr, epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=args.warmup_epochs / args.epochs,
            anneal_strategy='cos', div_factor=25, final_div_factor=100,
        )
        optimizers_for_train = optimizer
        schedulers_for_train = scheduler
    scaler = GradScaler() if args.amp else None

    # Resume optimizer momentum buffers if available (for training continuation)
    # Directly inject state tensors to avoid load_state_dict clobbering
    # the scheduler-injected keys (max_lr, min_lr, etc.) in param_groups
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt_opt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if args.fresh_optimizer:
            print("Fresh optimizer: skipping momentum resume (--fresh-optimizer set)")
        elif args.optimizer == 'muon' and not args.backbone_only:
            try:
                if isinstance(optimizers_for_train, list) and len(optimizers_for_train) == 2:
                    muon_sd = ckpt_opt.get('optimizer_state_dict_muon')
                    adamw_sd = ckpt_opt.get('optimizer_state_dict_adamw')
                    if muon_sd is not None and adamw_sd is not None:
                        optimizers_for_train[0].load_state_dict(muon_sd)
                        optimizers_for_train[1].load_state_dict(adamw_sd)
                        print("Optimizer states resumed (Muon + AdamW)")
            except Exception as e:
                print(f"  Could not load Muon/AdamW optimizer states: {e}")
                print("  Starting with fresh optimizer")
        elif 'optimizer_state_dict' in ckpt_opt and not args.backbone_only:
            try:
                old_opt = ckpt_opt['optimizer_state_dict']
                if isinstance(old_opt, dict) and 'state' in old_opt and len(old_opt['state']) > 0:
                    # Build param-id mapping: old state uses sequential int keys
                    # Map them to current optimizer's parameter objects
                    all_params = []
                    for group in optimizer.param_groups:
                        all_params.extend(group['params'])
                    loaded, skipped = 0, 0
                    for idx_str, buf in old_opt['state'].items():
                        idx = int(idx_str) if isinstance(idx_str, str) else idx_str
                        if idx < len(all_params):
                            param = all_params[idx]
                            # Shape check: skip if momentum buffer doesn't match param
                            exp_avg = buf.get('exp_avg')
                            if exp_avg is not None and exp_avg.shape != param.shape:
                                skipped += 1
                                continue
                            optimizer.state[param] = {
                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in buf.items()
                            }
                            loaded += 1
                    msg = f"Optimizer momentum buffers resumed ({loaded}/{len(all_params)} params)"
                    if skipped:
                        msg += f", {skipped} skipped (shape mismatch — different training config)"
                    print(msg)
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")
                print("  Starting with fresh optimizer")
        del ckpt_opt

    # ---- Training loop ----
    print(f"\n{'=' * 70}")
    version = args.output_dir.rstrip('/').split('/')[-1] if args.output_dir else 'run'
    print(f"STARTING TRAINING ({version})")
    print(f"{'=' * 70}")

    early_stop = EarlyStopping(patience=args.patience)
    # Select checkpoints by validation macro-F1 (val_f1).
    # For true stage-continuation runs, seed from the start checkpoint so we never
    # save a worse model than we started with. In backbone-only mode the head is
    # intentionally reinitialized, so do not seed/copy from the source checkpoint.
    seed_from_ckpt = not args.backbone_only
    best_score = _ckpt_start_f1 if seed_from_ckpt else 0.0
    best_path = output_dir / 'best.pth'
    if not seed_from_ckpt and _ckpt_start_f1 > 0:
        print("  backbone-only mode: not seeding best_score from checkpoint (head reinitialized)")
    if best_score > 0:
        print(f"  best_score seeded from checkpoint: {best_score:.4f} (must beat this to save)")
        # If continuation training never exceeds the seeded checkpoint,
        # keep a valid best.pth by copying the input checkpoint as baseline.
        if args.checkpoint and Path(args.checkpoint).exists() and not best_path.exists():
            import shutil
            shutil.copy2(args.checkpoint, best_path)
            print(f"  baseline best checkpoint copied to: {best_path}")
    drw_epoch = int(args.epochs * args.drw_start)
    no_val = args.use_eval_in_train  # final submission mode

    if no_val:
        print("FINAL SUBMISSION MODE: no validation, saving every epoch")
        print(f"  Use epoch count from dev runs (V5 peaked at epoch 2)")

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()

        # Update batch sampler epoch for fresh randomness
        if hasattr(train_loader, 'batch_sampler') and hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)

        if args.drw and args.loss == 'ldam' and epoch == drw_epoch:
            criterion.set_drw(True)

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizers_for_train, schedulers_for_train, scaler,
            device, epoch, args, ema=ema,
        )

        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  Train — loss: {train_loss:.4f}, F1: {train_f1:.4f}")

        if not no_val:
            # --- Development mode: validate and select best ---
            if ema is not None:
                ema.apply_shadow(model)
            val_loss, val_f1, val_bacc, val_preds, val_labels, val_logits = validate(
                model, val_loader, criterion, device, args,
            )
            if ema is not None:
                ema.restore(model)
            print(f"  Val   — loss: {val_loss:.4f}, F1: {val_f1:.4f}, BAcc: {val_bacc:.4f}")

            report = classification_report(val_labels, val_preds,
                                           target_names=CLASS_NAMES, output_dict=True)
            for cls in ['BNE', 'SNE', 'MMY', 'MY', 'PMY', 'VLY', 'LY', 'PLY']:
                if cls in report:
                    print(f"    {cls}: F1={report[cls]['f1-score']:.3f} "
                          f"(P={report[cls]['precision']:.3f}, R={report[cls]['recall']:.3f})")

            # Compute composite for analysis, but select checkpoints by macro-F1.
            TAIL_CLASSES = ['BNE', 'PLY', 'VLY', 'MMY', 'PMY', 'MY', 'PC']
            tail_f1s = [report[c]['f1-score'] for c in TAIL_CLASSES if c in report]
            tail_mean = np.mean(tail_f1s) if tail_f1s else 0.0
            composite = 0.5 * val_f1 + 0.5 * tail_mean
            print(f"  Tail mean F1: {tail_mean:.4f}, Composite: {composite:.4f}")

            if val_f1 > best_score:
                best_score = val_f1
                # When EMA is active, save EMA weights as model_state_dict.
                # This means inference.py loads EMA weights directly — no extra flags.
                if ema is not None:
                    ema.apply_shadow(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'optimizer_state_dict_muon': (
                        optimizers_for_train[0].state_dict()
                        if isinstance(optimizers_for_train, list) and len(optimizers_for_train) == 2
                        else None
                    ),
                    'optimizer_state_dict_adamw': (
                        optimizers_for_train[1].state_dict()
                        if isinstance(optimizers_for_train, list) and len(optimizers_for_train) == 2
                        else None
                    ),
                    'best_f1': val_f1,
                    'best_composite': composite,
                    'tail_mean_f1': tail_mean,
                    'val_bacc': val_bacc,
                    'args': vars(args),
                    'ema_state_dict': ema.state_dict() if ema is not None else None,
                }, best_path)
                if ema is not None:
                    ema.restore(model)
                print(f"  -> New best model saved (macro-F1: {val_f1:.4f}, "
                      f"composite: {composite:.4f}, tail: {tail_mean:.4f})")

            if early_stop(val_f1):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        # Save every epoch (useful for both modes)
        epoch_path = output_dir / f'epoch{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'train_f1': train_f1,
            'args': vars(args),
        }, epoch_path)

    # In final mode, the last epoch is the "best" (no selection possible)
    if no_val:
        import shutil
        shutil.copy2(epoch_path, best_path)
        print(f"\nFinal mode: using last epoch ({epoch + 1}) as best model")

    # ---- Final evaluation (only if we have a val set) ----
    if not no_val:
        print(f"\n{'=' * 70}")
        print(f"FINAL EVALUATION (best macro-F1: {best_score:.4f})")
        print(f"{'=' * 70}")

        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Best checkpoint audit -> epoch={ckpt.get('epoch', 'N/A') + 1 if isinstance(ckpt.get('epoch', None), int) else 'N/A'}, composite={ckpt.get('best_composite', 'N/A')}, macro_f1={ckpt.get('best_f1', 'N/A')}, tail_f1={ckpt.get('tail_mean_f1', 'N/A')}, bacc={ckpt.get('val_bacc', 'N/A')}")

        _, val_f1, val_bacc, val_preds, val_labels, _ = validate(
            model, val_loader, criterion, device, args,
        )
        print(f"Val F1: {val_f1:.4f}, BAcc: {val_bacc:.4f}")
        print(classification_report(val_labels, val_preds, target_names=CLASS_NAMES))

        # Confusion matrix for the confused groups
        cm = confusion_matrix(val_labels, val_preds)
        print("\nConfusion analysis (rows=true, cols=pred):")
        groups = [
            ('BNE/SNE', [2, 11]),
            ('MMY/MY/PMY', [5, 7, 10]),
            ('LY/VLY', [4, 12]),
        ]
        for name, idxs in groups:
            print(f"\n  {name}:")
            print(f"  {'':>6}", end='')
            for j in idxs:
                print(f"  {CLASS_NAMES[j]:>5}", end='')
            print()
            for i in idxs:
                print(f"  {CLASS_NAMES[i]:>6}", end='')
                for j in idxs:
                    print(f"  {cm[i, j]:>5}", end='')
                print()

        # TTA on validation
        if args.tta:
            print(f"\nApplying TTA ({args.tta_views} views)...")
            tta_tfms = get_tta_transforms(args.img_size, args.tta_views)
            tta_preds, _ = predict_tta(model, df_val, tta_tfms, device, args)
            tta_f1 = f1_score(val_labels, tta_preds, average='macro')
            tta_bacc = balanced_accuracy_score(val_labels, tta_preds)
            print(f"TTA Val F1: {tta_f1:.4f} (delta: {tta_f1 - val_f1:+.4f})")
            print(f"TTA Val BAcc: {tta_bacc:.4f}")

    print(f"\nDone. Best model saved to: {best_path}")


if __name__ == '__main__':
    main()

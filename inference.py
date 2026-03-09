"""
msfit inference script for WBCBench 2026.

Used by the public reproduction pipeline to generate TTA predictions and logits
for the three-model conservative hybrid submission.
"""

import os
import argparse
import random
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.cuda.amp import autocast

import timm

warnings.filterwarnings('ignore')

try:
    from .modeling import DinoBloomMultiScale
except ImportError:
    from modeling import DinoBloomMultiScale

# =============================================================================
# CONSTANTS
# =============================================================================

CLASS_NAMES = ['BA', 'BL', 'BNE', 'EO', 'LY', 'MMY', 'MO', 'MY', 'PC', 'PLY', 'PMY', 'SNE', 'VLY']
NUM_CLASSES = 13
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='WBC Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data-root', type=str, nargs='+',
                        default=['/kaggle/input/wbcbench-2026'],
                        help='WBCBench data root(s)')
    parser.add_argument('--predict-split', type=str, default='test',
                        choices=['test', 'eval', 'train'],
                        help='Which split to run inference on (default: test)')
    parser.add_argument('--output-dir', type=str, default='/kaggle/working')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=None,
                        help='Image size (auto-detected from checkpoint if not set)')
    parser.add_argument('--tta', action='store_true', default=True)
    parser.add_argument('--no-tta', action='store_false', dest='tta')
    parser.add_argument('--tta-views', type=int, default=8)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.add_argument('--save-logits', action='store_true', default=False,
                        help='Save raw logits alongside submission')
    # Model overrides (auto-detected from checkpoint if not specified)
    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--multi-scale', action='store_true', default=None)
    parser.add_argument('--no-multi-scale', action='store_false', dest='multi_scale')
    parser.add_argument('--block-indices', type=int, nargs='+', default=None)
    # CORAL feature alignment
    parser.add_argument('--coral', action='store_true', default=False,
                        help='Enable CORAL feature alignment (align test→train distribution)')
    parser.add_argument('--save-features', action='store_true', default=False,
                        help='Save extracted features as .npy files')
    parser.add_argument('--load-train-features', type=str, default=None,
                        help='Load pre-saved source features (.npy) instead of extracting')
    parser.add_argument('--save-pseudo-labels', action='store_true', default=False,
                        help='Save pseudo labels CSV with ID, labels, confidence columns')
    parser.add_argument('--save-top3', action='store_true', default=False,
                        help='Save top-3 predictions CSV with pred_1/prob_1 ... pred_3/prob_3')
    # BNE/SNE specialist
    parser.add_argument('--specialist-checkpoint', type=str, default=None,
                        help='Path to BNE/SNE specialist checkpoint for hard routing')
    parser.add_argument('--specialist-margin', type=float, default=None,
                        help='Confidence gating: only route BNE/SNE when |BNE_logit - SNE_logit| < margin. '
                             'Higher = more routing, lower = only uncertain cases. '
                             'Recommended: 2.0-5.0. If not set, routes all BNE/SNE.')
    # Standalone binary expert (any timm backbone)
    parser.add_argument('--expert-checkpoint', type=str, default=None,
                        help='Path to standalone binary expert (from train_binary_expert.py)')
    parser.add_argument('--expert-margin', type=float, default=None,
                        help='Confidence gating for expert: only route when margin < threshold')
    # Per-class threshold tuning
    parser.add_argument('--tune-thresholds', action='store_true', default=False,
                        help='Optimize per-class logit biases on val set to maximize macro-F1')
    parser.add_argument('--val-csv', type=str, default=None,
                        help='Validation CSV with labels (auto-detected from data-root if not set)')
    parser.add_argument('--class-biases', type=str, default=None,
                        help='Pre-computed biases: 13 comma-separated floats, e.g. "0,0,1.5,0,..."')
    return parser.parse_args()

# =============================================================================
# BNE/SNE SPECIALIST
# =============================================================================

BNE_IDX = 2   # global 13-class index
SNE_IDX = 11  # global 13-class index


class BNESNESpecialist(nn.Module):
    """Binary BNE vs SNE classifier (matches train_specialist.py)."""
    def __init__(self, feat_dim=768, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        else:
            # v1-compatible linear probe
            self.net = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, 2),
            )

    def forward(self, x):
        return self.net(x)


def load_specialist(checkpoint_path, device):
    """Load BNE/SNE specialist from checkpoint.

    Returns:
        specialist: BNESNESpecialist model
        binary_map: dict like {'BNE': 0, 'SNE': 1}
        specialist_backbone: DinoBloomMultiScale model (if fine-tuned) or None
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feat_dim = ckpt['feat_dim']
    hidden_dim = ckpt.get('hidden_dim', 0)  # 0 = v1 linear probe
    binary_map = ckpt['binary_map']  # {'BNE': 0, 'SNE': 1}
    state_dict = ckpt['specialist_state_dict']
    fine_tuned = ckpt.get('fine_tuned_backbone', False)

    # Detect v1 checkpoints (norm.*/classifier.* keys) and remap to Sequential
    if 'norm.weight' in state_dict and hidden_dim == 0:
        remap = {
            'norm.weight': 'net.0.weight', 'norm.bias': 'net.0.bias',
            'classifier.weight': 'net.2.weight', 'classifier.bias': 'net.2.bias',
        }
        state_dict = {remap.get(k, k): v for k, v in state_dict.items()}

    specialist = BNESNESpecialist(feat_dim=feat_dim, hidden_dim=hidden_dim).to(device)
    specialist.load_state_dict(state_dict)
    specialist.eval()
    val_f1 = ckpt.get('val_f1', 'N/A')
    bne_f1 = ckpt.get('bne_f1', 'N/A')
    print(f"  Specialist loaded: feat_dim={feat_dim}, val_f1={val_f1}, bne_f1={bne_f1}")
    print(f"  Binary map: {binary_map}")

    # Load fine-tuned backbone if available
    specialist_backbone = None
    if fine_tuned and 'backbone_state_dict' in ckpt:
        saved_args = ckpt.get('args', {})
        backbone_name = saved_args.get('backbone', 'dinobloom_base')
        img_size = saved_args.get('img_size', 384)

        # Infer backbone config from the original checkpoint args
        bb_ckpt_path = saved_args.get('backbone_checkpoint', '')
        # Build backbone matching the specialist's training config
        specialist_backbone = DinoBloomMultiScale(
            model_name=backbone_name,
            num_classes=NUM_CLASSES,
            img_size=img_size,
            pretrained=False,
            use_cosine=False,  # doesn't matter, we only use backbone+norm
            dropout=0.35,
            multi_scale=False,
            block_indices=(3, 7, 11),
        ).to(device)
        specialist_backbone.load_state_dict(ckpt['backbone_state_dict'], strict=False)
        specialist_backbone.eval()
        print(f"  Fine-tuned specialist backbone loaded ({backbone_name})")

    return specialist, binary_map, specialist_backbone


@torch.no_grad()
def apply_specialist_routing(preds, features_np, specialist, binary_map, device,
                             use_amp=True, label='', logits_13=None,
                             margin_threshold=None, specialist_backbone=None,
                             img_paths=None, img_size=384):
    """
    Route BNE/SNE predictions through the specialist.

    Args:
        preds: numpy array of global 13-class predictions
        features_np: numpy array of features (N, feat_dim) from main model
        specialist: loaded BNESNESpecialist model
        binary_map: dict like {'BNE': 0, 'SNE': 1}
        device: torch device
        label: string label for logging
        logits_13: optional (N, 13) tensor — main model logits for confidence gating
        margin_threshold: if set, only route samples where |BNE_logit - SNE_logit| < threshold
            This prevents overriding high-confidence correct predictions.
        specialist_backbone: optional fine-tuned backbone for specialist feature extraction
        img_paths: list of image paths (needed when specialist_backbone is set)
        img_size: image size for specialist backbone
    Returns:
        Modified preds (numpy array)
    """
    # Find samples where main model predicted BNE or SNE
    neutrophil_mask = np.isin(preds, [BNE_IDX, SNE_IDX])

    # Confidence gating: only route uncertain BNE/SNE predictions
    if margin_threshold is not None and logits_13 is not None:
        if isinstance(logits_13, torch.Tensor):
            logits_np = logits_13.numpy()
        else:
            logits_np = logits_13
        bne_sne_margin = np.abs(logits_np[:, BNE_IDX] - logits_np[:, SNE_IDX])
        uncertain_mask = bne_sne_margin < margin_threshold
        n_total_bne_sne = neutrophil_mask.sum()
        neutrophil_mask = neutrophil_mask & uncertain_mask
        n_gated = n_total_bne_sne - neutrophil_mask.sum()
        print(f"  Confidence gating ({label}): {n_total_bne_sne} BNE/SNE total, "
              f"{n_gated} high-confidence kept, {neutrophil_mask.sum()} routed")

    n_routed = neutrophil_mask.sum()
    if n_routed == 0:
        print(f"  Specialist routing ({label}): 0 BNE/SNE samples to route")
        return preds

    # Build reverse binary map: binary_idx → global_idx
    binary_to_global = {}
    for cls_name, binary_idx in binary_map.items():
        global_idx = CLASS_NAMES.index(cls_name)
        binary_to_global[binary_idx] = global_idx

    # Get features for specialist — use fine-tuned backbone if available
    if specialist_backbone is not None and img_paths is not None:
        # Re-extract features through specialist's own backbone
        routed_indices = np.where(neutrophil_mask)[0]
        routed_paths = [img_paths[i] for i in routed_indices]
        val_tfm = get_val_transforms(img_size)
        routed_ds = ImagePathDataset(routed_paths, transform=val_tfm, img_size=img_size)
        routed_loader = DataLoader(routed_ds, batch_size=64, shuffle=False,
                                   num_workers=2, pin_memory=True)
        routed_feats_list = []
        for batch_imgs in routed_loader:
            batch_imgs = batch_imgs.to(device)
            with autocast(enabled=use_amp):
                feats = specialist_backbone.backbone(batch_imgs)
                feats = specialist_backbone.norm(feats)
            routed_feats_list.append(feats.float().cpu())
        routed_feats = torch.cat(routed_feats_list, dim=0).to(device)
        print(f"    Re-extracted features through specialist backbone ({len(routed_paths)} images)")
    else:
        # Use main model features (frozen backbone specialist)
        routed_feats = torch.tensor(features_np[neutrophil_mask], dtype=torch.float32).to(device)

    with autocast(enabled=use_amp):
        specialist_logits = specialist(routed_feats)
    specialist_preds = specialist_logits.argmax(1).cpu().numpy()

    # Map binary predictions back to global indices
    global_preds = np.array([binary_to_global[p] for p in specialist_preds])

    # Count changes
    original_routed = preds[neutrophil_mask].copy()
    preds_modified = preds.copy()
    preds_modified[neutrophil_mask] = global_preds
    n_changed = (original_routed != global_preds).sum()

    # Stats
    n_bne_before = (original_routed == BNE_IDX).sum()
    n_sne_before = (original_routed == SNE_IDX).sum()
    n_bne_after = (global_preds == BNE_IDX).sum()
    n_sne_after = (global_preds == SNE_IDX).sum()

    print(f"  Specialist routing ({label}): {n_routed} routed, {n_changed} changed")
    print(f"    Before: BNE={n_bne_before}, SNE={n_sne_before} → "
          f"After: BNE={n_bne_after}, SNE={n_sne_after}")

    return preds_modified


# =============================================================================
# STANDALONE BINARY EXPERT (from train_binary_expert.py)
# =============================================================================

class BinaryExpert(nn.Module):
    """Standalone binary expert — any timm backbone + binary head.
    Supports standard timm models AND pathology FMs (UNI2-h, UNI, Virchow2)."""

    # Pathology FM registry (must match train_binary_expert.py)
    _FM_REGISTRY = {
        'uni2-h': {
            'timm_name': 'vit_giant_patch14_224',
            'feat_dim': 1536,
            'kwargs': dict(
                img_size=224, patch_size=14, depth=24, num_heads=24,
                init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667*2,
                num_classes=0, no_embed_class=True,
                mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU,
                reg_tokens=8, dynamic_img_size=True,
            ),
            'pool': 'cls',
        },
        'uni': {
            'timm_name': 'hf-hub:MahmoodLab/uni',
            'feat_dim': 1024,
            'kwargs': dict(init_values=1e-5, dynamic_img_size=True, num_classes=0),
            'pool': 'cls',
        },
        'virchow2': {
            'timm_name': 'hf-hub:paige-ai/Virchow2',
            'feat_dim': 1280,
            'kwargs': dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=nn.SiLU, dynamic_img_size=True),
            'pool': 'virchow2',
        },
    }

    def __init__(self, backbone_name, img_size=384, pretrained=False, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        self.pool_mode = 'standard'

        # Check if pathology FM
        fm_info = None
        for key, info in self._FM_REGISTRY.items():
            if key in backbone_name.lower():
                fm_info = info
                self.pool_mode = info['pool']
                break

        if fm_info is not None:
            self.backbone = timm.create_model(
                fm_info['timm_name'], pretrained=False, **fm_info['kwargs'])
            self.feat_dim = fm_info['feat_dim']
        else:
            supports_img_size = any(k in backbone_name.lower() for k in ['vit', 'swin', 'deit'])
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0,
                img_size=img_size if supports_img_size else None,
            )
            self.feat_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, 2),
        )

    def forward(self, x):
        out = self.backbone(x)
        if self.pool_mode == 'virchow2':
            features = out[:, 0]  # CLS token
        else:
            features = out
        logits = self.head(features)
        return {'logits': logits, 'features': features}


def load_expert(checkpoint_path, device):
    """Load standalone binary expert from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone_name = ckpt['backbone_name']
    img_size = ckpt.get('img_size', 384)
    dropout = ckpt.get('dropout', 0.3)
    binary_map = ckpt['binary_map']
    class_a = ckpt.get('class_a', list(binary_map.keys())[0])
    class_b = ckpt.get('class_b', list(binary_map.keys())[1])

    # HF login for gated models
    name_lower = backbone_name.lower()
    if any(k in name_lower for k in ['uni', 'virchow', 'conch']):
        try:
            from huggingface_hub import login as hf_login
            try:
                from kaggle_secrets import UserSecretsClient
                hf_login(token=UserSecretsClient().get_secret("HF_TOKEN"))
            except Exception:
                token = os.environ.get('HF_TOKEN')
                if token:
                    hf_login(token=token)
        except ImportError:
            pass

    expert = BinaryExpert(
        backbone_name=backbone_name, img_size=img_size,
        pretrained=False, dropout=dropout,
    ).to(device)
    expert.load_state_dict(ckpt['model_state_dict'])
    expert.eval()

    val_f1 = ckpt.get('val_f1', 'N/A')
    a_f1 = ckpt.get('class_a_f1', 'N/A')
    print(f"  Expert loaded: {backbone_name}, img_size={img_size}")
    print(f"  Classes: {class_a} vs {class_b}, {class_a} F1={a_f1}, macro F1={val_f1}")
    print(f"  Binary map: {binary_map}")
    return expert, binary_map, img_size


@torch.no_grad()
def apply_expert_routing(preds, expert, binary_map, device, img_paths,
                         img_size=384, use_amp=True, label='',
                         logits_13=None, margin_threshold=None):
    """
    Route predictions through standalone binary expert.
    The expert has its own backbone — runs images through it independently.
    """
    # Determine which global class indices to route
    class_names_local = list(binary_map.keys())
    global_indices = [CLASS_NAMES.index(c) for c in class_names_local]
    route_mask = np.isin(preds, global_indices)

    # Confidence gating
    if margin_threshold is not None and logits_13 is not None:
        if isinstance(logits_13, torch.Tensor):
            logits_np = logits_13.numpy()
        else:
            logits_np = logits_13
        margin = np.abs(logits_np[:, global_indices[0]] - logits_np[:, global_indices[1]])
        uncertain = margin < margin_threshold
        n_total = route_mask.sum()
        route_mask = route_mask & uncertain
        n_gated = n_total - route_mask.sum()
        print(f"  Expert gating ({label}): {n_total} total, "
              f"{n_gated} confident kept, {route_mask.sum()} routed")

    n_routed = route_mask.sum()
    if n_routed == 0:
        print(f"  Expert routing ({label}): 0 samples to route")
        return preds

    # Build binary→global mapping
    binary_to_global = {}
    for cls_name, binary_idx in binary_map.items():
        binary_to_global[binary_idx] = CLASS_NAMES.index(cls_name)

    # Get image paths for routed samples
    routed_indices = np.where(route_mask)[0]
    routed_paths = [img_paths[i] for i in routed_indices]

    # Run through expert's own backbone
    val_tfm = get_val_transforms(img_size)
    routed_ds = ImagePathDataset(routed_paths, transform=val_tfm, img_size=img_size)
    routed_loader = DataLoader(routed_ds, batch_size=64, shuffle=False,
                               num_workers=2, pin_memory=True)

    all_expert_preds = []
    for batch_imgs in routed_loader:
        batch_imgs = batch_imgs.to(device)
        with autocast(enabled=use_amp):
            out = expert(batch_imgs)
        all_expert_preds.extend(out['logits'].argmax(1).cpu().numpy())

    expert_preds = np.array(all_expert_preds)
    global_preds = np.array([binary_to_global[p] for p in expert_preds])

    # Apply changes
    original_routed = preds[route_mask].copy()
    preds_modified = preds.copy()
    preds_modified[route_mask] = global_preds
    n_changed = (original_routed != global_preds).sum()

    # Stats
    for cls_name in class_names_local:
        gidx = CLASS_NAMES.index(cls_name)
        before = (original_routed == gidx).sum()
        after = (global_preds == gidx).sum()
        print(f"    {cls_name}: {before} → {after}")

    print(f"  Expert routing ({label}): {n_routed} routed, {n_changed} changed")
    return preds_modified


# =============================================================================
# TRANSFORMS
# =============================================================================

class SimpleStainNormalize:
    def __init__(self, p_low=1, p_high=99):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, img):
        arr = np.array(img).astype(np.float32)
        for c in range(3):
            lo = np.percentile(arr[..., c], self.p_low)
            hi = np.percentile(arr[..., c], self.p_high)
            arr[..., c] = np.clip((arr[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))


def get_val_transforms(img_size=384):
    return transforms.Compose([
        SimpleStainNormalize(),
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BICUBIC, antialias=True),
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

    tta = [transforms.Compose(base + to_tensor)]
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

class WBCTestDataset(Dataset):
    def __init__(self, df, transform=None, img_size=384):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['img_path']).convert('RGB')
        except Exception as e:
            if not hasattr(self, '_warn_count'):
                self._warn_count = 0
            self._warn_count += 1
            if self._warn_count <= 5:
                print(f"WARNING: Failed to load {row['img_path']}: {e}")
            if self._warn_count == 5:
                print("WARNING: Suppressing further image load warnings...")
            image = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, row['ID']


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def predict(model, loader, device, use_amp=True):
    """Single-pass prediction."""
    model.eval()
    all_logits, all_ids = [], []
    for images, ids in tqdm(loader, desc="Predicting"):
        images = images.to(device)
        with autocast(enabled=use_amp):
            out = model(images)
        all_logits.append(out['logits'].cpu())
        all_ids.extend(ids)
    logits = torch.cat(all_logits)
    preds = logits.argmax(1).numpy()
    return preds, logits.numpy(), all_ids


@torch.no_grad()
def predict_with_tta(model, df, tta_transforms, device, batch_size=16,
                     num_workers=2, img_size=384, use_amp=True):
    """TTA prediction: average logits across augmented views."""
    model.eval()
    all_view_logits = []

    for i, tfm in enumerate(tta_transforms):
        print(f"  TTA {i + 1}/{len(tta_transforms)}...")
        ds = WBCTestDataset(df, transform=tfm, img_size=img_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        logits_list = []
        for images, _ in loader:
            images = images.to(device)
            with autocast(enabled=use_amp):
                out = model(images)
            logits_list.append(out['logits'].cpu())
        all_view_logits.append(torch.cat(logits_list))

    mean_logits = torch.stack(all_view_logits).mean(0)
    preds = mean_logits.argmax(1).numpy()
    ids = df['ID'].tolist()
    return preds, mean_logits.numpy(), ids


# =============================================================================
# CORAL FEATURE ALIGNMENT
# =============================================================================

class ImagePathDataset(Dataset):
    """Dataset for extracting features from a list of image paths."""
    def __init__(self, img_paths, transform=None, img_size=384):
        self.paths = list(img_paths)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image


@torch.no_grad()
def extract_features(model, loader, device, use_amp=True):
    """Extract features after backbone + LayerNorm (before classifier)."""
    model.eval()
    all_feats = []
    for batch in tqdm(loader, desc="Extracting features"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)
        with autocast(enabled=use_amp):
            if model.multi_scale:
                features = model.extract_multi_scale(images)
            else:
                features = model.backbone(images)
            features = model.norm(features)
        all_feats.append(features.float().cpu())
    return torch.cat(all_feats).numpy()


def coral_align(source_feats, target_feats, verbose=True):
    """
    CORAL feature alignment: align target (test) distribution to source (train).

    Per-dimension standardization — matches mean and variance of each feature
    dimension independently. Fast, robust, no matrix decomposition needed.
    """
    mu_s = source_feats.mean(axis=0)
    mu_t = target_feats.mean(axis=0)
    std_s = source_feats.std(axis=0) + 1e-8
    std_t = target_feats.std(axis=0) + 1e-8

    aligned = (target_feats - mu_t) / std_t * std_s + mu_s

    if verbose:
        shift = np.linalg.norm(mu_s - mu_t)
        std_ratio = std_s / std_t
        print(f"  CORAL: mean shift L2={shift:.4f}, "
              f"std ratio range=[{std_ratio.min():.3f}, {std_ratio.max():.3f}]")

    return aligned.astype(np.float32)


@torch.no_grad()
def classify_features(model, features_np, device, use_amp=True, batch_size=512):
    """Classify pre-extracted features using the model's classifier head."""
    model.eval()
    all_logits = []
    for i in range(0, len(features_np), batch_size):
        batch = torch.tensor(features_np[i:i+batch_size], dtype=torch.float32).to(device)
        with autocast(enabled=use_amp):
            logits = model.classifier(batch)
        all_logits.append(logits.float().cpu())
    return torch.cat(all_logits)


def save_top3_predictions(ids, logits, biases, out_path, final_preds=None, true_labels=None):
    """
    Save top-3 class predictions and probabilities for each sample.

    Uses softmax(logits + biases) so rankings match any class-bias tuning.
    """
    logits_cpu = logits.detach().cpu()
    bias_tensor = torch.tensor(biases, dtype=logits_cpu.dtype).view(1, -1)
    probs = torch.softmax(logits_cpu + bias_tensor, dim=1)
    top_probs, top_idx = torch.topk(probs, k=3, dim=1)

    top3_df = pd.DataFrame({'ID': ids})
    if true_labels is None:
        top3_df['label'] = ''
    else:
        top3_df['label'] = list(true_labels)
    if final_preds is not None:
        top3_df['labels'] = [IDX_TO_LABEL[int(p)] for p in final_preds]
    for rank in range(3):
        top3_df[f'pred_{rank + 1}'] = [IDX_TO_LABEL[int(i)] for i in top_idx[:, rank].tolist()]
        top3_df[f'prob_{rank + 1}'] = top_probs[:, rank].numpy()
    if true_labels is None:
        top3_df['correct'] = ''
    else:
        top3_df['correct'] = (top3_df['pred_1'] == top3_df['label'])

    top3_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(top3_df)} rows)")


def load_source_image_paths(data_roots):
    """Load phase1 + phase2 train + eval image paths for CORAL source distribution."""
    paths = []
    for root in data_roots:
        root = Path(root)
        # Phase 1
        p1_csv = root / 'phase1_label.csv'
        p1_dir = root / 'phase1'
        if p1_csv.exists() and p1_dir.exists():
            df = pd.read_csv(p1_csv)
            for _, row in df.iterrows():
                paths.append(str(p1_dir / row['ID']))
            print(f"  CORAL source: {len(df)} phase1 images from {p1_dir}")
        # Phase 2 train + eval
        for split, csv_name, subdir in [
            ('train', 'phase2_train.csv', 'phase2/train'),
            ('eval', 'phase2_eval.csv', 'phase2/eval'),
        ]:
            csv_path = root / csv_name
            img_dir = root / subdir
            if csv_path.exists() and img_dir.exists():
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    paths.append(str(img_dir / row['ID']))
                print(f"  CORAL source: {len(df)} {split} images from {img_dir}")
    return paths


# =============================================================================
# PER-CLASS THRESHOLD TUNING
# =============================================================================

LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def tune_class_biases(logits_np, true_labels, class_names, verbose=True):
    """
    Find per-class logit biases that maximize macro-F1 on the validation set.

    Instead of pred = argmax(logits), we use pred = argmax(logits + bias).
    A positive bias for a tail class makes the model predict it more often
    (higher recall), at the cost of precision in competing classes.

    Uses derivative-free global optimization (differential evolution) since
    macro-F1 is non-differentiable.

    Args:
        logits_np: (N, C) raw logits from the model
        true_labels: (N,) integer class labels
        class_names: list of class name strings
        verbose: print per-class breakdown

    Returns:
        biases: (C,) optimal bias vector
        best_f1: macro-F1 achieved with optimal biases
    """
    from scipy.optimize import differential_evolution
    from sklearn.metrics import f1_score

    n_classes = len(class_names)

    # --- Baseline (no biases) ---
    baseline_preds = logits_np.argmax(axis=1)
    baseline_f1 = f1_score(true_labels, baseline_preds, average='macro', zero_division=0)
    baseline_per_class = f1_score(true_labels, baseline_preds, average=None,
                                  labels=range(n_classes), zero_division=0)

    if verbose:
        print(f"\n  Baseline macro-F1: {baseline_f1:.4f}")
        print(f"  {'Class':<5} {'F1':>6} {'True':>6} {'Pred':>6}")
        print(f"  {'─'*26}")
        for i, name in enumerate(class_names):
            n_true = (true_labels == i).sum()
            n_pred = (baseline_preds == i).sum()
            print(f"  {name:<5} {baseline_per_class[i]:>6.3f} {n_true:>6d} {n_pred:>6d}")

    # --- Optimize biases ---
    # Fix class 0 (BA) bias at 0 — biases are relative, so one must be anchored.
    # Tune the other 12.
    def neg_macro_f1(biases_12):
        biases = np.zeros(n_classes)
        biases[1:] = biases_12
        preds = (logits_np + biases).argmax(axis=1)
        return -f1_score(true_labels, preds, average='macro', zero_division=0)

    bounds = [(-5, 5)] * (n_classes - 1)

    if verbose:
        print(f"\n  Optimizing {n_classes - 1} biases (differential evolution)...")

    result = differential_evolution(
        neg_macro_f1, bounds,
        maxiter=1000, seed=42, tol=1e-7,
        popsize=30, mutation=(0.5, 1.5), recombination=0.9,
    )

    biases = np.zeros(n_classes)
    biases[1:] = result.x
    best_f1 = -result.fun

    if verbose:
        tuned_preds = (logits_np + biases).argmax(axis=1)
        tuned_per_class = f1_score(true_labels, tuned_preds, average=None,
                                   labels=range(n_classes), zero_division=0)

        print(f"\n  Optimized macro-F1: {best_f1:.4f}  (delta = {best_f1 - baseline_f1:+.4f})")
        print(f"\n  {'Class':<5} {'Bias':>7} {'Before':>8} {'After':>8} {'Delta':>8}")
        print(f"  {'─'*38}")
        for i, name in enumerate(class_names):
            delta = tuned_per_class[i] - baseline_per_class[i]
            flag = " <--" if abs(delta) > 0.01 else ""
            print(f"  {name:<5} {biases[i]:>+7.3f} {baseline_per_class[i]:>8.3f} "
                  f"{tuned_per_class[i]:>8.3f} {delta:>+8.3f}{flag}")

        # Print copy-pasteable string
        bias_str = ','.join(f'{b:.4f}' for b in biases)
        print(f"\n  Copy-paste for --class-biases:")
        print(f"    --class-biases \"{bias_str}\"")

    return biases, best_f1


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and extract saved args
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})

    # Determine model configuration (from checkpoint or overrides)
    backbone = args.backbone or saved_args.get('backbone', 'dinobloom_base')
    multi_scale = args.multi_scale if args.multi_scale is not None else saved_args.get('multi_scale', True)
    block_indices = args.block_indices or saved_args.get('block_indices', [3, 7, 11])
    img_size = args.img_size or saved_args.get('img_size', 384)
    head = saved_args.get('head', 'cosine')
    cosine_scale = saved_args.get('cosine_scale', 30.0)
    dropout = saved_args.get('dropout', 0.35)
    mlp_hidden_dim = saved_args.get('mlp_hidden_dim', 512)
    mlp_dropout = saved_args.get('mlp_dropout', 0.20)

    print(f"Model config: backbone={backbone}, multi_scale={multi_scale}, "
          f"blocks={block_indices}, head={head}")

    # Build model
    model = DinoBloomMultiScale(
        model_name=backbone,
        num_classes=NUM_CLASSES,
        img_size=img_size,
        pretrained=False,  # will load from checkpoint
        use_cosine=(head == 'cosine'),
        use_mlp=(head == 'mlp'),
        cosine_scale=cosine_scale,
        dropout=dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_dropout=mlp_dropout,
        multi_scale=multi_scale,
        block_indices=tuple(block_indices),
    ).to(device)

    # Load weights — strict for head params, backbone loaded from pretrained init
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Filter out backbone keys from missing (expected when pretrained=False)
    head_missing = [k for k in missing if not k.startswith('backbone.')]
    if head_missing:
        print(f"  ERROR: Missing head keys: {head_missing}")
        print("  This likely means checkpoint architecture doesn't match. Aborting.")
        return
    if missing:
        print(f"  Missing backbone keys: {len(missing)} (will use random init — expected)")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.eval()

    val_f1 = ckpt.get('best_f1', ckpt.get('val_f1', 'N/A'))
    print(f"  Checkpoint val F1: {val_f1}")

    # Load BNE/SNE specialist if provided
    specialist = None
    binary_map = None
    specialist_backbone = None
    if args.specialist_checkpoint and Path(args.specialist_checkpoint).exists():
        print(f"\nLoading BNE/SNE specialist: {args.specialist_checkpoint}")
        specialist, binary_map, specialist_backbone = load_specialist(
            args.specialist_checkpoint, device)
        if args.specialist_margin is not None:
            print(f"  Confidence gating: margin threshold = {args.specialist_margin}")

    # Load standalone binary expert if provided
    expert = None
    expert_binary_map = None
    expert_img_size = img_size
    if args.expert_checkpoint and Path(args.expert_checkpoint).exists():
        print(f"\nLoading binary expert: {args.expert_checkpoint}")
        expert, expert_binary_map, expert_img_size = load_expert(
            args.expert_checkpoint, device)
        if args.expert_margin is not None:
            print(f"  Confidence gating: margin threshold = {args.expert_margin}")

    # Load inference data (test/eval/train)
    split_to_csv = {
        'test': 'phase2_test.csv',
        'eval': 'phase2_eval.csv',
        'train': 'phase2_train.csv',
    }
    split_to_subdir = {
        'test': 'test',
        'eval': 'eval',
        'train': 'train',
    }
    infer_csv_name = split_to_csv[args.predict_split]
    infer_subdir = split_to_subdir[args.predict_split]
    infer_csv = None
    infer_img_dir = None
    for root in args.data_root:
        root = Path(root)
        csv = root / infer_csv_name
        img_dir = root / 'phase2' / infer_subdir
        if csv.exists():
            infer_csv = csv
        if img_dir.exists():
            infer_img_dir = img_dir

    if infer_csv is None:
        print(f"ERROR: Could not find {infer_csv_name}")
        return
    if infer_img_dir is None:
        print(f"ERROR: Could not find image directory phase2/{infer_subdir}")
        return

    df_infer = pd.read_csv(infer_csv)
    df_infer['img_path'] = df_infer['ID'].apply(lambda x: str(infer_img_dir / x))
    true_labels = df_infer['labels'].tolist() if 'labels' in df_infer.columns else None
    print(f"Inference split: {args.predict_split}")
    print(f"Samples: {len(df_infer)} from {infer_csv}")
    print(f"Ground-truth labels available: {'yes' if true_labels is not None else 'no'}")

    # Derive tag from checkpoint filename for output naming
    ckpt_stem = Path(args.checkpoint).stem  # e.g., "best", "epoch2", "v4l_best"
    tag = ckpt_stem
    img_paths = df_infer['img_path'].tolist()
    ids = df_infer['ID'].tolist()
    val_tfm = get_val_transforms(img_size)

    # --- CORAL: load/extract source features ---
    source_feats = None
    if args.coral:
        print(f"\n{'='*60}")
        print("CORAL FEATURE ALIGNMENT")
        print(f"{'='*60}")
        if args.load_train_features:
            print(f"Loading source features: {args.load_train_features}")
            source_feats = np.load(args.load_train_features)
        else:
            source_paths = load_source_image_paths(args.data_root)
            if not source_paths:
                print("ERROR: No source images found for CORAL.")
                print("  Need phase2_train.csv + phase2/train/ in --data-root")
                print("  Falling back to standard inference.")
                args.coral = False
            else:
                print(f"  Total source images: {len(source_paths)}")
                source_ds = ImagePathDataset(source_paths, transform=val_tfm, img_size=img_size)
                source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers, pin_memory=True)
                source_feats = extract_features(model, source_loader, device, args.amp)
                if args.save_features:
                    feat_path = output_dir / 'source_features.npy'
                    np.save(feat_path, source_feats)
                    print(f"  Saved: {feat_path}")
                print(f"  Source features: {source_feats.shape}")

    # --- Extract test features ---
    print(f"\nExtracting test features...")
    test_ds = ImagePathDataset(img_paths, transform=val_tfm, img_size=img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_feats = extract_features(model, test_loader, device, args.amp)

    if args.save_features:
        np.save(output_dir / 'test_features.npy', test_feats)
        print(f"  Saved: {output_dir / 'test_features.npy'}")

    # --- Per-class threshold tuning ---
    biases = np.zeros(NUM_CLASSES)
    if args.class_biases:
        biases = np.array([float(x) for x in args.class_biases.split(',')])
        assert len(biases) == NUM_CLASSES, f"Expected {NUM_CLASSES} biases, got {len(biases)}"
        print(f"\nUsing pre-computed class biases:")
        for i, name in enumerate(CLASS_NAMES):
            if abs(biases[i]) > 0.001:
                print(f"  {name}: {biases[i]:+.4f}")

    elif args.tune_thresholds:
        print(f"\n{'='*60}")
        print("PER-CLASS THRESHOLD TUNING")
        print(f"{'='*60}")

        # Find val CSV
        val_csv_path = args.val_csv
        val_img_dir = None
        if val_csv_path is None:
            for root in args.data_root:
                root = Path(root)
                for candidate in ['phase2_eval.csv', 'phase2_val.csv']:
                    p = root / candidate
                    if p.exists():
                        val_csv_path = str(p)
                        break
                vdir = root / 'phase2' / 'eval'
                if vdir.exists():
                    val_img_dir = vdir
        else:
            val_csv_path = str(val_csv_path)
            # Try to find val image dir near the CSV
            for root in args.data_root:
                vdir = Path(root) / 'phase2' / 'eval'
                if vdir.exists():
                    val_img_dir = vdir

        if val_csv_path is None or not Path(val_csv_path).exists():
            print("ERROR: Cannot find validation CSV. Use --val-csv to specify.")
            print("  Skipping threshold tuning.")
        elif val_img_dir is None or not val_img_dir.exists():
            print("ERROR: Cannot find validation image dir (phase2/eval/).")
            print("  Skipping threshold tuning.")
        else:
            df_val = pd.read_csv(val_csv_path)
            df_val['img_path'] = df_val['ID'].apply(lambda x: str(val_img_dir / x))
            val_labels_int = np.array([LABEL_TO_IDX[l] for l in df_val['labels']])
            print(f"  Val samples: {len(df_val)} from {val_csv_path}")

            # Extract val features and classify
            val_ds = ImagePathDataset(df_val['img_path'].tolist(), transform=val_tfm, img_size=img_size)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            val_feats = extract_features(model, val_loader, device, args.amp)
            val_logits = classify_features(model, val_feats, device, args.amp).numpy()

            biases, tuned_f1 = tune_class_biases(val_logits, val_labels_int, CLASS_NAMES)
        print(f"{'='*60}\n")

    has_biases = np.abs(biases).max() > 0.001
    if has_biases:
        tag = tag + '_tuned'

    # --- No-TTA: standard prediction ---
    print(f"\nClassifying (no TTA)...")
    logits_no_tta = classify_features(model, test_feats, device, args.amp)
    preds_no_tta = (logits_no_tta.numpy() + biases).argmax(axis=1)

    # Apply specialist routing to no-TTA predictions
    if specialist is not None:
        preds_no_tta = apply_specialist_routing(
            preds_no_tta, test_feats, specialist, binary_map, device,
            args.amp, label='noTTA', logits_13=logits_no_tta,
            margin_threshold=args.specialist_margin,
            specialist_backbone=specialist_backbone,
            img_paths=img_paths, img_size=img_size)

    # Apply standalone expert routing to no-TTA predictions
    if expert is not None:
        preds_no_tta = apply_expert_routing(
            preds_no_tta, expert, expert_binary_map, device, img_paths,
            img_size=expert_img_size, use_amp=args.amp, label='noTTA',
            logits_13=logits_no_tta, margin_threshold=args.expert_margin)

    sub = pd.DataFrame({'ID': ids, 'labels': [IDX_TO_LABEL[p] for p in preds_no_tta]})
    sub_path = output_dir / f'submission_{tag}_noTTA.csv'
    sub.to_csv(sub_path, index=False)
    print(f"Saved: {sub_path} ({len(sub)} rows)")
    print(f"  Distribution: {sub['labels'].value_counts().to_dict()}")
    if args.save_top3:
        top3_path = output_dir / f'top3_{tag}_noTTA.csv'
        save_top3_predictions(
            ids, logits_no_tta, biases, top3_path,
            final_preds=preds_no_tta, true_labels=true_labels
        )

    if args.save_logits:
        np.save(output_dir / f'logits_{tag}_noTTA.npy', logits_no_tta.numpy())

    # --- Pseudo labels export ---
    if args.save_pseudo_labels:
        probs = torch.softmax(logits_no_tta, dim=1)
        confs, pred_idxs = probs.max(dim=1)
        pseudo_df = pd.DataFrame({
            'ID': ids,
            'labels': [IDX_TO_LABEL[p.item()] for p in pred_idxs],
            'confidence': confs.numpy(),
        })
        pseudo_path = output_dir / 'pseudo_labels.csv'
        pseudo_df.to_csv(pseudo_path, index=False)
        print(f"\nPseudo labels saved: {pseudo_path} ({len(pseudo_df)} rows)")
        print(f"  Confidence stats: mean={confs.mean():.3f}, "
              f"median={confs.median():.3f}, min={confs.min():.3f}")
        for thresh in [0.9, 0.8, 0.7, 0.5]:
            n = (confs >= thresh).sum().item()
            print(f"  >= {thresh}: {n} ({100*n/len(confs):.1f}%)")

    # --- No-TTA: CORAL prediction ---
    if args.coral:
        print(f"\nApplying CORAL alignment (no TTA)...")
        test_feats_aligned = coral_align(source_feats, test_feats)
        logits_coral = classify_features(model, test_feats_aligned, device, args.amp)
        preds_coral = (logits_coral.numpy() + biases).argmax(axis=1)

        # Apply specialist routing to CORAL no-TTA predictions
        if specialist is not None:
            preds_coral = apply_specialist_routing(
                preds_coral, test_feats, specialist, binary_map, device,
                args.amp, label='CORAL noTTA', logits_13=logits_coral,
                margin_threshold=args.specialist_margin,
                specialist_backbone=specialist_backbone,
                img_paths=img_paths, img_size=img_size)

        sub_coral = pd.DataFrame({'ID': ids, 'labels': [IDX_TO_LABEL[p] for p in preds_coral]})
        sub_coral_path = output_dir / f'submission_{tag}_CORAL_noTTA.csv'
        sub_coral.to_csv(sub_coral_path, index=False)
        print(f"Saved: {sub_coral_path} ({len(sub_coral)} rows)")
        print(f"  Distribution: {sub_coral['labels'].value_counts().to_dict()}")
        if args.save_top3:
            top3_coral_path = output_dir / f'top3_{tag}_CORAL_noTTA.csv'
            save_top3_predictions(
                ids, logits_coral, biases, top3_coral_path,
                final_preds=preds_coral, true_labels=true_labels
            )

        diff = (preds_no_tta != preds_coral).sum()
        print(f"  CORAL changed {diff}/{len(preds_no_tta)} predictions "
              f"({100*diff/len(preds_no_tta):.1f}%)")

        if args.save_logits:
            np.save(output_dir / f'logits_{tag}_CORAL_noTTA.npy', logits_coral.numpy())

    # --- TTA prediction ---
    if args.tta:
        print(f"\nPredicting with TTA ({args.tta_views} views)...")
        tta_tfms = get_tta_transforms(img_size, args.tta_views)
        all_logits_std = []
        all_logits_coral = [] if args.coral else None

        for i, tfm in enumerate(tta_tfms):
            print(f"  TTA {i+1}/{len(tta_tfms)}...")
            ds = ImagePathDataset(img_paths, transform=tfm, img_size=img_size)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            feats = extract_features(model, loader, device, args.amp)

            logits_view = classify_features(model, feats, device, args.amp)
            all_logits_std.append(logits_view)

            if args.coral:
                feats_aligned = coral_align(source_feats, feats, verbose=False)
                logits_coral_view = classify_features(model, feats_aligned, device, args.amp)
                all_logits_coral.append(logits_coral_view)

        # Standard TTA
        mean_logits_std = torch.stack(all_logits_std).mean(0)
        preds_tta = (mean_logits_std.numpy() + biases).argmax(axis=1)

        # Apply specialist routing to TTA predictions (uses identity-view features)
        if specialist is not None:
            preds_tta = apply_specialist_routing(
                preds_tta, test_feats, specialist, binary_map, device,
                args.amp, label='TTA', logits_13=mean_logits_std,
                margin_threshold=args.specialist_margin,
                specialist_backbone=specialist_backbone,
                img_paths=img_paths, img_size=img_size)

        # Apply standalone expert routing to TTA predictions
        if expert is not None:
            preds_tta = apply_expert_routing(
                preds_tta, expert, expert_binary_map, device, img_paths,
                img_size=expert_img_size, use_amp=args.amp, label='TTA',
                logits_13=mean_logits_std, margin_threshold=args.expert_margin)

        sub_tta = pd.DataFrame({'ID': ids, 'labels': [IDX_TO_LABEL[p] for p in preds_tta]})
        sub_tta_path = output_dir / f'submission_{tag}_TTA.csv'
        sub_tta.to_csv(sub_tta_path, index=False)
        print(f"Saved: {sub_tta_path} ({len(sub_tta)} rows)")
        print(f"  Distribution: {sub_tta['labels'].value_counts().to_dict()}")
        if args.save_top3:
            top3_tta_path = output_dir / f'top3_{tag}_TTA.csv'
            save_top3_predictions(
                ids, mean_logits_std, biases, top3_tta_path,
                final_preds=preds_tta, true_labels=true_labels
            )

        if args.save_logits:
            np.save(output_dir / f'logits_{tag}_TTA.npy', mean_logits_std.numpy())

        # CORAL TTA
        if args.coral:
            mean_logits_coral = torch.stack(all_logits_coral).mean(0)
            preds_coral_tta = (mean_logits_coral.numpy() + biases).argmax(axis=1)

            # Apply specialist routing to CORAL TTA predictions
            if specialist is not None:
                preds_coral_tta = apply_specialist_routing(
                    preds_coral_tta, test_feats, specialist, binary_map, device,
                    args.amp, label='CORAL TTA', logits_13=mean_logits_coral,
                    margin_threshold=args.specialist_margin,
                    specialist_backbone=specialist_backbone,
                    img_paths=img_paths, img_size=img_size)

            sub_coral_tta = pd.DataFrame({
                'ID': ids, 'labels': [IDX_TO_LABEL[p] for p in preds_coral_tta],
            })
            sub_coral_tta_path = output_dir / f'submission_{tag}_CORAL_TTA.csv'
            sub_coral_tta.to_csv(sub_coral_tta_path, index=False)
            print(f"Saved: {sub_coral_tta_path} ({len(sub_coral_tta)} rows)")
            print(f"  Distribution: {sub_coral_tta['labels'].value_counts().to_dict()}")
            if args.save_top3:
                top3_coral_tta_path = output_dir / f'top3_{tag}_CORAL_TTA.csv'
                save_top3_predictions(
                    ids, mean_logits_coral, biases, top3_coral_tta_path,
                    final_preds=preds_coral_tta, true_labels=true_labels
                )

            diff_coral = (preds_tta != preds_coral_tta).sum()
            print(f"  CORAL changed {diff_coral}/{len(preds_tta)} TTA predictions "
                  f"({100*diff_coral/len(preds_tta):.1f}%)")

        # Compare TTA vs no-TTA
        diff = (preds_no_tta != preds_tta).sum()
        print(f"\nTTA changed {diff}/{len(preds_no_tta)} predictions "
              f"({100*diff/len(preds_no_tta):.1f}%)")

    print("\nDone.")


if __name__ == '__main__':
    main()

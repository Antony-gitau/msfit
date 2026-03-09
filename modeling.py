"""
Shared model and backbone utilities for msfit.

This module holds the model definitions that are used by both training and
inference so they cannot drift apart over time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import SwiGLUPacked


DINOBLOOM_MODELS = {
    'dinobloom_small': 'hf-hub:1aurent/vit_small_patch14_224.dinobloom',
    'dinobloom_base': 'hf-hub:1aurent/vit_base_patch14_224.dinobloom',
    'dinobloom_large': 'hf-hub:1aurent/vit_large_patch14_224.dinobloom',
    'dinobloom_giant': 'hf-hub:1aurent/vit_giant_patch14_224.dinobloom',
}

PATHOLOGY_FM_REGISTRY = {
    'uni2-h': {
        'timm_name': 'vit_giant_patch14_224',
        'hf_repo': 'MahmoodLab/UNI2-h',
        'feat_dim': 1536,
        'native_size': 224,
        'kwargs': dict(
            img_size=224, patch_size=14, depth=24, num_heads=24,
            init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
            num_classes=0, no_embed_class=True,
            mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
            reg_tokens=8, dynamic_img_size=True,
        ),
        'pool': 'cls',
    },
    'uni': {
        'timm_name': 'hf-hub:MahmoodLab/uni',
        'hf_repo': 'MahmoodLab/UNI',
        'feat_dim': 1024,
        'native_size': 224,
        'kwargs': dict(init_values=1e-5, dynamic_img_size=True, num_classes=0),
        'pool': 'cls',
    },
    'virchow2': {
        'timm_name': 'hf-hub:paige-ai/Virchow2',
        'hf_repo': 'paige-ai/Virchow2',
        'feat_dim': 1280,
        'native_size': 224,
        'kwargs': dict(mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, dynamic_img_size=True),
        'pool': 'virchow2',
    },
}


def detect_pathology_fm(backbone_name: str):
    """Return (registry_key, registry_info) for known pathology backbones."""
    name_lower = backbone_name.lower()
    for key, info in PATHOLOGY_FM_REGISTRY.items():
        if key in name_lower:
            return key, info
    return None, None


class CosineClassifier(nn.Module):
    """Cosine similarity classifier. Scale is kept only for checkpoint compat."""

    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x_norm, w_norm)


class MLPClassifier(nn.Module):
    """Small MLP head used on top of backbone features."""

    def __init__(self, in_features, num_classes, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class DinoBloomMultiScale(nn.Module):
    """
    Backbone wrapper with optional multi-scale CLS fusion.

    For pathology FMs we disable multi-scale extraction and use the backbone's
    native output behavior. For Virchow2 this means selecting the CLS token from
    the returned token sequence.
    """

    def __init__(
        self,
        model_name='dinobloom_base',
        num_classes=13,
        img_size=384,
        pretrained=True,
        use_cosine=True,
        use_mlp=False,
        cosine_scale=30.0,
        dropout=0.35,
        mlp_hidden_dim=512,
        mlp_dropout=0.2,
        multi_scale=True,
        block_indices=(3, 7, 11),
    ):
        super().__init__()

        self.pool_mode = 'standard'
        fm_key, fm_info = detect_pathology_fm(model_name)

        if fm_info is not None:
            self.pool_mode = fm_info['pool']
            multi_scale = False
            if fm_key == 'uni2-h':
                from huggingface_hub import hf_hub_download

                self.backbone = timm.create_model(
                    fm_info['timm_name'], pretrained=False, **fm_info['kwargs']
                )
                weights_path = hf_hub_download(fm_info['hf_repo'], filename='pytorch_model.bin')
                self.backbone.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print(f"  Loaded UNI2-h weights from {weights_path}")
            else:
                self.backbone = timm.create_model(
                    fm_info['timm_name'], pretrained=pretrained, **fm_info['kwargs']
                )
            self.embed_dim = fm_info['feat_dim']
        else:
            hf_id = DINOBLOOM_MODELS.get(model_name, model_name)
            self.backbone = timm.create_model(
                hf_id, pretrained=pretrained, img_size=img_size, num_classes=0,
            )
            self.embed_dim = self.backbone.num_features

        self.multi_scale = multi_scale
        self.block_indices = sorted(block_indices)

        if multi_scale:
            n_scales = len(self.block_indices)
            self.feat_dim = self.embed_dim * n_scales
            self.block_norms = nn.ModuleList([
                nn.LayerNorm(self.embed_dim) for _ in range(n_scales)
            ])
        else:
            self.feat_dim = self.embed_dim

        self.norm = nn.LayerNorm(self.feat_dim)
        self.dropout = nn.Dropout(dropout)

        if use_mlp:
            self.classifier = MLPClassifier(
                self.feat_dim, num_classes,
                hidden_dim=mlp_hidden_dim, dropout=mlp_dropout,
            )
        elif use_cosine:
            self.classifier = CosineClassifier(self.feat_dim, num_classes, cosine_scale)
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)

    def extract_multi_scale(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if hasattr(self.backbone, 'norm_pre'):
            x = self.backbone.norm_pre(x)
        if hasattr(self.backbone, 'patch_drop'):
            x = self.backbone.patch_drop(x)

        cls_tokens = []
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.block_indices:
                cls_tokens.append(x[:, 0])

        normed = [norm(cls) for norm, cls in zip(self.block_norms, cls_tokens)]
        return torch.cat(normed, dim=-1)

    def forward(self, x):
        if self.multi_scale:
            features = self.extract_multi_scale(x)
        else:
            out = self.backbone(x)
            if self.pool_mode == 'virchow2':
                features = out[:, 0]
            else:
                features = out

        features = self.norm(features)
        features = self.dropout(features)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}

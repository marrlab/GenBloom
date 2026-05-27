"""
Inference-only model loading for GenBloom-G.

Provides load_contrastive_model(), which is the single entry point called by
dinov2/eval/multi_dataset_eval.py when --contrastive-checkpoint is used.
"""

import logging
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, projection_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Contrastive model wrapper — plug-compatible with WSILinearProbeEvaluator
# ---------------------------------------------------------------------------

class ContrastiveModelWrapper(nn.Module):
    """vision_model + projection_head with forward_features() interface."""

    def __init__(self, vision_model, projection_head, vision_tower: str = "transformer"):
        super().__init__()
        self.vision_model = vision_model
        self.projection_head = projection_head
        self.vision_tower = vision_tower
        self.return_unprojected = False

    @property
    def unprojected_dim(self) -> int:
        return self.projection_head.net[0].in_features

    def forward_features(self, x: torch.Tensor) -> dict:
        features = x.squeeze(0)  # (N, feature_dim)
        bag_emb = _embed_bag(self.vision_model, features, x.device, self.vision_tower)
        bag_emb = bag_emb.unsqueeze(0)  # (1, D)
        if self.return_unprojected:
            return {"x_norm_clstoken": bag_emb}
        return {"x_norm_clstoken": self.projection_head(bag_emb)}

    def forward(self, x: torch.Tensor) -> dict:
        return self.forward_features(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.projection_head.eval()
        if self.vision_model is not None:
            self.vision_model.eval()
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_cls_token(model, features: torch.Tensor, device: str) -> torch.Tensor:
    x = features.unsqueeze(0).to(device)
    return model.forward_features(x)["x_norm_clstoken"].squeeze(0)


def _embed_bag(vision_model, features: torch.Tensor, device: str, vision_tower: str) -> torch.Tensor:
    if vision_tower == "transformer":
        return _extract_cls_token(vision_model, features, device)
    else:  # mean_pool fallback
        return features.to(device).mean(dim=0)


def _build_vision_model(checkpoint_path: str, embed_dim: int, feature_dim: int,
                        depth: int, num_heads: int, patch_size: int, device: str):
    from dinov2.models import vision_transformer_no_pos
    from dinov2.layers import FeatureEmbed

    model = vision_transformer_no_pos.DinoVisionTransformerNoPos(
        img_size=[224, 224],
        patch_size=patch_size,
        in_chans=feature_dim,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        init_values=1e-5,
        embed_layer=FeatureEmbed,
    )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "teacher" not in ckpt:
        raise KeyError(f"Missing 'teacher' key. Available: {list(ckpt.keys())}")

    sd = {}
    for k, v in ckpt["teacher"].items():
        if k.startswith("backbone."):
            k = k.replace("backbone.", "", 1)
        elif k.startswith("dino_head.") or k.startswith("ibot_head."):
            continue
        k = re.sub(r"^blocks\.(\d+)\.", r"blocks.0.\1.", k)
        sd[k] = v
    sd.pop("pos_embed", None)

    msg = model.load_state_dict(sd, strict=False)
    critical = [k for k in msg.missing_keys if not k.startswith("pos_embed")]
    if any(k.startswith("blocks.") for k in critical):
        raise RuntimeError("Failed to load transformer blocks!")

    model.to(device).eval()
    logger.info(f"Vision model loaded (embed_dim={embed_dim}, feature_dim={feature_dim})")
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_contrastive_model(
    checkpoint_path: str,
    dinov2_checkpoint_path: str = None,
    embed_dim: int = 768,
    feature_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    patch_size: int = 1,
    device: str = "cuda",
):
    """Load a GenBloom-G checkpoint and return (ContrastiveModelWrapper, projection_dim).

    Args:
        checkpoint_path:       Path to best_model.pth.
        dinov2_checkpoint_path: Path to GenBloom-V teacher checkpoint
                                (required when vision_tower == "transformer").
        embed_dim / feature_dim / depth / num_heads / patch_size:
                                Architecture parameters (must match checkpoint).
        device:                 'cuda' or 'cpu'.

    Returns:
        (ContrastiveModelWrapper, int)  — model and projection dimension.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    vision_tower = ckpt["vision_tower"]

    proj_input_dim  = ckpt["projection_head"]["net.0.weight"].shape[1]
    projection_dim  = ckpt["projection_head"]["net.4.weight"].shape[0]

    if vision_tower == "transformer":
        if dinov2_checkpoint_path is None:
            raise ValueError("--dinov2-checkpoint is required for transformer tower")
        vision_model = _build_vision_model(
            dinov2_checkpoint_path, embed_dim, feature_dim, depth, num_heads, patch_size, device
        )
        if "vision_model" in ckpt:
            vision_model.load_state_dict(ckpt["vision_model"])
            logger.info("Loaded finetuned vision model weights")
    else:
        vision_model = None

    projection_head = ProjectionHead(proj_input_dim, projection_dim).to(device)
    projection_head.load_state_dict(ckpt["projection_head"])

    wrapper = ContrastiveModelWrapper(vision_model, projection_head, vision_tower).to(device)
    wrapper.eval()

    logger.info(f"Loaded checkpoint: tower={vision_tower}, proj_dim={projection_dim}")
    return wrapper, projection_dim

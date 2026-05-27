# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Vision Transformer without positional embeddings for WSI bag-level modeling.

This variant is designed for whole slide imaging (WSI) applications where:
- Patches are pre-encoded by a foundation model
- Spatial relationships between patches are not meaningful or should be ignored
- Each sample is a bag with variable number of patches
"""

import torch
import torch.nn as nn

from .vision_transformer import DinoVisionTransformer


class DinoVisionTransformerNoPos(DinoVisionTransformer):
    """
    DINOv2 Vision Transformer without positional embeddings.

    Inherits from DinoVisionTransformer but completely disables positional encoding,
    making it suitable for bag-level WSI modeling where patches don't have
    meaningful spatial relationships.

    This class ALWAYS operates without positional embeddings - there is no option
    to enable them.
    """

    def __init__(self, **kwargs):
        """
        Initialize the model without positional embeddings.

        Args:
            **kwargs: Arguments passed to DinoVisionTransformer
        """
        super().__init__(**kwargs)

        # Zero out positional embeddings and freeze them
        with torch.no_grad():
            self.pos_embed.fill_(0.0)
        self.pos_embed.requires_grad = False

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        Prepare tokens without positional encoding for bag-level modeling.

        Args:
            x: Input tensor of shape (B, N, D) for pre-encoded features
               where B=batch size (typically 1), N=number of patches, D=feature dim
            masks: Optional mask tensor

        Returns:
            Tensor with CLS token prepended and optionally register tokens
        """
        # For 3D input (B, N, D), apply patch_embed to project to embed_dim
        if x.ndim == 3:
            # Input is (B, N, patch_encoder_dim), need to project to embed_dim
            embedded = self.patch_embed(x)
        else:
            raise ValueError(f"Expected input with 3 dimensions, got {x.ndim}")

        # Apply mask if provided
        if masks is not None:
            embedded = torch.where(
                masks.unsqueeze(-1),
                self.mask_token.to(embedded.dtype).unsqueeze(0),
                embedded
            )

        # Prepend CLS token
        x = torch.cat((self.cls_token.expand(embedded.shape[0], -1, -1), embedded), dim=1)

        # NO positional embeddings added
        # (pos_embed is zeroed out in __init__, so even if added it has no effect)

        # Add register tokens if configured
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features(self, x, masks=None):
        """
        Forward pass optimized for bag-level features.

        Args:
            x: Input tensor of shape (1, N, D) where N can vary per bag
            masks: Optional mask tensor

        Returns:
            Dictionary with outputs including CLS token and patch tokens
        """
        # Handle both single bags and lists of bags
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        # Prepare tokens (no positional encoding if use_pos_embed=False)
        x = self.prepare_tokens_with_masks(x, masks)

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Normalize
        x_norm = self.norm(x)

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

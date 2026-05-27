# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Feature embedding layer for pre-encoded patches (e.g., from WSI patch encoders)
#
# Design Choice: Simple linear projection (no non-linearities)
# ============================================================
# This layer uses a single linear projection without activation functions to
# preserve pre-trained feature representations from foundation models (e.g., UNI).
# The goal is to learn aggregation in the transformer blocks, not to transform
# the features themselves. This approach:
#   - Maintains semantic structure of pre-trained features
#   - Reduces parameters and computation
#   - Separates concerns: feature extraction (UNI) vs. aggregation (DINOv2)

from typing import Callable, Optional

from torch import Tensor
import torch.nn as nn


class FeatureEmbed(nn.Module):
    """
    Feature embedding layer for pre-encoded patches: (B,N,D_in) -> (B,N,D_out)

    This layer is designed for scenarios where patches have already been encoded
    by a foundation model (e.g., in computational pathology with WSI analysis).
    Instead of using a convolutional layer to extract patches from images,
    this uses a simple linear projection to adapt pre-encoded features to the
    transformer embedding dimension.

    IMPORTANT: Uses a single linear layer (no non-linearities) to preserve the
    pre-trained feature representations. The goal is to learn aggregation in the
    transformer blocks, not to transform the features.

    Args:
        in_features: Input feature dimension from the patch encoder (e.g., 1024 for UNI).
        embed_dim: Output embedding dimension for the transformer (e.g., 1024 for ViT-Large).
        bias: Whether to use bias in the linear projection.
        norm_layer: Optional normalization layer applied after projection (typically None).

        # Unused parameters kept for API compatibility:
        hidden_features: Not used (kept for compatibility).
        act_layer: Not used (kept for compatibility).
        drop: Not used (kept for compatibility).
    """

    def __init__(
        self,
        in_features: int = None,
        embed_dim: int = 768,
        hidden_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Optional[Callable] = None,
        drop: float = 0.0,
        bias: bool = True,
        # These parameters are kept for API compatibility with PatchEmbed
        img_size: int = None,  # Not used for feature embedding
        patch_size: int = None,  # Not used for feature embedding
        in_chans: int = None,  # Can be used as alias for in_features
    ) -> None:
        super().__init__()

        # Allow in_chans to be used as an alias for in_features for API compatibility
        if in_features is None and in_chans is not None:
            in_features = in_chans
        elif in_features is None and in_chans is None:
            raise ValueError("Either in_features or in_chans must be specified")

        self.in_features = in_features
        self.embed_dim = embed_dim

        # Simple linear projection: in_features -> embed_dim
        # No non-linearities to preserve pre-trained feature representations
        self.project = nn.Linear(in_features, embed_dim, bias=bias)

        # Optional normalization after projection (typically not used)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # For compatibility with vision_transformer.py
        # Calculate num_patches from img_size and patch_size if provided
        if img_size is not None and patch_size is not None:
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            if isinstance(patch_size, int):
                patch_size = (patch_size, patch_size)
            grid_h = img_size[0] // patch_size[0]
            grid_w = img_size[1] // patch_size[1]
            self._num_patches = grid_h * grid_w
        else:
            # Will be dynamically determined from input
            self._num_patches = None

    @property
    def num_patches(self) -> int:
        """
        Returns the number of patches. This is dynamically determined from the input.
        For compatibility with the vision transformer, we return a cached value.
        """
        if self._num_patches is None:
            # Return a default value if not yet set (will be updated on first forward pass)
            return 0
        return self._num_patches

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for feature embedding.

        Args:
            x: Input tensor of shape (B, N, D_in) OR (1, N, D_in) where:
               B = batch size (typically 1 for WSI bags)
               N = number of patches (varies per bag/slide)
               D_in = input feature dimension

               Also supports 4D input (B, C, H, W) for compatibility

        Returns:
            Output tensor of shape (B, N, D_out) where D_out = embed_dim

        Note:
            For WSI applications with variable-length bags, it's recommended to process
            one bag at a time (batch_size=1) to avoid padding/masking complications.
        """
        # Handle both 3D (B, N, D) and 4D (B, C, H, W) inputs
        # 4D input is for compatibility with vision_transformer.prepare_tokens_with_masks
        if x.ndim == 4:
           raise ValueError("4D input not supported in feature_embed.py")
        elif x.ndim == 3:
            # Input is already (B, N, D)
            D = x.shape[2]
            N = x.shape[1]
            assert D == self.in_features, (
                f"Input feature dimension {D} does not match expected {self.in_features}"
            )
        else:
            raise ValueError(f"Expected input with 3 or 4 dimensions, got {x.ndim}")

        # Update num_patches based on current input
        # NOTE: This is updated per forward pass and may vary between samples
        # For batches with variable lengths, process samples individually
        self._num_patches = N

        # Linear projection: preserves pre-trained feature structure
        # while adapting dimensionality for the transformer
        x = self.project(x)

        return x

    def flops(self) -> float:
        """
        Calculate FLOPs for the feature embedding layer.

        For a linear projection: N * in_features * embed_dim

        Returns:
            Approximate FLOPs count
        """
        if self._num_patches is None:
            return 0

        N = self._num_patches

        # FLOPs for linear projection: N * in_features * embed_dim
        flops = N * self.in_features * self.embed_dim

        # FLOPs for normalization (if not Identity)
        if not isinstance(self.norm, nn.Identity):
            flops += N * self.embed_dim

        return flops

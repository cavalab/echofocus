"""PanEcho backbone wrapper."""

import torch
from torch import nn


class PanEchoBackbone(nn.Module):
    """Thin wrapper around the PanEcho backbone."""

    def __init__(self, backbone_only=True, trainable=True):
        """Load the PanEcho backbone.

        Args:
            backbone_only (bool): If True, load only the backbone.
            trainable (bool): If False, freeze backbone parameters.
        """
        super().__init__()
        self.model = torch.hub.load(
            "CarDS-Yale/PanEcho",
            "PanEcho",
            force_reload=False,
            backbone_only=backbone_only,
        )

        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, clips):
        """Embed clips using the PanEcho backbone.

        Args:
            clips (torch.Tensor): Clip batch shaped (B, 3, 16, 224, 224).

        Returns:
            torch.Tensor: Clip embeddings shaped (B, 768).
        """
        return self.model(clips)

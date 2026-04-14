#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from torch import nn

from u1_score_model_torch import SigmaEmbedding
from u1_lcnn_torch import LocalU1LCNN


def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def cossin_to_angles_torch(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 4:
        raise ValueError("x must have shape (batch, 4, L, L)")
    return torch.stack(
        [
            torch.atan2(x[:, 1], x[:, 0]),
            torch.atan2(x[:, 3], x[:, 2]),
        ],
        dim=1,
    )


def angles_to_cossin_torch(angles: torch.Tensor) -> torch.Tensor:
    if angles.ndim != 4 or angles.shape[1] != 2:
        raise ValueError("angles must have shape (batch, 2, L, L)")
    return torch.stack(
        [
            torch.cos(angles[:, 0]),
            torch.sin(angles[:, 0]),
            torch.cos(angles[:, 1]),
            torch.sin(angles[:, 1]),
        ],
        dim=1,
    )


def sample_wrapped_noisy_angles(
    clean_angles: torch.Tensor,
    sigma: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(clean_angles.shape, dtype=clean_angles.dtype, generator=generator).to(clean_angles.device)
    noisy_angles = wrap_angle_torch(clean_angles + sigma[:, None, None, None] * noise)
    wrapped_delta = wrap_angle_torch(noisy_angles - clean_angles)
    target_score = -wrapped_delta / (sigma[:, None, None, None] ** 2)
    return noisy_angles, target_score


class TimeConditionedU1AngleScoreNet(nn.Module):
    def __init__(
        self,
        lcnn_hidden_channels: int = 16,
        lcnn_out_channels: int = 8,
        sigma_emb_dim: int = 32,
        sigma_hidden_dim: int = 64,
        head_hidden_channels: int = 64,
    ):
        super().__init__()
        self.backbone = LocalU1LCNN(hidden_channels=lcnn_hidden_channels, out_channels=lcnn_out_channels)
        self.sigma_embedding = SigmaEmbedding(emb_dim=sigma_emb_dim, hidden_dim=sigma_hidden_dim)
        self.context_scale = nn.Linear(sigma_hidden_dim, lcnn_out_channels)
        self.context_shift = nn.Linear(sigma_hidden_dim, lcnn_out_channels)
        self.head_time = nn.Linear(sigma_hidden_dim, head_hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(4 + lcnn_out_channels + head_hidden_channels, head_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_channels, head_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_channels, 2, kernel_size=1),
        )

    def forward(self, noisy_links_cossin: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 1 or sigma.shape[0] != noisy_links_cossin.shape[0]:
            raise ValueError("sigma must have shape (batch,)")

        hidden_r, hidden_i = self.backbone.forward_hidden(noisy_links_cossin)
        invariant_context = hidden_r * hidden_r + hidden_i * hidden_i

        sigma_emb = self.sigma_embedding(sigma)
        context_scale = self.context_scale(sigma_emb)[:, :, None, None]
        context_shift = self.context_shift(sigma_emb)[:, :, None, None]
        invariant_context = invariant_context * (1.0 + context_scale) + context_shift

        time_map = self.head_time(sigma_emb)[:, :, None, None].expand(
            -1,
            -1,
            noisy_links_cossin.shape[-2],
            noisy_links_cossin.shape[-1],
        )
        head_input = torch.cat([noisy_links_cossin, invariant_context, time_map], dim=1)
        return self.head(head_input)


def tangent_score_matching_loss(pred_score: torch.Tensor, target_score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    weight = sigma[:, None, None, None] ** 2
    return torch.mean(weight * (pred_score - target_score) ** 2)


def denoise_angles_with_score(noisy_angles: torch.Tensor, pred_score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return wrap_angle_torch(noisy_angles + (sigma[:, None, None, None] ** 2) * pred_score)

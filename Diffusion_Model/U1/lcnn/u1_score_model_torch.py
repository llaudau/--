#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from torch import nn

from u1_lcnn_torch import LocalU1LCNN, cossin_to_angles_torch, plaquette_cosine_from_features


def normalize_link_features(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 4:
        raise ValueError("x must have shape (batch, 4, L, L)")
    x_norm = torch.sqrt(torch.clamp(x[:, 0] ** 2 + x[:, 1] ** 2, min=eps))
    y_norm = torch.sqrt(torch.clamp(x[:, 2] ** 2 + x[:, 3] ** 2, min=eps))
    return torch.stack(
        [
            x[:, 0] / x_norm,
            x[:, 1] / x_norm,
            x[:, 2] / y_norm,
            x[:, 3] / y_norm,
        ],
        dim=1,
    )


def sinusoidal_sigma_embedding(sigma: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        raise ValueError("embedding dimension must be at least 2")
    log_sigma = torch.log(torch.clamp(sigma, min=1e-8))[:, None]
    freq = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=sigma.device,
            dtype=sigma.dtype,
        )
    )[None, :]
    phase = log_sigma * freq
    emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class SigmaEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_sigma_embedding(sigma, self.emb_dim))


class TimeConditionedU1ScoreNet(nn.Module):
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
            nn.Conv2d(4 + 4 + lcnn_out_channels + 1 + head_hidden_channels, head_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_channels, head_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_hidden_channels, 4, kernel_size=1),
        )

    def forward(self, noisy_x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if noisy_x.ndim != 4 or noisy_x.shape[1] != 4:
            raise ValueError("noisy_x must have shape (batch, 4, L, L)")
        if sigma.ndim != 1 or sigma.shape[0] != noisy_x.shape[0]:
            raise ValueError("sigma must have shape (batch,)")

        projected_x = normalize_link_features(noisy_x)
        hidden_r, hidden_i = self.backbone.forward_hidden(projected_x)
        invariant_context = hidden_r * hidden_r + hidden_i * hidden_i
        plaquette = plaquette_cosine_from_features(projected_x)[:, None]

        sigma_emb = self.sigma_embedding(sigma)
        context_scale = self.context_scale(sigma_emb)[:, :, None, None]
        context_shift = self.context_shift(sigma_emb)[:, :, None, None]
        invariant_context = invariant_context * (1.0 + context_scale) + context_shift

        time_map = self.head_time(sigma_emb)[:, :, None, None].expand(
            -1,
            -1,
            noisy_x.shape[-2],
            noisy_x.shape[-1],
        )

        head_input = torch.cat([noisy_x, projected_x, invariant_context, plaquette, time_map], dim=1)
        return self.head(head_input)


def denoised_estimate_from_score(noisy_x: torch.Tensor, score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return noisy_x + (sigma[:, None, None, None] ** 2) * score


def score_matching_loss(pred_score: torch.Tensor, target_score: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    weight = sigma[:, None, None, None] ** 2
    return torch.mean(weight * (pred_score - target_score) ** 2)


def sample_log_uniform_sigma(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    u = torch.rand(batch_size, generator=generator).to(device)
    log_sigma = math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min))
    return torch.exp(log_sigma)

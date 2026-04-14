#!/usr/bin/env python3
from __future__ import annotations

import math

import torch
from torch import nn


def cossin_channels_to_complex_links(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or x.shape[1] != 4:
        raise ValueError("x must have shape (batch, 4, L, L)")
    return x[:, 0], x[:, 1], x[:, 2], x[:, 3]


def complex_mul(ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor, bi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ar * br - ai * bi, ar * bi + ai * br


def complex_conj(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return real, -imag


def complex_abs(real: torch.Tensor, imag: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(real * real + imag * imag + eps)


def wrap_angle_torch(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def cossin_to_angles_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.atan2(x[:, 1], x[:, 0]),
            torch.atan2(x[:, 3], x[:, 2]),
        ],
        dim=1,
    )


def plaquette_cosine_from_features(x: torch.Tensor) -> torch.Tensor:
    angles = cossin_to_angles_torch(x)
    x_links = angles[:, 0]
    y_links = angles[:, 1]
    plaquette = wrap_angle_torch(
        x_links
        + torch.roll(y_links, shifts=-1, dims=1)
        - torch.roll(x_links, shifts=-1, dims=2)
        - y_links
    )
    return torch.cos(plaquette)


def random_gauge_angles_torch(batch: int, L: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
    alpha = (2.0 * math.pi) * torch.rand((batch, L, L), generator=gen) - math.pi
    return alpha.to(device)


def gauge_transform_features(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    angles = cossin_to_angles_torch(x)
    transformed = torch.empty_like(angles)
    transformed[:, 0] = wrap_angle_torch(angles[:, 0] + alpha - torch.roll(alpha, shifts=-1, dims=1))
    transformed[:, 1] = wrap_angle_torch(angles[:, 1] + alpha - torch.roll(alpha, shifts=-1, dims=2))
    return torch.stack(
        [
            torch.cos(transformed[:, 0]),
            torch.sin(transformed[:, 0]),
            torch.cos(transformed[:, 1]),
            torch.sin(transformed[:, 1]),
        ],
        dim=1,
    )


def canonical_wilson_line_seed(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ux_r, ux_i, uy_r, uy_i = cossin_channels_to_complex_links(x)
    batch, L, _ = ux_r.shape
    device = x.device
    dtype = x.dtype

    path_r = torch.ones((batch, L, L), device=device, dtype=dtype)
    path_i = torch.zeros((batch, L, L), device=device, dtype=dtype)

    x_axis_r = torch.ones((batch, L), device=device, dtype=dtype)
    x_axis_i = torch.zeros((batch, L), device=device, dtype=dtype)
    for xi in range(1, L):
        prev_r = x_axis_r[:, xi - 1]
        prev_i = x_axis_i[:, xi - 1]
        link_r = ux_r[:, xi - 1, 0]
        link_i = ux_i[:, xi - 1, 0]
        x_axis_r[:, xi], x_axis_i[:, xi] = complex_mul(prev_r, prev_i, link_r, link_i)

    path_r[:, :, 0] = x_axis_r
    path_i[:, :, 0] = x_axis_i

    for xi in range(L):
        for yi in range(1, L):
            prev_r = path_r[:, xi, yi - 1]
            prev_i = path_i[:, xi, yi - 1]
            link_r = uy_r[:, xi, yi - 1]
            link_i = uy_i[:, xi, yi - 1]
            path_r[:, xi, yi], path_i[:, xi, yi] = complex_mul(prev_r, prev_i, link_r, link_i)

    return complex_conj(path_r, path_i)


def initial_covariant_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    seed_r, seed_i = canonical_wilson_line_seed(x)
    plaquette = plaquette_cosine_from_features(x)
    plaquette_avg = (
        plaquette
        + torch.roll(plaquette, shifts=1, dims=1)
        + torch.roll(plaquette, shifts=-1, dims=1)
        + torch.roll(plaquette, shifts=1, dims=2)
        + torch.roll(plaquette, shifts=-1, dims=2)
    ) / 5.0

    scalars = torch.stack(
        [
            torch.ones_like(plaquette),
            plaquette,
            plaquette_avg,
        ],
        dim=1,
    )
    return seed_r[:, None] * scalars, seed_i[:, None] * scalars


def transport_forward(feature_r: torch.Tensor, feature_i: torch.Tensor, links: torch.Tensor, mu: int) -> tuple[torch.Tensor, torch.Tensor]:
    ux_r, ux_i, uy_r, uy_i = cossin_channels_to_complex_links(links)
    link_r, link_i = (ux_r, ux_i) if mu == 0 else (uy_r, uy_i)
    shifted_r = torch.roll(feature_r, shifts=-1, dims=mu + 2)
    shifted_i = torch.roll(feature_i, shifts=-1, dims=mu + 2)
    return complex_mul(link_r[:, None], link_i[:, None], shifted_r, shifted_i)


def transport_backward(feature_r: torch.Tensor, feature_i: torch.Tensor, links: torch.Tensor, mu: int) -> tuple[torch.Tensor, torch.Tensor]:
    ux_r, ux_i, uy_r, uy_i = cossin_channels_to_complex_links(links)
    link_r, link_i = (ux_r, ux_i) if mu == 0 else (uy_r, uy_i)
    conj_r, conj_i = complex_conj(link_r, link_i)
    conj_r = torch.roll(conj_r, shifts=1, dims=mu + 1)
    conj_i = torch.roll(conj_i, shifts=1, dims=mu + 1)
    shifted_r = torch.roll(feature_r, shifts=1, dims=mu + 2)
    shifted_i = torch.roll(feature_i, shifts=1, dims=mu + 2)
    return complex_mul(conj_r[:, None], conj_i[:, None], shifted_r, shifted_i)


class LConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        scale = 0.1
        self.center = nn.Parameter(scale * torch.randn(out_channels, in_channels))
        self.forward_weight = nn.Parameter(scale * torch.randn(2, out_channels, in_channels))
        self.backward_weight = nn.Parameter(scale * torch.randn(2, out_channels, in_channels))

    def forward(self, feature_r: torch.Tensor, feature_i: torch.Tensor, links: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_r = torch.einsum("oi,bixy->boxy", self.center, feature_r)
        out_i = torch.einsum("oi,bixy->boxy", self.center, feature_i)

        for mu in range(2):
            fwd_r, fwd_i = transport_forward(feature_r, feature_i, links, mu)
            bwd_r, bwd_i = transport_backward(feature_r, feature_i, links, mu)
            out_r = out_r + torch.einsum("oi,bixy->boxy", self.forward_weight[mu], fwd_r)
            out_i = out_i + torch.einsum("oi,bixy->boxy", self.forward_weight[mu], fwd_i)
            out_r = out_r + torch.einsum("oi,bixy->boxy", self.backward_weight[mu], bwd_r)
            out_i = out_i + torch.einsum("oi,bixy->boxy", self.backward_weight[mu], bwd_i)

        return out_r, out_i


class EquivariantModReLU(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-12):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, feature_r: torch.Tensor, feature_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = complex_abs(feature_r, feature_i, self.eps)
        activated = torch.relu(magnitude + self.bias[None, :, None, None])
        scale = activated / torch.clamp(magnitude, min=self.eps)
        return feature_r * scale, feature_i * scale


class InvariantReadout(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.weight = nn.Parameter(0.1 * torch.randn(num_channels))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feature_r: torch.Tensor, feature_i: torch.Tensor) -> torch.Tensor:
        invariant = feature_r * feature_r + feature_i * feature_i
        return torch.einsum("c,bcxy->bxy", self.weight, invariant)[:, None] + self.bias.view(1, 1, 1, 1)


class LocalU1LCNN(nn.Module):
    def __init__(self, hidden_channels: int = 16, out_channels: int = 8):
        super().__init__()
        self.lconv1 = LConv2d(in_channels=3, out_channels=hidden_channels)
        self.act1 = EquivariantModReLU(hidden_channels)
        self.lconv2 = LConv2d(in_channels=hidden_channels, out_channels=out_channels)
        self.act2 = EquivariantModReLU(out_channels)
        self.readout = InvariantReadout(out_channels)

    def forward_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_r, feature_i = initial_covariant_features(x)
        feature_r, feature_i = self.lconv1(feature_r, feature_i, x)
        feature_r, feature_i = self.act1(feature_r, feature_i)
        feature_r, feature_i = self.lconv2(feature_r, feature_i, x)
        feature_r, feature_i = self.act2(feature_r, feature_i)
        return feature_r, feature_i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_r, feature_i = self.forward_hidden(x)
        return self.readout(feature_r, feature_i)


def expected_seed_phase(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    delta = alpha - alpha[:, :1, :1]
    return torch.cos(delta)[:, None], torch.sin(delta)[:, None]


def max_hidden_covariance_residual(
    hidden_before: tuple[torch.Tensor, torch.Tensor],
    hidden_after: tuple[torch.Tensor, torch.Tensor],
    alpha: torch.Tensor,
) -> float:
    before_r, before_i = hidden_before
    after_r, after_i = hidden_after
    phase_r, phase_i = expected_seed_phase(alpha)
    exp_r, exp_i = complex_mul(phase_r, phase_i, before_r, before_i)
    residual = torch.max(torch.sqrt((after_r - exp_r) ** 2 + (after_i - exp_i) ** 2))
    return float(residual.item())


def max_invariant_residual(output_before: torch.Tensor, output_after: torch.Tensor) -> float:
    return float(torch.max(torch.abs(output_after - output_before)).item())

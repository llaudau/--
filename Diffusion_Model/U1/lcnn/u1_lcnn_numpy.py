#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


TWOPI = 2.0 * math.pi


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + math.pi) % TWOPI - math.pi


def angles_to_complex_links(link_angles: np.ndarray) -> np.ndarray:
    if link_angles.ndim != 3 or link_angles.shape[0] != 2:
        raise ValueError("link_angles must have shape (2, L, L)")
    return np.exp(1j * link_angles)


def random_gauge_angles(L: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-math.pi, math.pi, size=(L, L))


def random_covariant_site_feature(
    num_channels: int,
    L: int,
    rng: np.random.Generator,
    amplitude_scale: float = 1.0,
) -> np.ndarray:
    amplitude = amplitude_scale * rng.uniform(0.2, 1.2, size=(num_channels, L, L))
    phase = rng.uniform(-math.pi, math.pi, size=(num_channels, L, L))
    return amplitude * np.exp(1j * phase)


def gauge_transform_links_complex(links: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if links.ndim != 3 or links.shape[0] != 2:
        raise ValueError("links must have shape (2, L, L)")
    if alpha.shape != links.shape[1:]:
        raise ValueError("alpha must have shape (L, L)")

    g = np.exp(1j * alpha)
    transformed = np.empty_like(links)
    transformed[0] = g * links[0] * np.conjugate(np.roll(g, shift=-1, axis=0))
    transformed[1] = g * links[1] * np.conjugate(np.roll(g, shift=-1, axis=1))
    return transformed


def gauge_transform_site_feature(feature: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if feature.ndim != 3:
        raise ValueError("feature must have shape (channels, L, L)")
    if alpha.shape != feature.shape[1:]:
        raise ValueError("alpha must have shape (L, L)")
    return feature * np.exp(1j * alpha)[None, :, :]


def transport_forward(feature: np.ndarray, links: np.ndarray, mu: int) -> np.ndarray:
    if mu not in (0, 1):
        raise ValueError("mu must be 0 or 1")
    return links[mu][None, :, :] * np.roll(feature, shift=-1, axis=mu + 1)


def transport_backward(feature: np.ndarray, links: np.ndarray, mu: int) -> np.ndarray:
    if mu not in (0, 1):
        raise ValueError("mu must be 0 or 1")
    backward_links = np.conjugate(np.roll(links[mu], shift=1, axis=mu))
    backward_feature = np.roll(feature, shift=1, axis=mu + 1)
    return backward_links[None, :, :] * backward_feature


def plaquette_cosine_from_angles(link_angles: np.ndarray) -> np.ndarray:
    x_links = link_angles[0]
    y_links = link_angles[1]
    plaquette_angle = wrap_angle(
        x_links
        + np.roll(y_links, shift=-1, axis=0)
        - np.roll(x_links, shift=-1, axis=1)
        - y_links
    )
    return np.cos(plaquette_angle)


@dataclass
class LConvWeights:
    center: np.ndarray
    forward: np.ndarray
    backward: np.ndarray

    @classmethod
    def random(
        cls,
        in_channels: int,
        out_channels: int,
        rng: np.random.Generator,
        scale: float = 0.1,
    ) -> "LConvWeights":
        return cls(
            center=scale * rng.standard_normal((out_channels, in_channels)),
            forward=scale * rng.standard_normal((2, out_channels, in_channels)),
            backward=scale * rng.standard_normal((2, out_channels, in_channels)),
        )


class LConvLayer:
    def __init__(self, weights: LConvWeights):
        self.weights = weights

    def __call__(self, feature: np.ndarray, links: np.ndarray) -> np.ndarray:
        if feature.ndim != 3:
            raise ValueError("feature must have shape (channels, L, L)")
        if links.shape[0] != 2:
            raise ValueError("links must have shape (2, L, L)")

        out_channels = self.weights.center.shape[0]
        out = np.zeros((out_channels, feature.shape[1], feature.shape[2]), dtype=np.complex128)

        out += np.einsum("oi,ixy->oxy", self.weights.center, feature)

        for mu in range(2):
            transported_forward = transport_forward(feature, links, mu)
            transported_backward = transport_backward(feature, links, mu)
            out += np.einsum("oi,ixy->oxy", self.weights.forward[mu], transported_forward)
            out += np.einsum("oi,ixy->oxy", self.weights.backward[mu], transported_backward)

        return out


class EquivariantModReLU:
    def __init__(self, bias: np.ndarray, eps: float = 1e-12):
        if bias.ndim != 1:
            raise ValueError("bias must have shape (channels,)")
        self.bias = bias.astype(np.float64)
        self.eps = eps

    @classmethod
    def random(cls, num_channels: int, rng: np.random.Generator, scale: float = 0.05) -> "EquivariantModReLU":
        return cls(scale * rng.standard_normal(num_channels))

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        magnitude = np.abs(feature)
        activated = np.maximum(magnitude + self.bias[:, None, None], 0.0)
        scale = activated / np.maximum(magnitude, self.eps)
        return feature * scale


class InvariantReadout:
    def __init__(self, channel_weights: np.ndarray, bias: float = 0.0):
        if channel_weights.ndim != 1:
            raise ValueError("channel_weights must have shape (channels,)")
        self.channel_weights = channel_weights.astype(np.float64)
        self.bias = float(bias)

    @classmethod
    def random(cls, num_channels: int, rng: np.random.Generator, scale: float = 0.1) -> "InvariantReadout":
        return cls(scale * rng.standard_normal(num_channels), 0.0)

    def local_invariant(self, feature: np.ndarray) -> np.ndarray:
        return np.einsum("c,cxy->xy", self.channel_weights, np.abs(feature) ** 2) + self.bias

    def global_invariant(self, feature: np.ndarray) -> float:
        return float(np.mean(self.local_invariant(feature)))


class MinimalU1LCNN:
    def __init__(
        self,
        lconv1: LConvLayer,
        activation1: EquivariantModReLU,
        lconv2: LConvLayer,
        activation2: EquivariantModReLU,
        readout: InvariantReadout,
    ):
        self.lconv1 = lconv1
        self.activation1 = activation1
        self.lconv2 = lconv2
        self.activation2 = activation2
        self.readout = readout

    @classmethod
    def random(
        cls,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        rng: np.random.Generator,
    ) -> "MinimalU1LCNN":
        return cls(
            lconv1=LConvLayer(LConvWeights.random(in_channels, hidden_channels, rng)),
            activation1=EquivariantModReLU.random(hidden_channels, rng),
            lconv2=LConvLayer(LConvWeights.random(hidden_channels, out_channels, rng)),
            activation2=EquivariantModReLU.random(out_channels, rng),
            readout=InvariantReadout.random(out_channels, rng),
        )

    def forward_hidden(self, feature: np.ndarray, links: np.ndarray) -> np.ndarray:
        hidden = self.lconv1(feature, links)
        hidden = self.activation1(hidden)
        hidden = self.lconv2(hidden, links)
        hidden = self.activation2(hidden)
        return hidden

    def forward_local_scalar(self, feature: np.ndarray, links: np.ndarray) -> np.ndarray:
        hidden = self.forward_hidden(feature, links)
        return self.readout.local_invariant(hidden)

    def forward_global_scalar(self, feature: np.ndarray, links: np.ndarray) -> float:
        hidden = self.forward_hidden(feature, links)
        return self.readout.global_invariant(hidden)


def max_covariant_residual(feature_before: np.ndarray, feature_after: np.ndarray, alpha: np.ndarray) -> float:
    expected = gauge_transform_site_feature(feature_before, alpha)
    return float(np.max(np.abs(feature_after - expected)))


def max_invariant_residual(output_before: np.ndarray, output_after: np.ndarray) -> float:
    return float(np.max(np.abs(output_after - output_before)))

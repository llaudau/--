#!/usr/bin/env python3
import math

import numpy as np


TWOPI = 2.0 * math.pi


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + math.pi) % TWOPI - math.pi


def random_gauge_angles(L: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-math.pi, math.pi, size=(L, L))


def gauge_transform_links(links: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if links.ndim != 3 or links.shape[0] != 2:
        raise ValueError("links must have shape (2, L, L)")
    if alpha.shape != links.shape[1:]:
        raise ValueError("alpha must have shape (L, L)")

    transformed = np.empty_like(links)
    transformed[0] = wrap_angle(links[0] + alpha - np.roll(alpha, shift=-1, axis=0))
    transformed[1] = wrap_angle(links[1] + alpha - np.roll(alpha, shift=-1, axis=1))
    return transformed


def plaquette_angles(links: np.ndarray) -> np.ndarray:
    x_links = links[0]
    y_links = links[1]
    return wrap_angle(
        x_links
        + np.roll(y_links, shift=-1, axis=0)
        - np.roll(x_links, shift=-1, axis=1)
        - y_links
    )


def average_plaquette(links: np.ndarray) -> float:
    return float(np.mean(np.cos(plaquette_angles(links))))


def topological_charge(links: np.ndarray) -> float:
    return float(np.sum(plaquette_angles(links)) / TWOPI)


def wilson_loop(links: np.ndarray, dx: int, dy: int) -> float:
    L = links.shape[1]
    values = []
    for x in range(L):
        for y in range(L):
            angle = 0.0
            cx, cy = x, y
            for _ in range(dx):
                angle += links[0, cx, cy]
                cx = (cx + 1) % L
            for _ in range(dy):
                angle += links[1, cx, cy]
                cy = (cy + 1) % L
            for _ in range(dx):
                cx = (cx - 1) % L
                angle -= links[0, cx, cy]
            for _ in range(dy):
                cy = (cy - 1) % L
                angle -= links[1, cx, cy]
            values.append(math.cos(wrap_angle(angle)))
    return float(np.mean(values))


def site_phase_to_matrix(phases: np.ndarray) -> np.ndarray:
    return np.exp(1j * phases)


def transport_site_phase_forward(site_phase: np.ndarray, links: np.ndarray, mu: int) -> np.ndarray:
    if mu not in (0, 1):
        raise ValueError("mu must be 0 or 1")
    shifted = np.roll(site_phase, shift=-1, axis=mu)
    return links[mu] * shifted * np.conjugate(links[mu])


def max_wrapped_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(wrap_angle(a - b))))

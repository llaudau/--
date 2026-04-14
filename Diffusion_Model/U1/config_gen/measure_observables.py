#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np


TWOPI = 2.0 * math.pi


def wrap_angle(theta: np.ndarray | float) -> np.ndarray | float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def bessel_i0(x: float, terms: int = 100) -> float:
    total = 1.0
    term = 1.0
    y = (x * x) / 4.0
    for k in range(1, terms):
        term *= y / (k * k)
        total += term
        if abs(term) < 1e-15:
            break
    return total


def bessel_i1(x: float, terms: int = 100) -> float:
    total = x / 2.0
    term = total
    y = (x * x) / 4.0
    for k in range(1, terms):
        term *= y / (k * (k + 1))
        total += term
        if abs(term) < 1e-15:
            break
    return total


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


def measure_ensemble(configs: np.ndarray, beta: float) -> dict:
    plaquettes = [average_plaquette(cfg) for cfg in configs]
    wilson_1x1 = [wilson_loop(cfg, 1, 1) for cfg in configs]
    wilson_2x2 = [wilson_loop(cfg, 2, 2) for cfg in configs]
    topologies = [topological_charge(cfg) for cfg in configs]

    avg_plaquette_value = float(np.mean(plaquettes))
    plaquette_std = float(np.std(plaquettes, ddof=1)) if len(plaquettes) > 1 else 0.0
    plaquette_sem = plaquette_std / math.sqrt(len(plaquettes)) if len(plaquettes) > 0 else 0.0
    exact_plaquette = float(bessel_i1(beta) / bessel_i0(beta))
    abs_error = abs(avg_plaquette_value - exact_plaquette)
    tolerance = max(5.0 * plaquette_sem, 0.08)

    return {
        "avg_plaquette": avg_plaquette_value,
        "avg_wilson_1x1": float(np.mean(wilson_1x1)),
        "avg_wilson_2x2": float(np.mean(wilson_2x2)),
        "avg_topological_charge": float(np.mean(topologies)),
        "plaquette_std": plaquette_std,
        "plaquette_sem": plaquette_sem,
        "exact_infinite_volume_plaquette": exact_plaquette,
        "plaquette_abs_error": abs_error,
        "plaquette_tolerance": tolerance,
        "pass_exactness_check": abs_error <= tolerance,
    }


def load_configs(npz_path: str) -> tuple[np.ndarray, dict]:
    with np.load(npz_path) as data:
        configs = data["configs"]
        metadata = json.loads(str(data["metadata"]))
    return configs, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure observables on a stored U(1) ensemble.")
    parser.add_argument("npz_path", type=str)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs, metadata = load_configs(args.npz_path)
    beta = float(metadata["beta"])
    diagnostics = measure_ensemble(configs, beta)
    status = "PASS" if diagnostics["pass_exactness_check"] else "FAIL"
    print(f"[{status}] Observable check completed")
    print(f"  input                   : {args.npz_path}")
    print(f"  avg plaquette           : {diagnostics['avg_plaquette']:.6f}")
    print(f"  exact plaquette ref     : {diagnostics['exact_infinite_volume_plaquette']:.6f}")
    print(f"  abs error               : {diagnostics['plaquette_abs_error']:.6f}")
    print(f"  tolerance               : {diagnostics['plaquette_tolerance']:.6f}")
    print(f"  avg W(1,1)              : {diagnostics['avg_wilson_1x1']:.6f}")
    print(f"  avg W(2,2)              : {diagnostics['avg_wilson_2x2']:.6f}")
    print(f"  avg topological charge  : {diagnostics['avg_topological_charge']:.6f}")

    if args.output_json:
        payload = {"metadata": metadata, "diagnostics": diagnostics}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

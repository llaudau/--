#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from u1_lcnn_numpy import (
    MinimalU1LCNN,
    angles_to_complex_links,
    gauge_transform_links_complex,
    gauge_transform_site_feature,
    max_covariant_residual,
    max_invariant_residual,
    plaquette_cosine_from_angles,
    random_covariant_site_feature,
    random_gauge_angles,
)


def run_validation(L: int, seed: int, in_channels: int, hidden_channels: int, out_channels: int) -> dict:
    rng = np.random.default_rng(seed)

    link_angles = rng.uniform(-np.pi, np.pi, size=(2, L, L))
    links = angles_to_complex_links(link_angles)
    feature = random_covariant_site_feature(in_channels, L, rng)
    model = MinimalU1LCNN.random(in_channels, hidden_channels, out_channels, rng)
    alpha = random_gauge_angles(L, rng)

    transformed_links = gauge_transform_links_complex(links, alpha)
    transformed_feature = gauge_transform_site_feature(feature, alpha)

    hidden_before = model.forward_hidden(feature, links)
    hidden_after = model.forward_hidden(transformed_feature, transformed_links)

    local_before = model.forward_local_scalar(feature, links)
    local_after = model.forward_local_scalar(transformed_feature, transformed_links)

    global_before = model.forward_global_scalar(feature, links)
    global_after = model.forward_global_scalar(transformed_feature, transformed_links)

    plaquette_target = plaquette_cosine_from_angles(link_angles)

    return {
        "L": L,
        "seed": seed,
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "hidden_covariance_residual": max_covariant_residual(hidden_before, hidden_after, alpha),
        "local_invariant_residual": max_invariant_residual(local_before, local_after),
        "global_invariant_residual": abs(global_after - global_before),
        "plaquette_target_mean": float(np.mean(plaquette_target)),
        "plaquette_target_std": float(np.std(plaquette_target)),
        "pass": (
            max_covariant_residual(hidden_before, hidden_after, alpha) < 1e-10
            and max_invariant_residual(local_before, local_after) < 1e-10
            and abs(global_after - global_before) < 1e-10
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the minimal U(1) L-CNN reference implementation.")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument("--in-channels", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=4)
    parser.add_argument("--out-channels", type=int, default=3)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_validation(
        L=args.L,
        seed=args.seed,
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
    )

    status = "PASS" if result["pass"] else "FAIL"
    print(f"[{status}] Minimal U(1) L-CNN validation completed")
    print(f"  hidden covariance residual : {result['hidden_covariance_residual']:.3e}")
    print(f"  local invariant residual   : {result['local_invariant_residual']:.3e}")
    print(f"  global invariant residual  : {result['global_invariant_residual']:.3e}")
    print(f"  plaquette target mean      : {result['plaquette_target_mean']:.6f}")
    print(f"  plaquette target std       : {result['plaquette_target_std']:.6f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

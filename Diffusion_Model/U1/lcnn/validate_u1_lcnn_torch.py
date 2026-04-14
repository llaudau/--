#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

import torch

from u1_lcnn_torch import (
    LocalU1LCNN,
    gauge_transform_features,
    max_hidden_covariance_residual,
    max_invariant_residual,
    plaquette_cosine_from_features,
    random_gauge_angles_torch,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def run_validation(L: int, batch_size: int, seed: int, hidden_channels: int, out_channels: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    x = 2.0 * torch.rand(batch_size, 4, L, L) - 1.0
    norm_x = torch.sqrt(torch.clamp(x[:, 0] ** 2 + x[:, 1] ** 2, min=1e-12))
    norm_y = torch.sqrt(torch.clamp(x[:, 2] ** 2 + x[:, 3] ** 2, min=1e-12))
    x[:, 0] = x[:, 0] / norm_x
    x[:, 1] = x[:, 1] / norm_x
    x[:, 2] = x[:, 2] / norm_y
    x[:, 3] = x[:, 3] / norm_y
    x = x.to(device)

    model = LocalU1LCNN(hidden_channels=hidden_channels, out_channels=out_channels).to(device)
    alpha = random_gauge_angles_torch(batch_size, L, device=device, seed=seed + 1)
    x_g = gauge_transform_features(x, alpha)

    with torch.no_grad():
        hidden_before = model.forward_hidden(x)
        hidden_after = model.forward_hidden(x_g)
        output_before = model(x)
        output_after = model(x_g)
        plaquette = plaquette_cosine_from_features(x)

    return {
        "L": L,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "hidden_channels": hidden_channels,
        "out_channels": out_channels,
        "hidden_covariance_residual": max_hidden_covariance_residual(hidden_before, hidden_after, alpha),
        "local_invariant_residual": max_invariant_residual(output_before, output_after),
        "plaquette_mean": float(torch.mean(plaquette).item()),
        "plaquette_std": float(torch.std(plaquette).item()),
        "pass": (
            max_hidden_covariance_residual(hidden_before, hidden_after, alpha) < 1e-5
            and max_invariant_residual(output_before, output_after) < 1e-5
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the PyTorch U(1) L-CNN implementation.")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    result = run_validation(args.L, args.batch_size, args.seed, args.hidden_channels, args.out_channels, device)
    status = "PASS" if result["pass"] else "FAIL"
    print(f"[{status}] PyTorch U(1) L-CNN validation completed")
    print(f"  device                     : {result['device']}")
    print(f"  hidden covariance residual : {result['hidden_covariance_residual']:.3e}")
    print(f"  local invariant residual   : {result['local_invariant_residual']:.3e}")
    print(f"  plaquette mean             : {result['plaquette_mean']:.6f}")
    print(f"  plaquette std              : {result['plaquette_std']:.6f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

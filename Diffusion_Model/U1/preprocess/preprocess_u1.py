#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np


def angles_to_cossin(configs: np.ndarray) -> np.ndarray:
    # Input shape: (N, 2, L, L), output shape: (N, 4, L, L)
    x = configs[:, 0]
    y = configs[:, 1]
    return np.stack(
        [
            np.cos(x),
            np.sin(x),
            np.cos(y),
            np.sin(y),
        ],
        axis=1,
    )


def cossin_to_angles(features: np.ndarray) -> np.ndarray:
    # Input shape: (N, 4, L, L), output shape: (N, 2, L, L)
    x = np.arctan2(features[:, 1], features[:, 0])
    y = np.arctan2(features[:, 3], features[:, 2])
    return np.stack([x, y], axis=1)


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def max_angle_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = wrap_angle(reconstructed - original)
    return float(np.max(np.abs(diff)))


def split_indices(num_samples: int, train_ratio: float, val_ratio: float, block_size: int) -> dict[str, np.ndarray]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    blocks = [np.arange(i, min(i + block_size, num_samples)) for i in range(0, num_samples, block_size)]
    n_blocks = len(blocks)
    n_train = max(1, int(round(train_ratio * n_blocks)))
    n_val = int(round(val_ratio * n_blocks))
    if n_train + n_val >= n_blocks:
        n_val = max(0, n_blocks - n_train - 1)
    n_test = max(0, n_blocks - n_train - n_val)
    if n_test == 0 and n_blocks >= 2:
        if n_val > 0:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        n_test = 1

    train_blocks = blocks[:n_train]
    val_blocks = blocks[n_train:n_train + n_val]
    test_blocks = blocks[n_train + n_val:]

    def cat_or_empty(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

    return {
        "train": cat_or_empty(train_blocks),
        "val": cat_or_empty(val_blocks),
        "test": cat_or_empty(test_blocks),
    }


def save_split_npz(path: str, features: np.ndarray, indices: np.ndarray, metadata: dict) -> None:
    np.savez_compressed(
        path,
        features=features[indices],
        indices=indices,
        metadata=json.dumps(metadata),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess U(1) configurations into cos/sin features.")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/config_gen/output/u1_mc_configs.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/preprocess/output",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--block-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with np.load(args.input) as data:
        configs = data["configs"]
        raw_metadata = json.loads(str(data["metadata"]))

    features = angles_to_cossin(configs)
    reconstructed = cossin_to_angles(features)
    recon_error = max_angle_reconstruction_error(configs, reconstructed)
    split = split_indices(
        num_samples=configs.shape[0],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        block_size=args.block_size,
    )

    metadata = {
        "source_input": args.input,
        "raw_metadata": raw_metadata,
        "representation": "cos_sin_per_direction",
        "input_shape": list(configs.shape),
        "feature_shape": list(features.shape),
        "channel_layout": ["cos_x", "sin_x", "cos_y", "sin_y"],
        "block_size": args.block_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_train": int(split["train"].shape[0]),
        "num_val": int(split["val"].shape[0]),
        "num_test": int(split["test"].shape[0]),
        "max_reconstruction_error": recon_error,
        "reconstruction_pass": recon_error < 1e-12,
    }

    np.savez_compressed(
        os.path.join(args.output_dir, "u1_features_all.npz"),
        features=features,
        metadata=json.dumps(metadata),
    )
    save_split_npz(os.path.join(args.output_dir, "u1_train.npz"), features, split["train"], metadata)
    save_split_npz(os.path.join(args.output_dir, "u1_val.npz"), features, split["val"], metadata)
    save_split_npz(os.path.join(args.output_dir, "u1_test.npz"), features, split["test"], metadata)

    with open(os.path.join(args.output_dir, "preprocess_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    status = "PASS" if metadata["reconstruction_pass"] else "FAIL"
    print(f"[{status}] U(1) preprocessing completed")
    print(f"  input                  : {args.input}")
    print(f"  output_dir             : {args.output_dir}")
    print(f"  raw shape              : {tuple(configs.shape)}")
    print(f"  feature shape          : {tuple(features.shape)}")
    print(f"  train/val/test         : {metadata['num_train']}/{metadata['num_val']}/{metadata['num_test']}")
    print(f"  block size             : {args.block_size}")
    print(f"  max recon error        : {metadata['max_reconstruction_error']:.3e}")


if __name__ == "__main__":
    main()

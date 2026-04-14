#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from u1_lcnn_torch import LocalU1LCNN, gauge_transform_features


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def cossin_to_angles(features: np.ndarray) -> np.ndarray:
    x = np.arctan2(features[:, 1], features[:, 0])
    y = np.arctan2(features[:, 3], features[:, 2])
    return np.stack([x, y], axis=1)


def plaquette_cosine_from_angles(angles: np.ndarray) -> np.ndarray:
    x_links = angles[:, 0]
    y_links = angles[:, 1]
    plaquette = wrap_angle(
        x_links
        + np.roll(y_links, shift=-1, axis=1)
        - np.roll(x_links, shift=-1, axis=2)
        - y_links
    )
    return np.cos(plaquette)[:, None, :, :]


class U1PlaquetteDataset(Dataset):
    def __init__(self, npz_path: str):
        with np.load(npz_path) as data:
            features = data["features"].astype(np.float32)
        angles = cossin_to_angles(features.astype(np.float64))
        targets = plaquette_cosine_from_angles(angles).astype(np.float32)
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass
class EvalMetrics:
    loss: float
    mae: float


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalMetrics:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            mse_sum += float(torch.sum((pred - y) ** 2).item())
            mae_sum += float(torch.sum(torch.abs(pred - y)).item())
            count += int(y.numel())
    return EvalMetrics(loss=mse_sum / count, mae=mae_sum / count)


def gauge_robustness_mae(model: nn.Module, dataset: U1PlaquetteDataset, device: torch.device, seed: int) -> float:
    model.eval()
    x = dataset.features.to(device)
    alpha = (2.0 * math.pi) * torch.rand((x.shape[0], x.shape[-1], x.shape[-1])) - math.pi
    alpha = alpha.to(device)
    x_g = gauge_transform_features(x, alpha)
    with torch.no_grad():
        pred = model(x)
        pred_g = model(x_g)
    return float(torch.mean(torch.abs(pred - pred_g)).item())


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)

    train_dataset = U1PlaquetteDataset(args.train_npz)
    val_dataset = U1PlaquetteDataset(args.val_npz)
    test_dataset = U1PlaquetteDataset(args.test_npz)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = LocalU1LCNN(hidden_channels=args.hidden_channels, out_channels=args.out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse_sum = 0.0
        train_mae_sum = 0.0
        count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_mse_sum += float(torch.sum((pred - y) ** 2).item())
            train_mae_sum += float(torch.sum(torch.abs(pred - y)).item())
            count += int(y.numel())

        train_loss = train_mse_sum / count
        train_mae = train_mae_sum / count
        val_metrics = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_metrics.loss,
                "val_mae": val_metrics.mae,
            }
        )

        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6e} train_mae={train_mae:.6e} "
            f"val_loss={val_metrics.loss:.6e} val_mae={val_metrics.mae:.6e}"
        )

    if best_state is None:
        raise RuntimeError("best_state was not captured during training")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    val_gauge_mae = gauge_robustness_mae(model, val_dataset, device, args.seed + 1)
    test_gauge_mae = gauge_robustness_mae(model, test_dataset, device, args.seed + 2)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "u1_lcnn_local_plaquette.pt")
    metrics_path = os.path.join(args.output_dir, "u1_lcnn_local_plaquette_metrics.json")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, checkpoint_path)

    result = {
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "test_npz": args.test_npz,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_channels": args.hidden_channels,
        "out_channels": args.out_channels,
        "seed": args.seed,
        "best_val_loss": best_val,
        "test_loss": test_metrics.loss,
        "test_mae": test_metrics.mae,
        "val_gauge_prediction_mae": val_gauge_mae,
        "test_gauge_prediction_mae": test_gauge_mae,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch U(1) L-CNN on local plaquette regression.")
    parser.add_argument("--train-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/preprocess/output/u1_train.npz")
    parser.add_argument("--val-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/preprocess/output/u1_val.npz")
    parser.add_argument("--test-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/preprocess/output/u1_test.npz")
    parser.add_argument("--output-dir", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/lcnn/output")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print("[DONE] U(1) L-CNN local plaquette training completed")
    print(f"  device                    : {result['device']}")
    print(f"  best val loss             : {result['best_val_loss']:.6e}")
    print(f"  test loss                 : {result['test_loss']:.6e}")
    print(f"  test mae                  : {result['test_mae']:.6e}")
    print(f"  val gauge prediction mae  : {result['val_gauge_prediction_mae']:.6e}")
    print(f"  test gauge prediction mae : {result['test_gauge_prediction_mae']:.6e}")
    print(f"  checkpoint                : {result['checkpoint_path']}")


if __name__ == "__main__":
    main()

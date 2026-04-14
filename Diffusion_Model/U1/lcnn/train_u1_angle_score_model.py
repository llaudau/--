#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from u1_score_model_torch import sample_log_uniform_sigma
from u1_angle_score_model_torch import (
    TimeConditionedU1AngleScoreNet,
    angles_to_cossin_torch,
    cossin_to_angles_torch,
    denoise_angles_with_score,
    sample_wrapped_noisy_angles,
    tangent_score_matching_loss,
    wrap_angle_torch,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class U1AngleDataset(Dataset):
    def __init__(self, npz_path: str):
        with np.load(npz_path) as data:
            features = torch.from_numpy(data["features"].astype(np.float32))
        self.angles = cossin_to_angles_torch(features)

    def __len__(self) -> int:
        return int(self.angles.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.angles[index]


@dataclass
class EvalMetrics:
    loss: float
    angle_recon_mse: float
    score_mse: float


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def evaluate(
    model: TimeConditionedU1AngleScoreNet,
    loader: DataLoader,
    device: torch.device,
    sigma_min: float,
    sigma_max: float,
    seed: int,
) -> EvalMetrics:
    model.eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    loss_sum = 0.0
    recon_sum = 0.0
    score_sum = 0.0
    batch_count = 0
    elem_count = 0

    with torch.no_grad():
        for clean_angles in loader:
            clean_angles = clean_angles.to(device)
            sigma = sample_log_uniform_sigma(clean_angles.shape[0], sigma_min, sigma_max, device=device, generator=gen)
            noisy_angles, target_score = sample_wrapped_noisy_angles(clean_angles, sigma, generator=gen)
            pred_score = model(angles_to_cossin_torch(noisy_angles), sigma)
            loss = tangent_score_matching_loss(pred_score, target_score, sigma)
            denoised = denoise_angles_with_score(noisy_angles, pred_score, sigma)
            recon_err = wrap_angle_torch(denoised - clean_angles)

            loss_sum += float(loss.item()) * clean_angles.shape[0]
            recon_sum += float(torch.sum(recon_err ** 2).item())
            score_sum += float(torch.sum((pred_score - target_score) ** 2).item())
            batch_count += clean_angles.shape[0]
            elem_count += int(clean_angles.numel())

    return EvalMetrics(
        loss=loss_sum / batch_count,
        angle_recon_mse=recon_sum / elem_count,
        score_mse=score_sum / elem_count,
    )


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)

    train_dataset = U1AngleDataset(args.train_npz)
    val_dataset = U1AngleDataset(args.val_npz)
    test_dataset = U1AngleDataset(args.test_npz)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = TimeConditionedU1AngleScoreNet(
        lcnn_hidden_channels=args.lcnn_hidden_channels,
        lcnn_out_channels=args.lcnn_out_channels,
        sigma_emb_dim=args.sigma_emb_dim,
        sigma_hidden_dim=args.sigma_hidden_dim,
        head_hidden_channels=args.head_hidden_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_recon_sum = 0.0
        train_score_sum = 0.0
        batch_count = 0
        elem_count = 0

        for clean_angles in train_loader:
            clean_angles = clean_angles.to(device)
            sigma = sample_log_uniform_sigma(clean_angles.shape[0], args.sigma_min, args.sigma_max, device=device)
            noisy_angles, target_score = sample_wrapped_noisy_angles(clean_angles, sigma)
            optimizer.zero_grad(set_to_none=True)
            pred_score = model(angles_to_cossin_torch(noisy_angles), sigma)
            loss = tangent_score_matching_loss(pred_score, target_score, sigma)
            loss.backward()
            optimizer.step()

            denoised = denoise_angles_with_score(noisy_angles, pred_score, sigma)
            recon_err = wrap_angle_torch(denoised - clean_angles)
            train_loss_sum += float(loss.item()) * clean_angles.shape[0]
            train_recon_sum += float(torch.sum(recon_err ** 2).item())
            train_score_sum += float(torch.sum((pred_score - target_score) ** 2).item())
            batch_count += clean_angles.shape[0]
            elem_count += int(clean_angles.numel())

        train_loss = train_loss_sum / batch_count
        train_recon_mse = train_recon_sum / elem_count
        train_score_mse = train_score_sum / elem_count
        val_metrics = evaluate(model, val_loader, device, args.sigma_min, args.sigma_max, args.seed + epoch)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_angle_recon_mse": train_recon_mse,
                "train_score_mse": train_score_mse,
                "val_loss": val_metrics.loss,
                "val_angle_recon_mse": val_metrics.angle_recon_mse,
                "val_score_mse": val_metrics.score_mse,
            }
        )

        if val_metrics.loss < best_val:
            best_val = val_metrics.loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6e} train_angle_recon_mse={train_recon_mse:.6e} "
            f"val_loss={val_metrics.loss:.6e} val_angle_recon_mse={val_metrics.angle_recon_mse:.6e}"
        )

    if best_state is None:
        raise RuntimeError("best_state was not captured during training")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, args.sigma_min, args.sigma_max, args.seed + 999)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "u1_angle_score_model.pt")
    metrics_path = os.path.join(args.output_dir, "u1_angle_score_model_metrics.json")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, checkpoint_path)

    result = {
        "train_npz": args.train_npz,
        "val_npz": args.val_npz,
        "test_npz": args.test_npz,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "lcnn_hidden_channels": args.lcnn_hidden_channels,
        "lcnn_out_channels": args.lcnn_out_channels,
        "sigma_emb_dim": args.sigma_emb_dim,
        "sigma_hidden_dim": args.sigma_hidden_dim,
        "head_hidden_channels": args.head_hidden_channels,
        "seed": args.seed,
        "best_val_loss": best_val,
        "test_loss": test_metrics.loss,
        "test_angle_recon_mse": test_metrics.angle_recon_mse,
        "test_score_mse": test_metrics.score_mse,
        "history": history,
        "checkpoint_path": checkpoint_path,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tangent-space U(1) angle score model with wrapped small-noise DSM.")
    parser.add_argument("--train-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/preprocess/u1_train.npz")
    parser.add_argument("--val-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/preprocess/u1_val.npz")
    parser.add_argument("--test-npz", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/l12_n192/preprocess/u1_test.npz")
    parser.add_argument("--output-dir", type=str, default="/Users/wangkehe/Git_repository/Diffusion_Model/U1/experiments/angle_score_model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma-max", type=float, default=0.20)
    parser.add_argument("--lcnn-hidden-channels", type=int, default=16)
    parser.add_argument("--lcnn-out-channels", type=int, default=8)
    parser.add_argument("--sigma-emb-dim", type=int, default=32)
    parser.add_argument("--sigma-hidden-dim", type=int, default=64)
    parser.add_argument("--head-hidden-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print("[DONE] U(1) angle-score-model training completed")
    print(f"  device               : {result['device']}")
    print(f"  best val loss        : {result['best_val_loss']:.6e}")
    print(f"  test loss            : {result['test_loss']:.6e}")
    print(f"  test angle recon mse : {result['test_angle_recon_mse']:.6e}")
    print(f"  test score mse       : {result['test_score_mse']:.6e}")
    print(f"  checkpoint           : {result['checkpoint_path']}")


if __name__ == "__main__":
    main()

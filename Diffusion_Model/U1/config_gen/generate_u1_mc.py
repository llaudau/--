#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass

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


@dataclass
class Measurements:
    avg_plaquette: float
    wilson_1x1: float
    wilson_2x2: float
    topological_charge: float


class U1Lattice2D:
    def __init__(self, L: int, beta: float, rng: np.random.Generator):
        self.L = L
        self.beta = beta
        self.rng = rng
        self.links = rng.uniform(-math.pi, math.pi, size=(2, L, L))

    def plaquette_angles(self) -> np.ndarray:
        x_links = self.links[0]
        y_links = self.links[1]
        return wrap_angle(
            x_links
            + np.roll(y_links, shift=-1, axis=0)
            - np.roll(x_links, shift=-1, axis=1)
            - y_links
        )

    def average_plaquette(self) -> float:
        return float(np.mean(np.cos(self.plaquette_angles())))

    def topological_charge(self) -> float:
        return float(np.sum(self.plaquette_angles()) / TWOPI)

    def wilson_loop(self, dx: int, dy: int) -> float:
        values = []
        for x in range(self.L):
            for y in range(self.L):
                angle = 0.0
                cx, cy = x, y
                for _ in range(dx):
                    angle += self.links[0, cx, cy]
                    cx = (cx + 1) % self.L
                for _ in range(dy):
                    angle += self.links[1, cx, cy]
                    cy = (cy + 1) % self.L
                for _ in range(dx):
                    cx = (cx - 1) % self.L
                    angle -= self.links[0, cx, cy]
                for _ in range(dy):
                    cy = (cy - 1) % self.L
                    angle -= self.links[1, cx, cy]
                values.append(math.cos(wrap_angle(angle)))
        return float(np.mean(values))

    def local_delta_action(self, mu: int, x: int, y: int, new_angle: float) -> float:
        old_angle = self.links[mu, x, y]
        old_total = self._affected_plaquette_cos(mu, x, y)
        self.links[mu, x, y] = wrap_angle(new_angle)
        new_total = self._affected_plaquette_cos(mu, x, y)
        self.links[mu, x, y] = old_angle
        return -self.beta * (new_total - old_total)

    def _affected_plaquette_cos(self, mu: int, x: int, y: int) -> float:
        if mu == 0:
            coords = [(x, y), (x, (y - 1) % self.L)]
        else:
            coords = [(x, y), ((x - 1) % self.L, y)]
        plaq = self.plaquette_angles()
        return sum(math.cos(plaq[cx, cy]) for cx, cy in coords)

    def sweep(self, proposal_width: float) -> float:
        accepts = 0
        total = 2 * self.L * self.L
        for mu in range(2):
            for x in range(self.L):
                for y in range(self.L):
                    proposal = self.links[mu, x, y] + self.rng.uniform(-proposal_width, proposal_width)
                    delta_s = self.local_delta_action(mu, x, y, proposal)
                    if delta_s <= 0.0 or self.rng.random() < math.exp(-delta_s):
                        self.links[mu, x, y] = wrap_angle(proposal)
                        accepts += 1
        return accepts / total

    def snapshot(self) -> np.ndarray:
        return np.array(self.links, copy=True)

    def measure(self) -> Measurements:
        return Measurements(
            avg_plaquette=self.average_plaquette(),
            wilson_1x1=self.wilson_loop(1, 1),
            wilson_2x2=self.wilson_loop(2, 2),
            topological_charge=self.topological_charge(),
        )


def run_simulation(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    lat = U1Lattice2D(args.L, args.beta, rng)
    acceptance_history: list[float] = []
    plaquette_history: list[float] = []
    configs: list[np.ndarray] = []
    measurements: list[Measurements] = []

    for _ in range(args.thermalization_sweeps):
        acceptance_history.append(lat.sweep(args.proposal_width))

    for _ in range(args.num_configs):
        for _ in range(args.skip_sweeps):
            acceptance_history.append(lat.sweep(args.proposal_width))
        meas = lat.measure()
        plaquette_history.append(meas.avg_plaquette)
        measurements.append(meas)
        configs.append(lat.snapshot())

    avg_plaquette = float(np.mean([m.avg_plaquette for m in measurements]))
    avg_w11 = float(np.mean([m.wilson_1x1 for m in measurements]))
    avg_w22 = float(np.mean([m.wilson_2x2 for m in measurements]))
    avg_topo = float(np.mean([m.topological_charge for m in measurements]))
    exact_plaquette = float(bessel_i1(args.beta) / bessel_i0(args.beta))
    plaquette_std = float(np.std([m.avg_plaquette for m in measurements], ddof=1)) if len(measurements) > 1 else 0.0
    plaquette_sem = plaquette_std / math.sqrt(len(measurements)) if len(measurements) > 0 else 0.0
    abs_error = abs(avg_plaquette - exact_plaquette)
    tolerance = max(5.0 * plaquette_sem, 0.08)
    passed = abs_error <= tolerance

    return {
        "metadata": {
            "L": args.L,
            "beta": args.beta,
            "thermalization_sweeps": args.thermalization_sweeps,
            "skip_sweeps": args.skip_sweeps,
            "num_configs": args.num_configs,
            "proposal_width": args.proposal_width,
            "seed": args.seed,
        },
        "configs": np.array(configs, dtype=np.float64),
        "diagnostics": {
            "avg_plaquette": avg_plaquette,
            "avg_wilson_1x1": avg_w11,
            "avg_wilson_2x2": avg_w22,
            "avg_topological_charge": avg_topo,
            "avg_acceptance_rate": float(np.mean(acceptance_history)),
            "plaquette_std": plaquette_std,
            "plaquette_sem": plaquette_sem,
            "exact_infinite_volume_plaquette": exact_plaquette,
            "plaquette_abs_error": abs_error,
            "plaquette_tolerance": tolerance,
            "pass_exactness_check": passed,
            "plaquette_history": plaquette_history,
        },
    }


def save_outputs(result: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "u1_mc_configs.npz"),
        configs=result["configs"],
        metadata=json.dumps(result["metadata"]),
    )
    with open(os.path.join(output_dir, "u1_mc_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": result["metadata"],
                "diagnostics": result["diagnostics"],
            },
            f,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a trusted 2D U(1) Metropolis ensemble.")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--thermalization-sweeps", type=int, default=600)
    parser.add_argument("--skip-sweeps", type=int, default=25)
    parser.add_argument("--num-configs", type=int, default=64)
    parser.add_argument("--proposal-width", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_simulation(args)
    save_outputs(result, args.output_dir)
    diag = result["diagnostics"]
    status = "PASS" if diag["pass_exactness_check"] else "FAIL"
    print(f"[{status}] U(1) Metropolis generation completed")
    print(f"  output_dir              : {args.output_dir}")
    print(f"  avg plaquette           : {diag['avg_plaquette']:.6f}")
    print(f"  exact plaquette ref     : {diag['exact_infinite_volume_plaquette']:.6f}")
    print(f"  abs error               : {diag['plaquette_abs_error']:.6f}")
    print(f"  tolerance               : {diag['plaquette_tolerance']:.6f}")
    print(f"  avg W(1,1)              : {diag['avg_wilson_1x1']:.6f}")
    print(f"  avg W(2,2)              : {diag['avg_wilson_2x2']:.6f}")
    print(f"  avg topological charge  : {diag['avg_topological_charge']:.6f}")
    print(f"  avg acceptance rate     : {diag['avg_acceptance_rate']:.6f}")


if __name__ == "__main__":
    main()

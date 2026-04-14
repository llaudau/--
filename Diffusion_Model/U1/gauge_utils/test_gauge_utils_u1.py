#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np

from gauge_utils_u1 import (
    average_plaquette,
    gauge_transform_links,
    max_wrapped_difference,
    plaquette_angles,
    random_gauge_angles,
    site_phase_to_matrix,
    topological_charge,
    transport_site_phase_forward,
    wilson_loop,
)


def run_tests(L: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    links = rng.uniform(-np.pi, np.pi, size=(2, L, L))
    alpha = random_gauge_angles(L, rng)
    transformed = gauge_transform_links(links, alpha)

    plaquette_before = plaquette_angles(links)
    plaquette_after = plaquette_angles(transformed)
    plaquette_angle_diff = max_wrapped_difference(plaquette_before, plaquette_after)

    avg_plaq_before = average_plaquette(links)
    avg_plaq_after = average_plaquette(transformed)
    avg_plaq_diff = abs(avg_plaq_before - avg_plaq_after)

    w11_before = wilson_loop(links, 1, 1)
    w11_after = wilson_loop(transformed, 1, 1)
    w11_diff = abs(w11_before - w11_after)

    w22_before = wilson_loop(links, 2, 2)
    w22_after = wilson_loop(transformed, 2, 2)
    w22_diff = abs(w22_before - w22_after)

    topo_before = topological_charge(links)
    topo_after = topological_charge(transformed)
    topo_diff = abs(topo_before - topo_after)

    site_phase = site_phase_to_matrix(rng.uniform(-np.pi, np.pi, size=(L, L)))
    transformed_site_phase = np.exp(1j * alpha) * site_phase * np.exp(-1j * alpha)
    transported_before = transport_site_phase_forward(site_phase, np.exp(1j * links), mu=0)
    transported_after = transport_site_phase_forward(transformed_site_phase, np.exp(1j * transformed), mu=0)
    expected_after = np.exp(1j * alpha) * transported_before * np.exp(-1j * alpha)
    transported_cov_diff = float(np.max(np.abs(transported_after - expected_after)))

    tol = 1e-10
    return {
        "L": L,
        "seed": seed,
        "plaquette_angle_diff": plaquette_angle_diff,
        "avg_plaquette_diff": avg_plaq_diff,
        "wilson_1x1_diff": w11_diff,
        "wilson_2x2_diff": w22_diff,
        "topological_charge_diff": topo_diff,
        "transport_covariance_diff": transported_cov_diff,
        "pass": all(
            diff < tol
            for diff in [
                plaquette_angle_diff,
                avg_plaq_diff,
                w11_diff,
                w22_diff,
                topo_diff,
                transported_cov_diff,
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test U(1) gauge invariance / covariance utilities.")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tests(args.L, args.seed)
    status = "PASS" if result["pass"] else "FAIL"
    print(f"[{status}] U(1) gauge utility tests completed")
    print(f"  plaquette angle diff      : {result['plaquette_angle_diff']:.3e}")
    print(f"  avg plaquette diff        : {result['avg_plaquette_diff']:.3e}")
    print(f"  wilson 1x1 diff           : {result['wilson_1x1_diff']:.3e}")
    print(f"  wilson 2x2 diff           : {result['wilson_2x2_diff']:.3e}")
    print(f"  topological charge diff   : {result['topological_charge_diff']:.3e}")
    print(f"  transport covariance diff : {result['transport_covariance_diff']:.3e}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()

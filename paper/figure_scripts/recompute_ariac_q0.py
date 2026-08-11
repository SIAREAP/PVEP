#!/usr/bin/env python3
"""Recompute the repository-frozen ARIAC first-check-admissibility result."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "ariac" / "ariac_q0_definition_audit.csv"
EXPECTED_COLUMNS = {
    "order_id",
    "eps",
    "n_subgoals",
    "n_pass_static_first_check",
    "q0_static",
    "definition",
}


def main() -> None:
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected columns: {reader.fieldnames}")
        rows = list(reader)

    if len(rows) != 50:
        raise ValueError(f"Expected 50 trials, found {len(rows)}")
    order_ids = [row["order_id"] for row in rows]
    if len(set(order_ids)) != 50:
        raise ValueError("order_id values must be unique")

    q_values: list[float] = []
    total_subgoals = 0
    total_passed = 0
    for row in rows:
        if not math.isclose(float(row["eps"]), 0.0, abs_tol=1e-12):
            raise ValueError(f"Nonzero epsilon in {row['order_id']}")
        if row["definition"] != "initial_requirement_truth_match":
            raise ValueError(f"Unexpected definition in {row['order_id']}")
        n_subgoals = int(row["n_subgoals"])
        n_passed = int(row["n_pass_static_first_check"])
        q0_static = float(row["q0_static"])
        if n_subgoals <= 0 or not 0 <= n_passed <= n_subgoals:
            raise ValueError(f"Invalid counts in {row['order_id']}")
        expected_q0 = n_passed / n_subgoals
        if not math.isclose(q0_static, expected_q0, abs_tol=1e-12):
            raise ValueError(
                f"q0_static mismatch in {row['order_id']}: "
                f"{q0_static} != {n_passed}/{n_subgoals}"
            )
        q_values.append(q0_static)
        total_subgoals += n_subgoals
        total_passed += n_passed

    macro = sum(q_values) / len(q_values)
    micro = total_passed / total_subgoals
    result = {
        "source": str(SOURCE.relative_to(ROOT)),
        "n_trials": len(rows),
        "n_subgoals": total_subgoals,
        "n_pass_static_first_check": total_passed,
        "q0_macro": macro,
        "q0_macro_reported": round(macro, 3),
        "q0_micro_descriptive": micro,
        "primary_aggregation": "trial-level macro mean",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

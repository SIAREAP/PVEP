from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_margin_pvep_demo import (
    BELIEF_SCOPES,
    MarginTaskRun,
    build_demo_from_task_row,
    method_cost_components,
    run_method,
)
from run_margin_pvep_tasks import _write_details, _write_inspect_events, _write_summary


RISK_THRESHOLD = 0.3
DEFAULT_NARROW_WINDOW_C = 15
SCOPE_ORDER = ("broad", "narrow")
CONFIG_ORDER = ("full", "no-reflow", "no-pomcp")
CONFIG_METHOD = {
    "full": "llm+pomdp+reflow",
    "no-reflow": "llm+pomdp+no_reflow",
    "no-pomcp": "llm+reflow_no_pomcp",
}
CONFIG_SOURCE_KEY = {
    "full": "full",
    "no-reflow": "no_reflow",
    "no-pomcp": "no_pomcp",
}
BASE_RESULT_FIELDS = (
    "task_id",
    "material",
    "seed",
    "actual_required_temp_c",
)

WIDE_CONFIG_PREFIX = {
    "full": "full",
    "no-reflow": "no_reflow",
    "no-pomcp": "no_pomcp",
}

GROUPED_METHODS = (
    ("Full", "full"),
    ("No-Reflow", "no_reflow"),
    ("No-POMCP", "no_pomcp"),
)
GROUPED_METRICS = (
    ("RVR", "rvr"),
    ("Submit temp (C)", "submission_temp_c"),
    ("Cost", "cost"),
    ("Inspect cost", "inspect_cost"),
)
FLAT_RESULT_FIELDS = BASE_RESULT_FIELDS + tuple(
    f"{method_prefix}_{scope}_{metric_key}"
    for _method_label, method_prefix in GROUPED_METHODS
    for scope in SCOPE_ORDER
    for _metric_label, metric_key in GROUPED_METRICS
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_csv_list(value: str, *, allowed: Sequence[str], flag: str) -> tuple[str, ...]:
    items = tuple(x.strip() for x in str(value).split(",") if x.strip())
    invalid = tuple(x for x in items if x not in set(allowed))
    if not items:
        raise ValueError(f"{flag} cannot be empty")
    if invalid:
        raise ValueError(f"{flag} contains invalid values {invalid}; allowed={tuple(allowed)}")
    if len(set(items)) != len(items):
        raise ValueError(f"{flag} contains duplicate values: {items}")
    return items


def _index_tasks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError("tasks CSV contains an empty task_id")
        if task_id in indexed:
            raise ValueError(f"tasks CSV contains duplicate task_id={task_id!r}")
        if not str(row.get("world_seed", "")).strip():
            raise ValueError(f"task_id={task_id!r} has no world_seed")
        indexed[task_id] = row
    return indexed


def _load_proposal_map(
    path: Path,
    *,
    expected_task_ids: Sequence[str],
    task_index: Mapping[str, Mapping[str, Any]],
    expected_method: str,
    expected_prompt_type: str,
) -> dict[str, int]:
    rows = _read_csv(path)
    proposals: dict[str, int] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"proposal source {path} contains an empty task_id")
        if task_id in proposals:
            raise ValueError(f"proposal source {path} contains duplicate task_id={task_id!r}")
        if task_id in set(expected_task_ids):
            if str(row.get("method", "")) != str(expected_method):
                raise ValueError(
                    f"proposal source {path} has method={row.get('method')!r} for "
                    f"task_id={task_id!r}; expected {expected_method!r}"
                )
            if str(row.get("prompt_type", "")) != str(expected_prompt_type):
                raise ValueError(
                    f"proposal source {path} has prompt_type={row.get('prompt_type')!r} "
                    f"for task_id={task_id!r}; expected {expected_prompt_type!r}"
                )
            task = task_index[task_id]
            if str(row.get("material", "")) != str(task.get("material", "")):
                raise ValueError(
                    f"proposal source {path} material mismatch for task_id={task_id!r}"
                )
            if not math.isclose(
                float(row["t_needed_c"]),
                float(task["t_needed_c"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    f"proposal source {path} t_needed_c mismatch for task_id={task_id!r}"
                )
        proposals[task_id] = int(float(row["proposal_temp_c"]))

    expected = set(str(x) for x in expected_task_ids)
    actual = set(proposals)
    if not expected.issubset(actual):
        missing = sorted(expected - actual)
        raise ValueError(
            f"proposal source {path} does not align with tasks; "
            f"missing={missing[:5]}"
        )
    return {task_id: int(proposals[task_id]) for task_id in expected_task_ids}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_cell(payload: Mapping[str, Any]):
    scope = str(payload["scope"])
    config = str(payload["config"])
    method = str(CONFIG_METHOD[config])
    task_rows = tuple(dict(row) for row in payload["task_rows"])
    proposal_map = {str(k): int(v) for k, v in dict(payload["proposal_map"]).items()}
    narrow_window_c = int(payload["narrow_window_c"])
    pomcp_max_depth = int(payload["pomcp_max_depth"])
    pomcp_max_it = int(payload["pomcp_max_it"])
    pomcp_particles = int(payload["pomcp_particles"])
    proposal_source_file = str(payload["proposal_source_file"])
    proposal_source_sha256 = str(payload["proposal_source_sha256"])

    runs: list[MarginTaskRun] = []
    details: list[dict[str, Any]] = []
    for row in task_rows:
        task_id = str(row["task_id"])
        seed = int(float(row["world_seed"]))
        planner_seed = _derive_stream_seed(seed, config=config, stream="planner")
        observation_seed = _derive_stream_seed(
            seed,
            config=config,
            stream="real-observation",
        )
        proposal_temp_c = int(proposal_map[task_id])
        demo = build_demo_from_task_row(
            row,
            proposal_temp_c=int(proposal_temp_c),
            assembly_cooling_c=10,
            delta_step_c=5,
            required_temp_min_c=250,
            required_temp_max_c=440,
            risk_threshold=float(RISK_THRESHOLD),
            belief_scope=str(scope),
            belief_window_c=int(narrow_window_c),
        )
        result = run_method(
            method,
            demo,
            prompt_history=bool(config in {"full", "no-pomcp"}),
            pomcp_max_depth=int(pomcp_max_depth),
            pomcp_max_it=int(pomcp_max_it),
            pomcp_particles=int(pomcp_particles),
            random_seed=int(planner_seed),
            observation_seed=int(observation_seed),
        )
        run = MarginTaskRun(
            task_id=str(task_id),
            material=str(row["material"]),
            t_needed_c=float(row["t_needed_c"]),
            proposal_temp_c=int(proposal_temp_c),
            demo=demo,
            method_result=result,
        )
        runs.append(run)

        required_bins = tuple(
            int(demo.target_temp_c) + int(delta) for delta in demo.delta_bins_c
        )
        true_required_bin_c = int(math.ceil(float(row["t_needed_c"]) / 5.0) * 5)
        costs = method_cost_components(result.episode, demo)
        details.append(
            {
                "task_id": str(task_id),
                "material": str(row["material"]),
                "seed": int(seed),
                "scope": str(scope),
                "config": str(config),
                "rvr": int(result.episode.rvr),
                "pass": int(bool(result.episode.success)),
                "n_heat": int(result.n_heat),
                "n_inspect": int(result.n_inspect),
                "total_cost": float(result.total_cost),
                "inspect_cost": float(costs["inspect_action_cost"]),
                "risk_threshold": float(RISK_THRESHOLD),
                "belief_window_c": int(narrow_window_c),
                "planner_seed": int(planner_seed),
                "observation_seed": int(observation_seed),
                "proposal_source_file": str(proposal_source_file),
                "proposal_source_sha256": str(proposal_source_sha256),
                "proposal_temp_c": int(proposal_temp_c),
                "t_needed_c": float(row["t_needed_c"]),
                "true_required_bin_c": int(true_required_bin_c),
                "belief_min_required_c": int(min(required_bins)),
                "belief_max_required_c": int(max(required_bins)),
                "belief_n_bins": int(len(required_bins)),
                "truth_in_belief_support": int(true_required_bin_c in set(required_bins)),
                "aborted": int(bool(result.episode.aborted)),
                "final_temp_c": int(result.final_temp_c),
                "n_assemble": int(result.n_assemble),
                "actions": " ".join(step.action.value for step in result.episode.trace),
            }
        )
    return str(scope), str(config), tuple(runs), tuple(details)


def _derive_stream_seed(world_seed: int, *, config: str, stream: str) -> int:
    payload = f"{int(world_seed)}|{str(config)}|{str(stream)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _write_csv(path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_name = str(f.name)
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def _validate_formal_inputs(
    *,
    scopes: Sequence[str],
    configs: Sequence[str],
    narrow_window_c: int,
    task_count: int,
) -> None:
    if set(scopes) != set(SCOPE_ORDER) or len(scopes) != len(SCOPE_ORDER):
        raise ValueError(f"formal run requires scopes={SCOPE_ORDER}")
    if set(configs) != set(CONFIG_ORDER) or len(configs) != len(CONFIG_ORDER):
        raise ValueError(f"formal run requires configs={CONFIG_ORDER}")
    if int(narrow_window_c) != int(DEFAULT_NARROW_WINDOW_C):
        raise ValueError(
            "formal run is preregistered at "
            f"narrow_window_c={DEFAULT_NARROW_WINDOW_C}, got {narrow_window_c}"
        )
    if int(task_count) != 90:
        raise ValueError(f"formal run requires exactly 90 unique tasks, found {task_count}")


def _summarize(details: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in details:
        grouped.setdefault((str(row["scope"]), str(row["config"])), []).append(row)

    rows: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for config in CONFIG_ORDER:
            xs = grouped.get((scope, config), [])
            if not xs:
                continue
            n = len(xs)
            rows.append(
                {
                    "scope": scope,
                    "config": config,
                    "n_trials": int(n),
                    "pass_count": int(sum(int(x["pass"]) for x in xs)),
                    "pass_rate": sum(float(x["pass"]) for x in xs) / float(n),
                    "rvr_count": int(sum(int(x["rvr"]) for x in xs)),
                    "rvr_rate": sum(float(x["rvr"]) for x in xs) / float(n),
                    "avg_n_heat": sum(float(x["n_heat"]) for x in xs) / float(n),
                    "avg_n_inspect": sum(float(x["n_inspect"]) for x in xs) / float(n),
                    "avg_total_cost": sum(float(x["total_cost"]) for x in xs) / float(n),
                    "truth_in_belief_count": int(
                        sum(int(x["truth_in_belief_support"]) for x in xs)
                    ),
                    "truth_in_belief_rate": sum(
                        float(x["truth_in_belief_support"]) for x in xs
                    )
                    / float(n),
                }
            )
    return rows


def _build_grouped_result_rows(
    details: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pivot each task into method/scope groups with four outcome columns."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in details:
        grouped.setdefault(str(row["task_id"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for task_id in sorted(grouped):
        task_rows = grouped[task_id]
        materials = {str(row["material"]) for row in task_rows}
        seeds = {int(float(row["seed"])) for row in task_rows}
        required_temps = {float(row["t_needed_c"]) for row in task_rows}
        risks = {
            float(row.get("risk_threshold", RISK_THRESHOLD)) for row in task_rows
        }
        if (
            len(materials) != 1
            or len(seeds) != 1
            or len(required_temps) != 1
            or risks != {RISK_THRESHOLD}
        ):
            raise AssertionError(
                f"wide-table pairing metadata mismatch for task_id={task_id!r}: "
                f"materials={materials}, seeds={seeds}, "
                f"required_temps={required_temps}, risks={risks}"
            )

        output: dict[str, Any] = {
            "task_id": task_id,
            "material": next(iter(materials)),
            "seed": next(iter(seeds)),
            "actual_required_temp_c": next(iter(required_temps)),
        }
        seen: set[tuple[str, str]] = set()
        for row in task_rows:
            scope = str(row["scope"])
            config = str(row["config"])
            key = (scope, config)
            if key in seen:
                raise AssertionError(
                    f"duplicate paired result for task_id={task_id!r}, cell={key}"
                )
            seen.add(key)
            prefix = f"{WIDE_CONFIG_PREFIX[config]}_{scope}"
            output[f"{prefix}_rvr"] = int(row["rvr"])
            output[f"{prefix}_submission_temp_c"] = int(row["final_temp_c"])
            output[f"{prefix}_cost"] = float(row["total_cost"])
            inspect_cost = row.get("inspect_cost", row.get("inspect_action_cost"))
            if inspect_cost is None:
                raise AssertionError(
                    f"missing inspect cost for task_id={task_id!r}, cell={key}"
                )
            output[f"{prefix}_inspect_cost"] = float(inspect_cost)

        rows.append(output)
    return rows


def _grouped_header_rows() -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    base_top = ("Task ID", "Material", "Seed", "Actual required temp (C)")
    top = list(base_top)
    scope_row = [""] * len(BASE_RESULT_FIELDS)
    metric_row = [""] * len(BASE_RESULT_FIELDS)
    columns_per_scope = len(GROUPED_METRICS)
    columns_per_method = len(SCOPE_ORDER) * columns_per_scope
    for method_label, _method_prefix in GROUPED_METHODS:
        top.extend([method_label] + [""] * (columns_per_method - 1))
        for scope in SCOPE_ORDER:
            scope_row.extend(
                [scope.title()] + [""] * (columns_per_scope - 1)
            )
            metric_row.extend(label for label, _key in GROUPED_METRICS)
    return tuple(top), tuple(scope_row), tuple(metric_row)


def _write_grouped_results_csv(
    path: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8-sig",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_name = str(f.name)
            writer = csv.writer(f)
            writer.writerows(_grouped_header_rows())
            for row in rows:
                writer.writerow([row.get(field, "") for field in FLAT_RESULT_FIELDS])
        os.replace(temp_name, path)
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()


def _validate_pairing(details: Sequence[Mapping[str, Any]]) -> None:
    indexed = {
        (str(row["scope"]), str(row["config"]), str(row["task_id"])): row
        for row in details
    }
    configs = sorted({str(row["config"]) for row in details})
    task_ids = sorted({str(row["task_id"]) for row in details})
    scopes = {str(row["scope"]) for row in details}
    if not {"broad", "narrow"}.issubset(scopes):
        return

    for config in configs:
        for task_id in task_ids:
            broad = indexed[("broad", config, task_id)]
            narrow = indexed[("narrow", config, task_id)]
            for field in (
                "material",
                "seed",
                "planner_seed",
                "observation_seed",
                "proposal_source_file",
                "proposal_source_sha256",
                "proposal_temp_c",
                "t_needed_c",
            ):
                if broad[field] != narrow[field]:
                    raise AssertionError(
                        f"scope pairing failed for config={config}, task_id={task_id}, "
                        f"field={field}: broad={broad[field]!r}, narrow={narrow[field]!r}"
                    )

            if config == "no-pomcp":
                for field in (
                    "rvr",
                    "pass",
                    "n_heat",
                    "n_inspect",
                    "total_cost",
                    "final_temp_c",
                    "actions",
                ):
                    if broad[field] != narrow[field]:
                        raise AssertionError(
                            f"no-pomcp must be scope-invariant for task_id={task_id}, "
                            f"field={field}: broad={broad[field]!r}, narrow={narrow[field]!r}"
                        )


def _print_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print(
        "scope  config       n   pass     RVR  avg_heat  avg_inspect  avg_cost  truth_in_support",
        flush=True,
    )
    for row in rows:
        print(
            f"{str(row['scope']):<6} "
            f"{str(row['config']):<12} "
            f"{int(row['n_trials']):>3} "
            f"{float(row['pass_rate']):>6.1%} "
            f"{float(row['rvr_rate']):>7.1%} "
            f"{float(row['avg_n_heat']):>9.2f} "
            f"{float(row['avg_n_inspect']):>12.2f} "
            f"{float(row['avg_total_cost']):>9.2f} "
            f"{float(row['truth_in_belief_rate']):>16.1%}",
            flush=True,
        )


def main() -> None:
    root = Path(__file__).resolve().parent
    frozen_source_dir = (
        root / "results" / "four_methods_risk0p3_noperturb_20260710_143013"
    )
    official_output_csv = root / "tables" / "rotor_scope2x2.csv"
    official_cell_results_dir = root / "results" / "scope2x2_risk0p3" / "cells"
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired rotor/sleeve scope experiment with frozen proposals. "
            "Risk is preregistered and fixed at 0.3."
        )
    )
    parser.add_argument(
        "--tasks-csv",
        type=Path,
        default=root / "data" / "sleeve_tasks_head10x3x3.csv",
    )
    parser.add_argument(
        "--full-proposals-csv",
        type=Path,
        default=frozen_source_dir / "llm_pomdp_reflow_details.csv",
    )
    parser.add_argument(
        "--no-reflow-proposals-csv",
        type=Path,
        default=frozen_source_dir / "llm_details.csv",
    )
    parser.add_argument(
        "--no-pomcp-proposals-csv",
        type=Path,
        default=frozen_source_dir / "llm_reflow_no_pomcp_details.csv",
    )
    parser.add_argument("--scopes", type=str, default=",".join(SCOPE_ORDER))
    parser.add_argument("--configs", type=str, default=",".join(CONFIG_ORDER))
    parser.add_argument("--narrow-window-c", type=int, default=DEFAULT_NARROW_WINDOW_C)
    parser.add_argument("--pomcp-max-depth", type=int, default=20)
    parser.add_argument("--pomcp-max-it", type=int, default=200)
    parser.add_argument("--pomcp-particles", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Allow a partial sanity run; pilot runs cannot write official fixed outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only the first N tasks for sanity checks; 0 runs all 90 tasks.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=official_output_csv,
    )
    parser.add_argument(
        "--cell-results-dir",
        type=Path,
        default=official_cell_results_dir,
    )
    args = parser.parse_args()

    scopes = _parse_csv_list(args.scopes, allowed=BELIEF_SCOPES, flag="--scopes")
    configs = _parse_csv_list(args.configs, allowed=CONFIG_ORDER, flag="--configs")
    if int(args.narrow_window_c) < 0:
        raise ValueError("--narrow-window-c must be non-negative")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")
    if int(args.limit) > 0 and not bool(args.pilot):
        raise ValueError("--limit is only allowed together with --pilot")
    if bool(args.pilot):
        official_paths = (
            official_output_csv.resolve(),
            official_cell_results_dir.resolve(),
        )
        requested_paths = (
            Path(args.output_csv).resolve(),
            Path(args.cell_results_dir).resolve(),
        )
        collisions = [str(path) for path in requested_paths if path in set(official_paths)]
        if collisions:
            raise ValueError(
                "pilot runs cannot write official fixed outputs; choose separate paths: "
                + ", ".join(collisions)
            )
    for value, name in (
        (args.pomcp_max_depth, "--pomcp-max-depth"),
        (args.pomcp_max_it, "--pomcp-max-it"),
        (args.pomcp_particles, "--pomcp-particles"),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")

    task_rows = _read_csv(Path(args.tasks_csv))
    task_index = _index_tasks(task_rows)
    if not bool(args.pilot):
        _validate_formal_inputs(
            scopes=scopes,
            configs=configs,
            narrow_window_c=int(args.narrow_window_c),
            task_count=len(task_rows),
        )
    if int(args.limit) > 0:
        task_rows = task_rows[: int(args.limit)]
    if not task_rows:
        raise ValueError("tasks CSV contains no selected rows")
    task_ids = tuple(str(row["task_id"]) for row in task_rows)
    selected_task_index = {task_id: task_index[task_id] for task_id in task_ids}

    source_paths = {
        "full": Path(args.full_proposals_csv),
        "no_reflow": Path(args.no_reflow_proposals_csv),
        "no_pomcp": Path(args.no_pomcp_proposals_csv),
    }
    source_expectations = {
        "full": ("llm+pomdp+reflow", "reflow"),
        "no_reflow": ("llm", "no_reflow"),
        "no_pomcp": ("llm+reflow_no_pomcp", "reflow"),
    }
    proposal_maps = {
        key: _load_proposal_map(
            path,
            expected_task_ids=task_ids,
            task_index=selected_task_index,
            expected_method=source_expectations[key][0],
            expected_prompt_type=source_expectations[key][1],
        )
        for key, path in source_paths.items()
    }
    source_hashes = {key: _file_sha256(path) for key, path in source_paths.items()}

    jobs: list[dict[str, Any]] = []
    for scope in scopes:
        for config in configs:
            source_key = CONFIG_SOURCE_KEY[config]
            jobs.append(
                {
                    "scope": scope,
                    "config": config,
                    "task_rows": tuple(task_rows),
                    "proposal_map": proposal_maps[source_key],
                    "proposal_source_file": str(source_paths[source_key].resolve()),
                    "proposal_source_sha256": str(source_hashes[source_key]),
                    "narrow_window_c": int(args.narrow_window_c),
                    "pomcp_max_depth": int(args.pomcp_max_depth),
                    "pomcp_max_it": int(args.pomcp_max_it),
                    "pomcp_particles": int(args.pomcp_particles),
                }
            )

    cell_outputs: dict[tuple[str, str], tuple[tuple[MarginTaskRun, ...], tuple[dict[str, Any], ...]]] = {}
    with ProcessPoolExecutor(max_workers=min(int(args.workers), len(jobs))) as executor:
        future_to_cell = {
            executor.submit(_run_cell, job): (str(job["scope"]), str(job["config"]))
            for job in jobs
        }
        for future in as_completed(future_to_cell):
            scope, config, runs, details = future.result()
            cell_outputs[(scope, config)] = (runs, details)
            print(
                f"[done] scope={scope} config={config} trials={len(runs)} ",
                flush=True,
            )

    all_details: list[dict[str, Any]] = []
    for scope in SCOPE_ORDER:
        for config in CONFIG_ORDER:
            cell = cell_outputs.get((scope, config))
            if cell is None:
                continue
            _runs, details = cell
            all_details.extend(dict(row) for row in details)

    _validate_pairing(all_details)
    expected_cells = len(scopes) * len(configs)
    expected_rows = len(task_rows) * expected_cells
    unique_keys = {
        (str(row["scope"]), str(row["config"]), str(row["task_id"]))
        for row in all_details
    }
    if len(cell_outputs) != expected_cells:
        raise AssertionError(
            f"missing experiment cells: expected={expected_cells}, actual={len(cell_outputs)}"
        )
    if len(all_details) != expected_rows or len(unique_keys) != expected_rows:
        raise AssertionError(
            "result row contract failed: "
            f"expected={expected_rows}, rows={len(all_details)}, unique={len(unique_keys)}"
        )
    if not bool(args.pilot):
        for scope in SCOPE_ORDER:
            for config in CONFIG_ORDER:
                runs, details = cell_outputs[(scope, config)]
                if len(runs) != 90 or len(details) != 90:
                    raise AssertionError(
                        f"formal cell {scope}/{config} must contain 90 trials; "
                        f"runs={len(runs)}, details={len(details)}"
                    )
        if len(all_details) != 540:
            raise AssertionError(
                f"formal output must contain 540 rows, found {len(all_details)}"
            )

    for scope in SCOPE_ORDER:
        for config in CONFIG_ORDER:
            cell = cell_outputs.get((scope, config))
            if cell is None:
                continue
            runs, _details = cell
            prefix = Path(args.cell_results_dir) / f"{scope}_{config.replace('-', '_')}"
            _write_details(prefix.with_name(prefix.name + "_details.csv"), runs)
            _write_summary(prefix.with_name(prefix.name + "_summary.csv"), runs)
            _write_inspect_events(prefix.with_name(prefix.name + "_inspect_events.csv"), runs)
    primary_rows = _build_grouped_result_rows(all_details)
    summary_rows = _summarize(all_details)
    _write_grouped_results_csv(Path(args.output_csv), rows=primary_rows)
    _print_summary(summary_rows)
    print(f"risk_threshold={RISK_THRESHOLD}", flush=True)
    print(f"output_csv={Path(args.output_csv)}", flush=True)
    print(f"cell_results_dir={Path(args.cell_results_dir)}", flush=True)


if __name__ == "__main__":
    main()

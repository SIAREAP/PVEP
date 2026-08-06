from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from run_margin_pvep_demo import (
    BELIEF_SCOPES,
    REFLOW_PROMPT_METHODS,
    MarginTaskRun,
    PolicyConfig,
    aggregate_task_results,
    apply_proposal_shift,
    build_demo_from_task_row,
    format_task_aggregate,
    method_cost_components,
    run_margin_tasks_csv,
    run_method,
)

DEFAULT_TASK_METHODS = (
    "llm",
    "llm+pomdp+no_reflow",
)


@dataclass(frozen=True)
class OnlineReflowProposal:
    method: str
    task_id: str
    proposal_temp_c: int
    raw: str
    n_examples_before: int
    success: bool
    final_temp_c: int
    raw_proposal_temp_c: int = 0
    proposal_perturb_eps: float = 0.0
    proposal_perturb_shift_c: int = 0


def _parse_methods(value: str) -> Sequence[str]:
    methods = [x.strip() for x in str(value).split(",") if x.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    return tuple(methods)


def _write_details(path: Path, rows, *, prompt_type: str = "no_history", history_available: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "round_id",
        "material",
        "t_needed_c",
        "true_t_needed_c",
        "proposal_temp_c",
        "prompt_type",
        "history_available",
        "assembly_cooling_c",
        "residual_needed_c",
        "true_residual_needed_c",
        "delta_needed_c",
        "true_delta_needed_c",
        "method",
        "policy_name",
        "fixed_K_if_any",
        "planner",
        "initial_belief_source",
        "initial_belief_entropy",
        "initial_expected_residual_c",
        "initial_expected_delta_c",
        "inspect_model_type",
        "success",
        "rvr",
        "actual_RVR",
        "aborted",
        "final_temp_c",
        "n_heat",
        "n_inspect",
        "n_assemble",
        "extra_heat_c",
        "heat_action_cost",
        "inspect_action_cost",
        "assemble_action_cost",
        "overheat_action_cost",
        "abort_action_cost",
        "total_cost",
        "predicted_tail_risk_before_assemble",
        "actions",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result = row.method_result
            episode = result.episode
            costs = method_cost_components(episode, row.demo)
            writer.writerow(
                {
                    "task_id": row.task_id,
                    "round_id": 0,
                    "material": row.material,
                    "t_needed_c": float(row.t_needed_c),
                    "true_t_needed_c": float(row.t_needed_c),
                    "proposal_temp_c": int(row.proposal_temp_c),
                    "prompt_type": "reflow" if bool(result.prompt_history) else "no_reflow",
                    "history_available": int(bool(result.prompt_history)),
                    "assembly_cooling_c": int(row.demo.assembly_cooling_c),
                    "residual_needed_c": int(row.demo.true_delta_needed_c),
                    "true_residual_needed_c": int(row.demo.true_delta_needed_c),
                    "delta_needed_c": int(row.demo.true_delta_needed_c),
                    "true_delta_needed_c": int(row.demo.true_delta_needed_c),
                    "method": result.method,
                    "policy_name": result.method,
                    "fixed_K_if_any": _fixed_k_from_method(result.method),
                    "planner": result.planner,
                    "initial_belief_source": result.belief_source,
                    "initial_belief_entropy": float(result.initial_belief_entropy),
                    "initial_expected_residual_c": float(result.initial_expected_delta_c),
                    "initial_expected_delta_c": float(result.initial_expected_delta_c),
                    "inspect_model_type": result.inspect_model_type,
                    "success": int(episode.success),
                    "rvr": int(episode.rvr),
                    "actual_RVR": int(episode.rvr),
                    "aborted": int(episode.aborted),
                    "final_temp_c": int(result.final_temp_c),
                    "n_heat": int(result.n_heat),
                    "n_inspect": int(result.n_inspect),
                    "n_assemble": int(result.n_assemble),
                    "extra_heat_c": int(result.extra_heat_c),
                    "heat_action_cost": float(costs["heat_action_cost"]),
                    "inspect_action_cost": float(costs["inspect_action_cost"]),
                    "assemble_action_cost": float(costs["assemble_action_cost"]),
                    "overheat_action_cost": float(costs["overheat_action_cost"]),
                    "abort_action_cost": float(costs["abort_action_cost"]),
                    "total_cost": float(result.total_cost),
                    "predicted_tail_risk_before_assemble": _tail_risk_before_assemble(episode),
                    "actions": " ".join(step.action.value for step in episode.trace),
                }
            )


def _write_summary(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "n_tasks",
        "pass_count",
        "pass_rate",
        "rvr_count",
        "rvr_rate",
        "avg_total_cost",
        "avg_heat_action_cost",
        "avg_inspect_action_cost",
        "avg_assemble_action_cost",
        "avg_overheat_action_cost",
        "avg_abort_action_cost",
        "avg_n_heat",
        "avg_n_inspect",
        "avg_n_assemble",
        "avg_final_temp_c",
    ]
    by_method = {}
    order = []
    for row in rows:
        method = str(row.method_result.method)
        if method not in by_method:
            order.append(method)
        by_method.setdefault(method, []).append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in order:
            xs = list(by_method[str(method)])
            n = len(xs)
            costs = [method_cost_components(x.method_result.episode, x.demo) for x in xs]
            writer.writerow(
                {
                    "method": str(method),
                    "n_tasks": int(n),
                    "pass_count": int(sum(1 for x in xs if x.method_result.episode.success)),
                    "pass_rate": sum(1.0 if x.method_result.episode.success else 0.0 for x in xs) / max(1, n),
                    "rvr_count": int(sum(int(x.method_result.episode.rvr) for x in xs)),
                    "rvr_rate": sum(float(x.method_result.episode.rvr) for x in xs) / max(1, n),
                    "avg_total_cost": sum(float(x.method_result.total_cost) for x in xs) / max(1, n),
                    "avg_heat_action_cost": sum(float(c["heat_action_cost"]) for c in costs) / max(1, n),
                    "avg_inspect_action_cost": sum(float(c["inspect_action_cost"]) for c in costs) / max(1, n),
                    "avg_assemble_action_cost": sum(float(c["assemble_action_cost"]) for c in costs) / max(1, n),
                    "avg_overheat_action_cost": sum(float(c["overheat_action_cost"]) for c in costs) / max(1, n),
                    "avg_abort_action_cost": sum(float(c["abort_action_cost"]) for c in costs) / max(1, n),
                    "avg_n_heat": sum(float(x.method_result.n_heat) for x in xs) / max(1, n),
                    "avg_n_inspect": sum(float(x.method_result.n_inspect) for x in xs) / max(1, n),
                    "avg_n_assemble": sum(float(x.method_result.n_assemble) for x in xs) / max(1, n),
                    "avg_final_temp_c": sum(float(x.method_result.final_temp_c) for x in xs) / max(1, n),
                }
            )


def _fixed_k_from_method(method: str) -> str:
    if str(method) == "llm+reflow_no_pomcp":
        return "40"
    if str(method).startswith("fixed+"):
        return str(method).split("_", 1)[0].replace("fixed+", "")
    return ""


def _tail_risk_before_assemble(episode) -> float:
    for step in reversed(tuple(episode.trace)):
        if step.action.value == "ASSEMBLE":
            return float(step.p_violate)
    return float("nan")


def _write_inspect_events(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "method",
        "step",
        "material",
        "current_temp_c",
        "current_heat_c",
        "true_delta_needed_c",
        "true_residual_needed_c",
        "margin_c",
        "obs",
        "belief_entropy_before",
        "belief_entropy_after",
        "inspect_model_type",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            method = row.method_result.method
            for event in row.method_result.episode.inspect_events:
                writer.writerow(
                    {
                        "task_id": row.task_id,
                        "method": method,
                        "step": int(event.step),
                        "material": event.material,
                        "current_temp_c": int(event.current_temp_c),
                        "current_heat_c": int(event.current_heat_c),
                        "true_delta_needed_c": int(event.true_delta_needed_c),
                        "true_residual_needed_c": int(event.true_delta_needed_c),
                        "margin_c": int(event.margin_c),
                        "obs": event.obs.value,
                        "belief_entropy_before": float(event.belief_entropy_before),
                        "belief_entropy_after": float(event.belief_entropy_after),
                        "inspect_model_type": event.inspect_model_type,
                    }
                )


def _write_online_reflow_proposals(path: Path, proposals: Sequence[OnlineReflowProposal]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task_id",
        "proposal_temp_c",
        "raw_proposal_temp_c",
        "proposal_perturb_eps",
        "proposal_perturb_shift_c",
        "raw",
        "n_examples_before",
        "success",
        "final_temp_c",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for proposal in proposals:
            writer.writerow(
                {
                    "method": str(proposal.method),
                    "task_id": str(proposal.task_id),
                    "proposal_temp_c": int(proposal.proposal_temp_c),
                    "raw_proposal_temp_c": int(proposal.raw_proposal_temp_c),
                    "proposal_perturb_eps": float(proposal.proposal_perturb_eps),
                    "proposal_perturb_shift_c": int(proposal.proposal_perturb_shift_c),
                    "raw": str(proposal.raw),
                    "n_examples_before": int(proposal.n_examples_before),
                    "success": int(bool(proposal.success)),
                    "final_temp_c": int(proposal.final_temp_c),
                }
            )


def _load_task_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _configure_llm_cache(cache_path: str) -> None:
    path = str(cache_path).strip()
    if path:
        os.environ["SLEEVE_LLM_CACHE_PATH"] = path
        return
    os.environ.pop("SLEEVE_LLM_CACHE_PATH", None)


def _proposal_temps_from_llm(
    rows,
    *,
    room_temp_c: int,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    reflow_examples,
    label: str = "llm",
    max_workers: int = 1,
    executor_factory=None,
) -> dict[str, int]:
    from run_pomcp_sleeve_demo import get_recommended_temp_from_llm

    task_rows = tuple(rows)
    proposals: dict[str, int] = {}
    n_rows = len(task_rows)

    def fetch(row):
        t_rec, raw = get_recommended_temp_from_llm(
            model=str(model),
            initial_inner_mm=float(row["initial_inner_mm"]),
            target_shaft_mm=float(row["target_shaft_mm"]),
            material=str(row["material"]),
            room_temp_c=int(room_temp_c),
            provider=str(provider),
            base_url=str(base_url) if str(base_url).strip() else None,
            api_key=str(api_key) if str(api_key).strip() else None,
            reflow_examples=list(reflow_examples or []),
        )
        return int(t_rec), str(raw)

    def record(idx: int, row, t_rec: int, raw: str) -> None:
        task_id = str(row.get("task_id", ""))
        proposals[str(task_id)] = int(t_rec)
        print(
            f"[proposal:{label}] {idx + 1:03d}/{n_rows:03d} "
            f"task_id={task_id} material={row.get('material', '')} T0={int(t_rec)} raw={str(raw).strip()!r}",
            flush=True,
        )

    workers = max(1, int(max_workers))
    if workers > 1 and n_rows > 1:
        factory = executor_factory or ThreadPoolExecutor
        with factory(max_workers=min(workers, n_rows)) as executor:
            futures = [executor.submit(fetch, row) for row in task_rows]
            for idx, (row, future) in enumerate(zip(task_rows, futures)):
                t_rec, raw = future.result()
                record(idx, row, int(t_rec), str(raw))
        return proposals

    for idx, row in enumerate(task_rows):
        t_rec, raw = fetch(row)
        record(idx, row, int(t_rec), str(raw))
    return proposals


def _clamp_temp_c(value: int, *, min_temp_c: int = 250, max_temp_c: int = 450) -> int:
    return int(max(int(min_temp_c), min(int(max_temp_c), int(value))))


def _apply_underheat_perturbation(
    temp_c: int,
    *,
    eps: float,
    max_shift_c: int,
    min_temp_c: int = 250,
    max_temp_c: int = 450,
) -> tuple[int, int]:
    eps_clamped = max(0.0, min(1.0, float(eps)))
    nominal_shift_c = int(float(eps_clamped) * float(max_shift_c))
    perturbed = _clamp_temp_c(
        int(temp_c) - int(nominal_shift_c),
        min_temp_c=int(min_temp_c),
        max_temp_c=int(max_temp_c),
    )
    return int(perturbed), int(temp_c) - int(perturbed)


def _reflow_example(
    row: Mapping[str, Any],
    *,
    room_temp_c: int,
    final_temp_c: int,
    success: bool,
) -> dict[str, Any]:
    return {
        "task_id": str(row.get("task_id", "")),
        "size_group_target_mm": float(row.get("size_group_target_mm", row.get("target_shaft_mm", 0.0))),
        "initial_inner_mm": float(row["initial_inner_mm"]),
        "target_shaft_mm": float(row["target_shaft_mm"]),
        "material": str(row["material"]),
        "room_temp_c": int(room_temp_c),
        "final_temp_c": int(final_temp_c),
        "success": bool(success),
    }


def _select_recent_same_material_reflow_examples(
    row: Mapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    *,
    max_examples: int,
) -> tuple[Mapping[str, Any], ...]:
    material = str(row.get("material", ""))
    same_material = [ex for ex in examples if str(ex.get("material", "")) == material]
    return tuple(same_material[-int(max_examples):])


def run_online_reflow_method(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    llm_func: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], str], tuple[int, str]],
    room_temp_c: int,
    assembly_cooling_c: int,
    max_reflow_examples: int = 20,
    proposal_shift_c: int = 0,
    pomcp_max_depth: int = 20,
    pomcp_max_it: int = 200,
    pomcp_particles: int = 200,
    risk_threshold: float = 0.3,
    belief_scope: str = "broad",
    belief_window_c: int = 15,
    use_world_seed: bool = False,
    proposal_perturb_eps: float = 0.0,
    proposal_perturb_max_shift_c: int = 50,
    progress_label: str = "",
) -> tuple[tuple[MarginTaskRun, ...], tuple[OnlineReflowProposal, ...]]:
    if str(method) not in set(REFLOW_PROMPT_METHODS):
        raise ValueError(f"online reflow method must be one of {REFLOW_PROMPT_METHODS}, got {method!r}")
    max_examples = int(max_reflow_examples)
    if max_examples <= 0:
        raise ValueError("max_reflow_examples must be positive")

    examples: list[Mapping[str, Any]] = []
    runs: list[MarginTaskRun] = []
    proposals: list[OnlineReflowProposal] = []
    n_tasks = len(task_rows)
    for idx, row in enumerate(task_rows, 1):
        task_id = str(row.get("task_id", ""))
        current_examples = _select_recent_same_material_reflow_examples(
            row,
            examples,
            max_examples=max_examples,
        )
        proposal_temp_c, raw = llm_func(row, current_examples, str(method))
        proposal_temp_c = _clamp_temp_c(int(proposal_temp_c) + int(proposal_shift_c))
        raw_proposal_temp_c = int(proposal_temp_c)
        proposal_temp_c, perturb_shift_c = _apply_underheat_perturbation(
            int(proposal_temp_c),
            eps=float(proposal_perturb_eps),
            max_shift_c=int(proposal_perturb_max_shift_c),
        )
        demo = build_demo_from_task_row(
            row,
            proposal_temp_c=int(proposal_temp_c),
            assembly_cooling_c=int(assembly_cooling_c),
            risk_threshold=float(risk_threshold),
            belief_scope=str(belief_scope),
            belief_window_c=int(belief_window_c),
        )
        result = run_method(
            str(method),
            demo,
            prompt_history=True,
            pomcp_max_depth=int(pomcp_max_depth),
            pomcp_max_it=int(pomcp_max_it),
            pomcp_particles=int(pomcp_particles),
            random_seed=(
                int(float(row["world_seed"]))
                if bool(use_world_seed) and str(row.get("world_seed", "")).strip()
                else None
            ),
        )
        examples.append(
            _reflow_example(
                row,
                room_temp_c=int(room_temp_c),
                final_temp_c=int(result.final_temp_c),
                success=bool(result.episode.success),
            )
        )
        proposals.append(
            OnlineReflowProposal(
                method=str(method),
                task_id=str(task_id),
                proposal_temp_c=int(proposal_temp_c),
                raw=str(raw),
                n_examples_before=len(current_examples),
                success=bool(result.episode.success),
                final_temp_c=int(result.final_temp_c),
                raw_proposal_temp_c=int(raw_proposal_temp_c),
                proposal_perturb_eps=float(proposal_perturb_eps),
                proposal_perturb_shift_c=int(perturb_shift_c),
            )
        )
        runs.append(
            MarginTaskRun(
                task_id=str(task_id),
                material=str(row.get("material", "")),
                t_needed_c=float(row["t_needed_c"]),
                proposal_temp_c=int(proposal_temp_c),
                demo=demo,
                method_result=result,
            )
        )
        if str(progress_label).strip():
            print(
                f"[{progress_label}] {idx:03d}/{n_tasks:03d} "
                f"task_id={task_id} examples_before={len(current_examples)} "
                f"T0={int(proposal_temp_c)} success={int(result.episode.success)}",
                flush=True,
            )
    return tuple(runs), tuple(proposals)


def run_online_reflow_methods(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    llm_func: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]], str], tuple[int, str]],
    room_temp_c: int,
    assembly_cooling_c: int,
    max_reflow_examples: int = 20,
    proposal_shift_c: int = 0,
    risk_threshold: float = 0.3,
    belief_scope: str = "broad",
    belief_window_c: int = 15,
    use_world_seed: bool = False,
    proposal_perturb_eps: float = 0.0,
    proposal_perturb_max_shift_c: int = 50,
) -> tuple[tuple[MarginTaskRun, ...], tuple[OnlineReflowProposal, ...]]:
    all_runs: list[MarginTaskRun] = []
    all_proposals: list[OnlineReflowProposal] = []
    for method in methods:
        runs, proposals = run_online_reflow_method(
            task_rows,
            method=str(method),
            llm_func=llm_func,
            room_temp_c=int(room_temp_c),
            assembly_cooling_c=int(assembly_cooling_c),
            max_reflow_examples=int(max_reflow_examples),
            proposal_shift_c=int(proposal_shift_c),
            risk_threshold=float(risk_threshold),
            belief_scope=str(belief_scope),
            belief_window_c=int(belief_window_c),
            use_world_seed=bool(use_world_seed),
            proposal_perturb_eps=float(proposal_perturb_eps),
            proposal_perturb_max_shift_c=int(proposal_perturb_max_shift_c),
            progress_label=str(method),
        )
        all_runs.extend(runs)
        all_proposals.extend(proposals)
    return tuple(all_runs), tuple(all_proposals)


def _proposal_temps_from_column(rows, column: str) -> dict[str, int]:
    proposals: dict[str, int] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        proposals[str(task_id)] = int(float(row[str(column)]))
    return proposals


def _run_openai_non_reflow_branch(args: Mapping[str, Any]):
    methods = tuple(str(method) for method in args["methods"])
    if not methods:
        return ()
    proposal_temps = _proposal_temps_from_llm(
        tuple(args["task_rows"]),
        room_temp_c=int(args["room_temp_c"]),
        provider=str(args["llm_provider"]),
        base_url=str(args["llm_base_url"]),
        api_key=str(args["llm_api_key"]),
        model=str(args["openai_model"]),
        reflow_examples=[],
        label="llm:no_reflow",
        max_workers=int(args["llm_workers"]),
    )
    if int(args["proposal_shift_c"]) != 0:
        proposal_temps = apply_proposal_shift(
            proposal_temps,
            shift_c=int(args["proposal_shift_c"]),
            min_temp_c=250,
            max_temp_c=450,
        )
    return run_margin_tasks_csv(
        Path(str(args["tasks_path"])),
        methods=methods,
        proposal_temp_c=int(args["proposal_temp_c"]) + int(args["proposal_shift_c"]),
        proposal_temps_c=proposal_temps,
        assembly_cooling_c=int(args["assembly_cooling_c"]),
        room_temp_c=int(args["room_temp_c"]),
        risk_threshold=float(args["risk_threshold"]),
        belief_scope=str(args["belief_scope"]),
        belief_window_c=int(args["belief_window_c"]),
        use_world_seed=bool(args["use_world_seed"]),
    )


def _run_openai_reflow_branch(args: Mapping[str, Any]):
    methods = tuple(str(method) for method in args["methods"])
    if not methods:
        return (), ()

    from run_pomcp_sleeve_demo import get_recommended_temp_from_llm

    def online_llm(row, reflow_examples, method):
        return get_recommended_temp_from_llm(
            model=str(args["openai_model"]),
            initial_inner_mm=float(row["initial_inner_mm"]),
            target_shaft_mm=float(row["target_shaft_mm"]),
            material=str(row["material"]),
            room_temp_c=int(args["room_temp_c"]),
            provider=str(args["llm_provider"]),
            base_url=str(args["llm_base_url"]) if str(args["llm_base_url"]).strip() else None,
            api_key=str(args["llm_api_key"]) if str(args["llm_api_key"]).strip() else None,
            reflow_examples=list(reflow_examples or []),
        )

    return run_online_reflow_methods(
        tuple(args["task_rows"]),
        methods=methods,
        llm_func=online_llm,
        room_temp_c=int(args["room_temp_c"]),
        assembly_cooling_c=int(args["assembly_cooling_c"]),
        max_reflow_examples=20,
        proposal_shift_c=int(args["proposal_shift_c"]),
        risk_threshold=float(args["risk_threshold"]),
        belief_scope=str(args["belief_scope"]),
        belief_window_c=int(args["belief_window_c"]),
        use_world_seed=bool(args["use_world_seed"]),
        proposal_perturb_eps=float(args["proposal_perturb_eps"]),
        proposal_perturb_max_shift_c=int(args["proposal_perturb_max_shift_c"]),
    )


def _run_openai_reflow_comparison_branches(
    *,
    tasks_path: Path,
    task_rows: Sequence[Mapping[str, Any]],
    non_reflow_methods: Sequence[str],
    reflow_methods: Sequence[str],
    proposal_temp_c: int,
    room_temp_c: int,
    assembly_cooling_c: int,
    proposal_shift_c: int,
    risk_threshold: float,
    belief_scope: str = "broad",
    belief_window_c: int = 15,
    use_world_seed: bool = False,
    llm_provider: str,
    llm_base_url: str,
    llm_api_key: str,
    openai_model: str,
    llm_workers: int = 8,
    proposal_perturb_eps: float = 0.0,
    proposal_perturb_max_shift_c: int = 50,
    executor_factory=None,
):
    common_args = {
        "tasks_path": str(tasks_path),
        "task_rows": tuple(dict(row) for row in task_rows),
        "proposal_temp_c": int(proposal_temp_c),
        "room_temp_c": int(room_temp_c),
        "assembly_cooling_c": int(assembly_cooling_c),
        "proposal_shift_c": int(proposal_shift_c),
        "risk_threshold": float(risk_threshold),
        "belief_scope": str(belief_scope),
        "belief_window_c": int(belief_window_c),
        "use_world_seed": bool(use_world_seed),
        "proposal_perturb_eps": float(proposal_perturb_eps),
        "proposal_perturb_max_shift_c": int(proposal_perturb_max_shift_c),
        "llm_provider": str(llm_provider),
        "llm_base_url": str(llm_base_url),
        "llm_api_key": str(llm_api_key),
        "openai_model": str(openai_model),
        "llm_workers": int(llm_workers),
    }
    non_reflow_args = {
        **common_args,
        "branch_name": "non_reflow",
        "methods": tuple(str(method) for method in non_reflow_methods),
    }
    reflow_args = {
        **common_args,
        "branch_name": "reflow",
        "methods": tuple(str(method) for method in reflow_methods),
    }

    if non_reflow_methods and reflow_methods:
        factory = executor_factory or ProcessPoolExecutor
        with factory(max_workers=2) as executor:
            non_reflow_future = executor.submit(_run_openai_non_reflow_branch, non_reflow_args)
            reflow_future = executor.submit(_run_openai_reflow_branch, reflow_args)
            non_reflow_rows = tuple(non_reflow_future.result())
            reflow_rows, online_reflow_proposals = reflow_future.result()
        return tuple(non_reflow_rows) + tuple(reflow_rows), tuple(online_reflow_proposals)

    if non_reflow_methods:
        return tuple(_run_openai_non_reflow_branch(non_reflow_args)), ()
    if reflow_methods:
        reflow_rows, online_reflow_proposals = _run_openai_reflow_branch(reflow_args)
        return tuple(reflow_rows), tuple(online_reflow_proposals)
    return (), ()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the new margin-bin POMCP PVEP model on a sleeve tasks CSV."
    )
    parser.add_argument(
        "--tasks-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "sleeve_tasks_head10x3.csv"),
    )
    parser.add_argument("--proposal-temp-c", type=int, default=300)
    parser.add_argument(
        "--proposal-shift-c",
        type=int,
        default=0,
        help="Apply this shift to every task proposal after loading/generating it; use -30/-50 for hard sets.",
    )
    parser.add_argument(
        "--proposal-source",
        type=str,
        choices=["fixed", "openai", "column"],
        default="openai",
        help="How to obtain per-task proposal_temp_c. openai calls the LLM prompt.",
    )
    parser.add_argument("--proposal-column", type=str, default="proposal_temp_c")
    parser.add_argument("--room-temp-c", type=int, default=16)
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["openai", "anthropic", "anthropic_compatible", "bigmodel_anthropic"],
        default=os.environ.get("SLEEVE_LLM_PROVIDER", "anthropic"),
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get(
            "SLEEVE_LLM_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/anthropic"),
        ),
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=os.environ.get("SLEEVE_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", os.environ.get("BIGMODEL_API_KEY", ""))),
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=os.environ.get("SLEEVE_LLM_MODEL", os.environ.get("OPENAI_MODEL", "glm-4.7")),
    )
    parser.add_argument(
        "--llm-cache-path",
        type=str,
        default="",
        help="Optional LLM cache path. Empty by default so each run calls the LLM freshly.",
    )
    parser.add_argument(
        "--llm-workers",
        type=int,
        default=8,
        help="Parallel workers for independent no-reflow LLM proposal requests.",
    )
    parser.add_argument("--assembly-cooling-c", type=int, default=10)
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.3,
        help="Allow ASSEMBLE when predicted violation probability is <= this threshold; use 1.0 to disable the gate.",
    )
    parser.add_argument(
        "--belief-scope",
        type=str,
        choices=list(BELIEF_SCOPES),
        default="broad",
        help="Initial belief support: global 250--440 C or proposal-centered narrow window.",
    )
    parser.add_argument(
        "--belief-window-c",
        type=int,
        default=15,
        help="Half-width of the proposal-centered narrow belief support in C.",
    )
    parser.add_argument(
        "--use-world-seed",
        action="store_true",
        help="Use each task's world_seed for POMCP and observation randomness.",
    )
    parser.add_argument(
        "--proposal-perturb-eps",
        type=float,
        default=0.0,
        help="Underheat perturbation ratio applied to reflow LLM proposals; shift=floor(eps*max_shift_c).",
    )
    parser.add_argument(
        "--proposal-perturb-max-shift-c",
        type=int,
        default=50,
        help="Maximum underheat perturbation in C for reflow LLM proposals.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_TASK_METHODS),
    )
    parser.add_argument(
        "--details-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "margin_pvep_head10x3_details.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="",
        help="Optional per-method aggregate CSV with pass/RVR and action-cost breakdowns.",
    )
    parser.add_argument(
        "--inspect-events-csv",
        type=str,
        default="",
        help="Optional per-INSPECT event log CSV path.",
    )
    parser.add_argument(
        "--online-reflow-proposals-csv",
        type=str,
        default="",
        help="Optional CSV for online reflow LLM proposals and rolling-example counts.",
    )
    args = parser.parse_args()

    tasks_path = Path(str(args.tasks_csv))
    task_rows = _load_task_rows(tasks_path)
    methods = _parse_methods(str(args.methods))
    needs_reflow_proposals = bool(set(methods).intersection(set(REFLOW_PROMPT_METHODS)))
    reflow_methods = tuple(str(method) for method in methods if str(method) in set(REFLOW_PROMPT_METHODS))
    non_reflow_methods = tuple(str(method) for method in methods if str(method) not in set(REFLOW_PROMPT_METHODS))
    proposal_temps = None
    reflow_proposal_temps = None
    online_reflow_proposals: tuple[OnlineReflowProposal, ...] = ()
    rows = None
    if str(args.proposal_source) == "openai":
        _configure_llm_cache(str(args.llm_cache_path))
        if bool(needs_reflow_proposals):
            rows, online_reflow_proposals = _run_openai_reflow_comparison_branches(
                tasks_path=tasks_path,
                task_rows=task_rows,
                non_reflow_methods=non_reflow_methods,
                reflow_methods=reflow_methods,
                proposal_temp_c=int(args.proposal_temp_c),
                room_temp_c=int(args.room_temp_c),
                assembly_cooling_c=int(args.assembly_cooling_c),
                proposal_shift_c=int(args.proposal_shift_c),
                risk_threshold=float(args.risk_threshold),
                belief_scope=str(args.belief_scope),
                belief_window_c=int(args.belief_window_c),
                use_world_seed=bool(args.use_world_seed),
                proposal_perturb_eps=float(args.proposal_perturb_eps),
                proposal_perturb_max_shift_c=int(args.proposal_perturb_max_shift_c),
                llm_provider=str(args.llm_provider),
                llm_base_url=str(args.llm_base_url),
                llm_api_key=str(args.llm_api_key),
                openai_model=str(args.openai_model),
                llm_workers=int(args.llm_workers),
            )
        elif non_reflow_methods:
            proposal_temps = _proposal_temps_from_llm(
                task_rows,
                room_temp_c=int(args.room_temp_c),
                provider=str(args.llm_provider),
                base_url=str(args.llm_base_url),
                api_key=str(args.llm_api_key),
                model=str(args.openai_model),
                reflow_examples=[],
                label="llm:no_reflow",
                max_workers=int(args.llm_workers),
            )
    elif str(args.proposal_source) == "column":
        proposal_temps = _proposal_temps_from_column(task_rows, str(args.proposal_column))
        if bool(needs_reflow_proposals):
            reflow_proposal_temps = proposal_temps
    if proposal_temps is not None and int(args.proposal_shift_c) != 0:
        proposal_temps = apply_proposal_shift(
            proposal_temps,
            shift_c=int(args.proposal_shift_c),
            min_temp_c=250,
            max_temp_c=450,
        )
    if reflow_proposal_temps is not None and int(args.proposal_shift_c) != 0:
        reflow_proposal_temps = apply_proposal_shift(
            reflow_proposal_temps,
            shift_c=int(args.proposal_shift_c),
            min_temp_c=250,
            max_temp_c=450,
        )
    fixed_proposal_temp_c = int(args.proposal_temp_c) + int(args.proposal_shift_c)

    if rows is None:
        rows = run_margin_tasks_csv(
            tasks_path,
            methods=methods,
            proposal_temp_c=int(fixed_proposal_temp_c),
            proposal_temps_c=proposal_temps,
            reflow_proposal_temps_c=reflow_proposal_temps,
            assembly_cooling_c=int(args.assembly_cooling_c),
            room_temp_c=int(args.room_temp_c),
            risk_threshold=float(args.risk_threshold),
            belief_scope=str(args.belief_scope),
            belief_window_c=int(args.belief_window_c),
            use_world_seed=bool(args.use_world_seed),
        )
    aggregates = aggregate_task_results(rows)

    print(format_task_aggregate(aggregates))
    print()
    print(f"details_csv={str(Path(args.details_csv))}")
    _write_details(
        Path(str(args.details_csv)),
        rows,
    )
    if str(args.summary_csv).strip():
        _write_summary(Path(str(args.summary_csv)), rows)
        print(f"summary_csv={str(Path(args.summary_csv))}")
    if str(args.inspect_events_csv).strip():
        _write_inspect_events(Path(str(args.inspect_events_csv)), rows)
        print(f"inspect_events_csv={str(Path(args.inspect_events_csv))}")
    if str(args.online_reflow_proposals_csv).strip() and online_reflow_proposals:
        _write_online_reflow_proposals(Path(str(args.online_reflow_proposals_csv)), online_reflow_proposals)
        print(f"online_reflow_proposals_csv={str(Path(args.online_reflow_proposals_csv))}")


if __name__ == "__main__":
    main()

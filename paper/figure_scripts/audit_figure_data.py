from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from data_utils import bootstrap_mean_interval, file_sha256, first_existing_path, read_csv, wilson_interval
from gen_fig6_scope_intervention import load_counts as load_scope_counts
from paper_plot_style import RESULTS_DIR, SCRIPT_DIR


SOURCES = {
    "ariac_sweep": RESULTS_DIR / "ariac" / "Vbinary_sweep_per_order.csv",
    "ariac_admissibility": RESULTS_DIR / "ariac" / "ariac_pertrial_admissibility.csv",
    "ariac_mechanism": first_existing_path(
        RESULTS_DIR / "ariac" / "实验整理_更新版_mix更正.csv",
        RESULTS_DIR / "ariac" / "nominal_ablation_per_scenario.csv",
    ),
    "ariac_vocabulary": first_existing_path(
        RESULTS_DIR / "ariac" / "欠定任务40_variants结果.csv",
        RESULTS_DIR / "ariac" / "vocabulary_coverage_variants.csv",
    ),
    "ariac_scaling": RESULTS_DIR / "ariac" / "flat_scaling_q200_q2000_raw.csv",
    "tv_results": RESULTS_DIR / "tv" / "final10.csv",
    "tv_summary": RESULTS_DIR / "tv" / "final10_summary.csv",
    "rotor_methods": RESULTS_DIR / "rotor" / "table1_main_5_methods.csv",
    "rotor_corruption": RESULTS_DIR / "rotor" / "table2_perturbation_sweep.csv",
    "rotor_scope": RESULTS_DIR / "rotor" / "rotor_scope2x2.csv",
}


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", double_precision=6))


def _parse_ariac_score(value: object) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == "-":
        return 0.0
    return float(sum(float(term.strip()) for term in text.split("+")))


def _mcnemar_exact(mild: np.ndarray, severe: np.ndarray) -> dict[str, float | int]:
    mild_only = int(np.sum((mild == 1) & (severe == 0)))
    severe_only = int(np.sum((mild == 0) & (severe == 1)))
    discordant = mild_only + severe_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(mild_only, severe_only), discordant, 0.5).pvalue)
    return {"mild_only": mild_only, "severe_only": severe_only, "p_value": p_value}


def build_report() -> dict[str, object]:
    sweep_all = read_csv(SOURCES["ariac_sweep"])
    nominal_max = sweep_all[
        (sweep_all["config"] == "V_full") & np.isclose(sweep_all["epsilon"], 0.0)
    ][["order_id", "max_score"]].rename(columns={"order_id": "trial_name"})
    if len(nominal_max) != 50 or nominal_max["trial_name"].nunique() != 50:
        raise ValueError("Expected one nominal maximum score for each of 50 ARIAC scenarios")
    sweep = sweep_all[sweep_all["epsilon"].isin([0.25, 0.50, 0.75, 1.00])].copy()
    sweep["score_percent"] = 100.0 * sweep["score"] / sweep["max_score"]
    sweep["full_score"] = np.isclose(sweep["score"], sweep["max_score"]).astype(int)
    sweep["any_score"] = (sweep["score"] > 0).astype(int)
    ariac_summary = (
        sweep.groupby(["config", "epsilon"])
        .agg(
            n=("order_id", "size"),
            full_score_completion=("full_score", "mean"),
            mean_score_percent=("score_percent", "mean"),
            any_score_completion=("any_score", "mean"),
            mean_wall_time_s=("wall_time_s", "mean"),
        )
        .reset_index()
    )

    quality = read_csv(SOURCES["ariac_admissibility"])
    quality_summary = (
        quality.groupby(["config", "eps"])
        .agg(n=("order_id", "size"), mean_admissibility=("admissibility", "mean"), mean_inspections=("inspect_sum", "mean"))
        .reset_index()
    )

    full = sweep[sweep["config"] == "V_full"]
    full_completion = full.pivot(index="order_id", columns="epsilon", values="full_score")
    full_score = full.pivot(index="order_id", columns="epsilon", values="score_percent")
    completion_difference = (full_completion[1.0] - full_completion[0.25]).to_numpy(float)
    score_difference = (full_score[1.0] - full_score[0.25]).to_numpy(float)
    ariac_completion_ci90 = bootstrap_mean_interval(
        completion_difference, confidence=0.90, n_boot=100_000, seed=20260803
    )
    ariac_score_ci90 = bootstrap_mean_interval(
        score_difference, confidence=0.90, n_boot=100_000, seed=20260804
    )

    tv = read_csv(SOURCES["tv_results"])
    tv_rows: list[dict[str, float | int]] = []
    tv_methods = {
        0.00: "PVEP",
        0.25: "PVEP_eps_0.25",
        0.50: "PVEP_eps_0.50",
        0.75: "PVEP_eps_0.75",
        1.00: "PVEP_eps_1.00",
    }
    for eps, method in tv_methods.items():
        cell = tv[tv["method"] == method]
        unsafe_episode_count = int((1 - cell["task_pass"].astype(int)).sum())
        tv_rows.append(
            {
                "epsilon": eps,
                "n": len(cell),
                "pass_rate": float(cell["task_pass"].mean()),
                "unsafe_episode_count": unsafe_episode_count,
                "unsafe_episode_rate": float(unsafe_episode_count / len(cell)),
                "mean_total_cost": float(cell["total_cost"].mean()),
                "total_cost_population_std": float(cell["total_cost"].std(ddof=0)),
                "fasten_violations": int(cell["fasten_violation_count"].sum()),
                "mean_probes": float(cell["probe_count"].mean()),
                "mean_unsafe_fasten": float(cell["fasten_violation_count"].mean()),
            }
        )
    tv_mild = tv[tv["method"] == "PVEP_eps_0.25"].sort_values("group")["task_pass"].to_numpy(int)
    tv_severe = tv[tv["method"] == "PVEP_eps_1.00"].sort_values("group")["task_pass"].to_numpy(int)
    tv_difference = tv_severe - tv_mild
    tv_ci90 = bootstrap_mean_interval(
        tv_difference, confidence=0.90, n_boot=100_000, seed=20260805
    )

    rotor = read_csv(SOURCES["rotor_corruption"])
    rotor_rows: list[dict[str, float | int]] = []
    for eps, tag in zip((0.25, 0.50, 0.75, 1.00), ("0p25", "0p50", "0p75", "1p00")):
        prefix = f"eps_{tag}_"
        margin = rotor[prefix + "proposal_temp_c"] - rotor["t_needed_c"]
        final_margin = rotor[prefix + "final_temp_c"] - rotor["t_needed_c"]
        rotor_rows.append(
            {
                "epsilon": eps,
                "n": len(rotor),
                "risk_violations": int(rotor[prefix + "rvr"].sum()),
                "mean_margin_c": float(margin.mean()),
                "min_margin_c": float(margin.min()),
                "max_margin_c": float(margin.max()),
                "mean_final_margin_c": float(final_margin.mean()),
                "mean_process_cost": float(rotor[prefix + "total_cost"].mean()),
                "mean_heating_actions": float(rotor[prefix + "n_heat"].mean()),
                "mean_inspections": float(rotor[prefix + "n_inspect"].mean()),
            }
        )

    mechanism_all = read_csv(SOURCES["ariac_mechanism"]).merge(
        nominal_max, on="trial_name", how="left", validate="one_to_one"
    )
    if mechanism_all["max_score"].isna().any():
        raise ValueError("Missing a maximum score in the ARIAC method comparison")
    mechanism = mechanism_all.copy()
    mechanism["scope"] = mechanism["regime"].map(
        {
            "normal": "routine",
            "priority": "routine",
            "dropped_part": "challenge",
            "faulty_part": "challenge",
            "mix_challenges": "mixed",
        }
    )
    mechanism = mechanism[mechanism["scope"].isin(["routine", "challenge"])]
    mechanism_rows: list[dict[str, object]] = []
    for scope, cell in mechanism.groupby("scope"):
        row: dict[str, object] = {"scope": scope, "n": len(cell)}
        for column in ("pomdp_our_completion", "our_error_completion", "our_raw_completion"):
            row[column.replace("_completion", "_strict_count")] = int((cell[column] == 1).sum())
        mechanism_rows.append(row)

    coverage = read_csv(SOURCES["ariac_vocabulary"])
    coverage_summary = (
        coverage.groupby("condition")
        .agg(n=("trial_name", "size"), strict_symbolic=("strict_symbolic_applicable", "sum"), vlm=("grounding_exact_match", "sum"))
        .reset_index()
    )

    scaling = read_csv(SOURCES["ariac_scaling"])
    scaling = scaling[
        (scaling["horizon"] == 8 * scaling["task_size"] + 4)
        & (scaling["latent_variables"] == scaling["task_size"])
    ].drop_duplicates(
        ["tree_queries", "task_size", "horizon", "latent_variables", "method", "seed"]
    )
    scaling_counts = scaling.groupby(["tree_queries", "task_size", "method"]).size()
    if not (scaling_counts == 20).all():
        raise ValueError(f"Canonical scaling slice must contain 20 unique seeds per cell:\n{scaling_counts}")
    scaling_summary = (
        scaling.groupby(["tree_queries", "task_size", "method"])
        .agg(n=("seed", "size"), candidate_actions=("candidate_actions_mean", "mean"), success_rate=("success", "mean"))
        .reset_index()
    )

    ariac_primary_columns = {
        "FM": "open_loop_vlm_nl_score",
        "FM + Repair": "vlm_nl_re_score",
        "PVEP w/o POMDP": "vlm_pddl_re_score",
        "PVEP w/o SG": "our_error_score",
        "PVEP": "pomdp_our_score",
    }
    ariac_primary = []
    for index, (method, column) in enumerate(ariac_primary_columns.items()):
        values = 100.0 * mechanism_all[column].map(_parse_ariac_score) / mechanism_all["max_score"]
        low, high = bootstrap_mean_interval(values.to_numpy(float), seed=20260840 + index)
        ariac_primary.append(
            {
                "method": method,
                "released_score_column": column,
                "n": len(values),
                "mean_score_percent": float(values.mean()),
                "bootstrap_ci95_percent": [float(low), float(high)],
            }
        )

    regime_names = {
        "normal": "Normal",
        "priority": "Priority",
        "dropped_part": "Dropped part",
        "faulty_part": "Faulty part",
        "mix_challenges": "Mixed",
    }
    ariac_condition_scores: list[dict[str, object]] = []
    for regime, display_name in regime_names.items():
        cell = mechanism_all[mechanism_all["regime"] == regime]
        if len(cell) != 10:
            raise ValueError(f"ARIAC condition {regime} must contain 10 scenarios")
        for method, column in ariac_primary_columns.items():
            values = 100.0 * cell[column].map(_parse_ariac_score) / cell["max_score"]
            ariac_condition_scores.append(
                {
                    "condition": display_name,
                    "method": method,
                    "n": len(values),
                    "mean_score_percent": float(values.mean()),
                }
            )

    tv_primary = []
    for method in ("MLM", "Human", "PVEP_no_POMDP", "PVEP_no_SG", "PVEP"):
        cell = tv[tv["method"] == method]
        tv_primary.append(
            {
                "method": method,
                "n": len(cell),
                "safety_passes": int(cell["task_pass"].sum()),
                "unsafe_episodes": int((1 - cell["task_pass"].astype(int)).sum()),
                "mean_episode_cost": float(cell["total_cost"].mean()),
            }
        )

    rotor_methods = read_csv(SOURCES["rotor_methods"])
    rotor_primary = []
    for prefix in ("human", "llm", "llm_reflow", "pomdp_no_reflow", "pomdp_reflow"):
        rotor_primary.append(
            {
                "method": prefix,
                "n": len(rotor_methods),
                "passes": int(rotor_methods[prefix + "_pass"].sum()),
                "risk_violations": int(rotor_methods[prefix + "_rvr"].sum()),
                "mean_process_cost": float(rotor_methods[prefix + "_total_cost"].mean()),
            }
        )

    scope_counts = load_scope_counts()
    scope_rows = [
        {"scope": scope, "configuration": config, "violations": count, "n": trials}
        for (scope, config), (count, trials) in sorted(scope_counts.items())
    ]

    return {
        "policy": "All plotted values are derived from the curated results directory; publication assets supply panel-a imagery only.",
        "sources": {
            name: {"path": str(path.relative_to(RESULTS_DIR.parent)), "sha256": file_sha256(path)}
            for name, path in SOURCES.items()
        },
        "fig2_ariac": {
            "method_score_comparison": ariac_primary,
            "task_condition_scores": ariac_condition_scores,
            "corruption_sweep": _records(ariac_summary),
            "ariac_admissibility": _records(quality_summary),
            "mechanism": mechanism_rows,
        },
        "fig3_tv": {
            "corruption_sweep": tv_rows,
            "method_comparison": tv_primary,
        },
        "fig4_rotor": {
            "corruption_sweep": rotor_rows,
            "method_comparison": rotor_primary,
        },
        "fig5_coverage_scaling": {
            "coverage": _records(coverage_summary),
            "scaling": _records(scaling_summary),
        },
        "supplementary_scope_intervention": scope_rows,
        "inferential_checks": {
            "ariac_full_severe_minus_mild": {
                "full_score_completion_difference_pp": float(100.0 * completion_difference.mean()),
                "completion_bootstrap_ci90_pp": [float(100.0 * x) for x in ariac_completion_ci90],
                "completion_mcnemar": _mcnemar_exact(
                    full_completion[0.25].to_numpy(int), full_completion[1.0].to_numpy(int)
                ),
                "mean_score_difference_pp": float(score_difference.mean()),
                "score_bootstrap_ci90_pp": list(ariac_score_ci90),
            },
            "tv_severe_minus_mild": {
                "pass_rate_difference_pp": float(100.0 * tv_difference.mean()),
                "bootstrap_ci90_pp": [float(100.0 * x) for x in tv_ci90],
                "mcnemar": _mcnemar_exact(tv_mild, tv_severe),
            },
            "rotor_zero_event_summary": {
                "per_condition_n": 90,
                "per_condition_wilson_upper_percent": float(100.0 * wilson_interval(0, 90)[1]),
                "descriptive_pooled_events": 0,
                "descriptive_pooled_trials": 360,
                "note": "The same 90 trial identities repeat across four conditions; no pooled Bernoulli interval is computed.",
            },
        },
    }


def main() -> None:
    report = build_report()
    output = SCRIPT_DIR / "figure_data_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

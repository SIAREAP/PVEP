import os
from pathlib import Path
from typing import Any

try:
    from juliacall import Main as jl
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "缺少 Python 依赖 `juliacall`。请先安装：\n"
        "  pip install juliacall\n"
        "并确保本机 Julia 可用（`julia --version`）。"
    ) from e


def _as_py_dict(x) -> dict:
    # juliacall often returns already-converted dicts; keep this defensive.
    if isinstance(x, dict):
        return x
    try:
        return dict(x)
    except Exception:  # pragma: no cover
        return {"_": x}


def _ensure_julia_deps() -> None:
    """
    Ensure the Julia packages required by `solve_pomdp.jl` are available.

    We keep this in Python (instead of committing a Julia Project/Manifest) to avoid
    huge `Manifest.toml` churn and to keep setup one-shot for users.
    """
    jl.seval(
        r"""
import Pkg
required = [
  "PDDL",
  "POMDPs",
  "POMDPTools",
  "BasicPOMCP",
  "ParticleFilters",
  "StatsBase",
  "CommonRLInterface",
]

for p in required
    if Base.find_package(p) === nothing
        Pkg.add(p)
    end
end
"""
    )


class JuliaInfoClient:
    """
    Thin bridge for the hybrid controller:
    - Julia handles *only* trigger-time information decisions and belief updates.
    - Python owns the real UP state and classical replanning.
    """

    def __init__(
        self,
        *,
        domain_path: str,
        p_vlm_path: str,
        solve_pomdp_jl: str | None = None,
        p_high: float = 0.8,
        seed: int = 0,
        tree_queries: int = 200,
        max_depth: int = 3,
        c: float = 1.0,
    ):
        if solve_pomdp_jl is None:
            solve_pomdp_jl = os.environ.get(
                "ARIAC_SOLVE_POMDP_JL",
                str(Path(__file__).with_name("solve_pomdp.jl")),
            )
        _ensure_julia_deps()
        # Avoid Julia warnings like:
        #   WARNING: replacing module SymbolicMDPs.
        #   WARNING: replacing module AriacKitting.
        #
        # Those happen when the same `solve_pomdp.jl` is `include`d multiple times
        # within the same Julia session (juliacall keeps a persistent Julia runtime).
        #
        # Default behavior: only include once per process. If you *intentionally*
        # edit the Julia file and want to reload it, set:
        #   ARIAC_RELOAD_JULIA_MODULES=1
        try:
            force_reload = str(os.environ.get("ARIAC_RELOAD_JULIA_MODULES", "0")).lower() in ("1", "true", "yes", "on")
            already_loaded = bool(jl.seval("isdefined(Main, :AriacKitting) && isdefined(Main, :SymbolicMDPs)"))
            if force_reload or (not already_loaded):
                jl.include(str(solve_pomdp_jl))
        except Exception:
            # If the guard fails for any reason, fall back to include (safe, but may warn).
            jl.include(str(solve_pomdp_jl))
        self._m = jl.AriacKitting

        self.domain_path = str(domain_path)
        self.p_vlm_path = str(p_vlm_path)
        self.p_high = float(p_high)

        self.seed = int(seed)
        self.tree_queries = int(tree_queries)
        self.max_depth = int(max_depth)
        self.c = float(c)

    def init_prior(self) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        res = self._m.init_prior(self.domain_path, self.p_vlm_path, p_high=self.p_high)
        prior_loc = _as_py_dict(res.prior_loc)
        prior_good = _as_py_dict(res.prior_good)
        # Ensure float conversion
        prior_loc2: dict[str, dict[str, float]] = {}
        for p, d in prior_loc.items():
            prior_loc2[str(p)] = {str(b): float(v) for b, v in _as_py_dict(d).items()}
        prior_good2 = {str(p): float(v) for p, v in prior_good.items()}
        return prior_loc2, prior_good2

    def solve_pre_pick(
        self,
        *,
        part: str,
        robot: str,
        candidate_bins: list[str],
        loc_probs: list[float],
        inspect_cost: float,
        miss_penalty: float,
    ) -> str:
        a = self._m.solve_trigger(
            "pre_pick",
            robot=str(robot),
            candidate_bins=[str(x) for x in candidate_bins],
            loc_probs=[float(x) for x in loc_probs],
            inspect_cost=float(inspect_cost),
            miss_penalty=float(miss_penalty),
            tree_queries=int(self.tree_queries),
            max_depth=int(self.max_depth),
            c=float(self.c),
        )
        return str(a)

    def solve_quality(
        self,
        *,
        trigger_type: str,
        order: str,
        part: str,
        slot: str,
        agv: str,
        p_good: float,
        qc_cost: float,
        repair_cost: float,
        fail_penalty: float,
    ) -> str:
        a = self._m.solve_trigger(
            str(trigger_type),
            order=str(order),
            part=str(part),
            slot=str(slot),
            agv=str(agv),
            p_good=float(p_good),
            qc_cost=float(qc_cost),
            repair_cost=float(repair_cost),
            fail_penalty=float(fail_penalty),
            tree_queries=int(self.tree_queries),
            max_depth=int(self.max_depth),
            c=float(self.c),
        )
        return str(a)

    def update_loc_target(self, belief_loc: dict[str, dict[str, float]], *, part: str, location: str, saw: bool) -> None:
        # Julia returns a new dict; update in-place on the Python side to keep references stable.
        updated = self._m.update_belief_loc_target(belief_loc, str(part), str(location), bool(saw))
        upd_py = _as_py_dict(updated)
        belief_loc.clear()
        for p, d in upd_py.items():
            belief_loc[str(p)] = {str(b): float(v) for b, v in _as_py_dict(d).items()}

    def update_quality(self, p_good: float, *, obs_label: str) -> float:
        return float(self._m.update_belief_quality(float(p_good), str(obs_label)))



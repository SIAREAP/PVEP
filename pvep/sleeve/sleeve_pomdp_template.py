from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Tuple

from structs import (
    InitialModel,
    Observation,
    ObservationModel,
    RewardModel,
    State,
    TransitionModel,
)


class SleeveActions(IntEnum):
    HEAT = 0
    PLACE = 2
    INSPECT_ID = 3
    ASSEMBLE = 4
    HEAT_TO_TARGET = 5


class SleeveLocationEnum(IntEnum):
    INFEED = 0
    HEATER = 1
    INSPECT = 2
    ASSEMBLY = 3


class SleeveIDStatusEnum(IntEnum):
    IN_SPEC = 0
    TOO_SMALL = 1
    TOO_LARGE = 2
    NONE = 3  # returned when no measurement is produced


@dataclass(frozen=True)
class SleeveObservation(Observation):
    """
    Observation includes:
      - meas: ID判定（仅 INSPECT_ID 动作时为 {IN_SPEC/TOO_SMALL/TOO_LARGE}；其他动作时为 NONE）
      - temp: 当前温度（离散到整数；若你的真实系统是连续，可在 env 内部保留 float，但观测输出 int）
      - loc: 当前位姿/工位（离散枚举）
    """
    meas: int
    temp: int
    loc: int

    def encode(self) -> "SleeveObservation":  # type: ignore[override]
        return copy.deepcopy(self)

    @staticmethod
    def decode(obj: "SleeveObservation") -> "SleeveObservation":
        return copy.deepcopy(obj)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash((int(self.meas), int(self.temp), int(self.loc)))


def obs_from_env(env_obs: Any) -> SleeveObservation:
    """把黑盒环境返回的 observation 转成模板内的 SleeveObservation（不 import environments）。"""
    if isinstance(env_obs, SleeveObservation):
        return SleeveObservation.decode(env_obs)

    # tolerate dict
    if isinstance(env_obs, dict):
        return SleeveObservation(
            meas=int(env_obs.get("meas", SleeveIDStatusEnum.NONE)),
            temp=int(env_obs.get("temp", 16)),
            loc=int(env_obs.get("loc", SleeveLocationEnum.INFEED)),
        )

    # tolerate objects with attributes
    try:
        return SleeveObservation(
            meas=int(getattr(env_obs, "meas")),
            temp=int(getattr(env_obs, "temp")),
            loc=int(getattr(env_obs, "loc")),
        )
    except Exception:
        # last resort: plain tuple
        try:
            meas, temp, loc = env_obs
            return SleeveObservation(meas=int(meas), temp=int(temp), loc=int(loc))
        except Exception as e:
            raise TypeError(f"Unsupported env_obs type for obs_from_env: {type(env_obs)}") from e


@dataclass(frozen=True)
class SleeveState(State):
    """
    Latent state:
      - h: {ENOUGH, LOW}
      - delta_needed: LOW 情况下，需要额外升温的阈值（离散，单位℃，5℃步进）
    Observable state components:
      - temp, loc, inspected, inspect_passed, done
    """
    h: int
    delta_needed: int
    temp: int
    loc: int
    inspected: int  # 0/1
    inspect_passed: int  # 0/1  (true iff last INSPECT_ID at/above spec succeeded)
    done: int  # 0/1

    def encode(self) -> "SleeveState":  # type: ignore[override]
        return copy.deepcopy(self)

    @staticmethod
    def decode(obj: "SleeveState") -> "SleeveState":
        return copy.deepcopy(obj)

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(
            (
                int(self.h),
                int(self.delta_needed),
                int(self.temp),
                int(self.loc),
                int(self.inspected),
                int(self.inspect_passed),
                int(self.done),
            )
        )


def sleeve_empty_state() -> SleeveState:
    # Placeholder; not used by planning logic directly.
    return SleeveState(
        h=0,
        delta_needed=0,
        temp=16,
        loc=int(SleeveLocationEnum.INFEED),
        inspected=0,
        inspect_passed=0,
        done=0,
    )


def sleeve_empty_observation() -> SleeveObservation:
    return SleeveObservation(meas=int(SleeveIDStatusEnum.NONE), temp=16, loc=int(SleeveLocationEnum.INFEED))


# ============================================================
# Domain parameters (per-size specs can be swapped by the caller)
#
# You can override these by wrapping the generated model functions.
T_MIN = 16
T_MAX = 400
HEAT_DELTA = 5
T_REF = 16

# ------------------------
# LLM 输出的 TARGET_TEMP
# ------------------------
# 在真实系统里：
# - d16 已测、阈值已知，因此不确定性不在几何本身
# - 不确定性只在“LLM 给的 TARGET_TEMP 是否足够”
#
# 我们用 H∈{ENOUGH,LOW} 表达 80%/20% 的先验，并用 LOW_SUCCESS_CDF 表达
# “对于 LOW 的那 20%，额外加热 deltaT 越大越容易达标”的经验曲线。
TARGET_TEMP = 300

class SleeveHiddenH(IntEnum):
    ENOUGH = 0
    LOW = 1


# LOW 情况下，delta_needed 的经验 CDF：P(delta_needed <= deltaT) = cdf(deltaT)
# （deltaT=0 表示刚到 TARGET_TEMP；deltaT 以 5℃ 为步进）
LOW_SUCCESS_CDF: Dict[int, float] = {
    0: 0.0,
    5: 0.3,
    10: 0.8,
    15: 0.95,
    20: 1.0,  # tail bucket
}

def belief_prior_for_material(material: str) -> Tuple[float, Dict[int, float]]:
    """
    按材料返回 POMDP 规划器/粒子滤波使用的“先验信念”：
    - p_enough: P(H=ENOUGH)（LLM 的 TARGET_TEMP 足够的先验概率）
    - low_success_cdf: 在 H=LOW 时 delta_needed 的经验 CDF（必须从 0 开始，且最后一个点概率为 1.0）

    设计目标：
    - 三种材料有不同的先验（用户要求）
    - tail bucket 覆盖到 100℃（与 TrueDynamicsParams.delta_needed_cap_c 对齐，避免 PF all-impossible）
    """
    m = str(material).strip().lower()
    # 同时兼容中文材料名和 CLI 的 ASCII alias
    if m in {"steel", "carbon_steel", "碳素钢"}:
        # 更相信“刚到 TARGET 就够”（更偏 ENOUGH，LOW 也更偏小 delta）
        p_enough = 0.85
        cdf = {
            0: 0.0,
            5: 0.55,
            10: 0.85,
            15: 0.94,
            20: 0.97,
            30: 0.99,
            40: 0.995,
            60: 0.999,
            100: 1.0,  # tail bucket (cap)
        }
        return float(p_enough), {int(k): float(v) for k, v in cdf.items()}

    if m in {"copper", "copper_alloy", "铜合金"}:
        # 适中：ENOUGH 先验略低，LOW 的 delta 分布更分散
        p_enough = 0.70
        cdf = {
            0: 0.0,
            5: 0.25,
            10: 0.55,
            15: 0.75,
            20: 0.88,
            30: 0.96,
            40: 0.985,
            60: 0.995,
            100: 1.0,
        }
        return float(p_enough), {int(k): float(v) for k, v in cdf.items()}

    if m in {"aluminum", "aluminium", "aluminum_alloy", "铝合金"}:
        # 更保守：更可能需要额外加热（ENOUGH 先验更低，LOW 的 tail 更重）
        p_enough = 0.55
        cdf = {
            0: 0.0,
            5: 0.15,
            10: 0.35,
            15: 0.55,
            20: 0.70,
            30: 0.84,
            40: 0.92,
            60: 0.975,
            100: 1.0,
        }
        return float(p_enough), {int(k): float(v) for k, v in cdf.items()}

    # fallback: keep old defaults, but extend tail bucket to 100℃ for stability
    p_enough = 0.80
    cdf = dict(LOW_SUCCESS_CDF)
    if int(max(cdf)) < 100:
        cdf = dict(cdf)
        cdf[100] = 1.0
    return float(p_enough), {int(k): float(v) for k, v in cdf.items()}

ROLLOUT_LOW_PASS_TARGET = 0.95
ROLLOUT_SAFE_DELTA = min(
    (dt for dt, p in LOW_SUCCESS_CDF.items() if float(p) >= float(ROLLOUT_LOW_PASS_TARGET)),
    default=max(LOW_SUCCESS_CDF),
)
ROLLOUT_SAFE_TEMP = int(TARGET_TEMP) + int(ROLLOUT_SAFE_DELTA)


def set_target_temp(
    target_temp_c: int,
    *,
    room_temp_c: int | None = None,
    ensure_t_max_margin_c: int = 50,
    cap_t_max_c: int | None = None,
) -> int:
    """
    Update global TARGET_TEMP at runtime and recompute derived rollout parameters.

    Notes:
    - The template uses globals (TARGET_TEMP, ROLLOUT_SAFE_TEMP, etc.) inside transition/observation/reward/rollout.
      Updating them before env/wrapper creation is sufficient.
    - Optionally update T_MIN/T_REF from room temperature, and ensure T_MAX is high enough.
    """
    global TARGET_TEMP, T_MIN, T_MAX, T_REF, ROLLOUT_SAFE_DELTA, ROLLOUT_SAFE_TEMP

    if room_temp_c is not None:
        T_MIN = int(room_temp_c)
        T_REF = int(room_temp_c)

    TARGET_TEMP = int(target_temp_c)

    # Ensure heating/clipping bounds can accommodate the new target.
    if int(T_MAX) < int(TARGET_TEMP) + int(ensure_t_max_margin_c):
        T_MAX = int(TARGET_TEMP) + int(ensure_t_max_margin_c)

    # Optional upper cap for fairness / protocol constraints (e.g. max setpoint).
    # Keep invariant: T_MAX >= TARGET_TEMP.
    if cap_t_max_c is not None:
        cap = int(cap_t_max_c)
        T_MAX = max(int(TARGET_TEMP), min(int(T_MAX), int(cap)))

    # Recompute safe rollout threshold.
    ROLLOUT_SAFE_DELTA = min(
        (dt for dt, p in LOW_SUCCESS_CDF.items() if float(p) >= float(ROLLOUT_LOW_PASS_TARGET)),
        default=max(LOW_SUCCESS_CDF),
    )
    ROLLOUT_SAFE_TEMP = int(TARGET_TEMP) + int(ROLLOUT_SAFE_DELTA)
    return int(TARGET_TEMP)


def set_low_success_cdf(
    low_success_cdf: Dict[int, float],
    *,
    rollout_low_pass_target: float | None = None,
) -> None:
    """
    样本回流接口：用新的经验 CDF 覆盖 LOW_SUCCESS_CDF，并据此重算 rollout 安全阈值。

    注意：
    - 该函数只影响“规划器的先验/rollout”，不改变真实环境（真实环境由外部固定隐变量决定）。
    - 调用后，对“新创建的 env/wrapper”生效；已创建实例不会自动更新其 init_sampler。
    """
    global LOW_SUCCESS_CDF, ROLLOUT_LOW_PASS_TARGET, ROLLOUT_SAFE_DELTA, ROLLOUT_SAFE_TEMP

    cdf = {int(k): float(v) for k, v in dict(low_success_cdf).items()}
    items = sorted(cdf.items())
    if not items or items[0][0] != 0:
        raise ValueError("LOW_SUCCESS_CDF must start with key=0.")
    if abs(float(items[-1][1]) - 1.0) > 1e-9:
        raise ValueError("LOW_SUCCESS_CDF must end with probability 1.0 (tail bucket).")
    prev = -1.0
    for dt, p in items:
        if dt < 0:
            raise ValueError("LOW_SUCCESS_CDF keys must be non-negative.")
        if p < prev - 1e-12:
            raise ValueError("LOW_SUCCESS_CDF must be non-decreasing.")
        prev = float(p)

    LOW_SUCCESS_CDF = {int(k): float(v) for k, v in items}

    if rollout_low_pass_target is not None:
        ROLLOUT_LOW_PASS_TARGET = float(rollout_low_pass_target)

    # Recompute safe rollout threshold based on current TARGET_TEMP.
    ROLLOUT_SAFE_DELTA = min(
        (dt for dt, p in LOW_SUCCESS_CDF.items() if float(p) >= float(ROLLOUT_LOW_PASS_TARGET)),
        default=max(LOW_SUCCESS_CDF),
    )
    ROLLOUT_SAFE_TEMP = int(TARGET_TEMP) + int(ROLLOUT_SAFE_DELTA)


def _delta_needed_pmf_from_cdf(cdf: Dict[int, float]) -> List[Tuple[int, float]]:
    items = sorted((int(k), float(v)) for k, v in cdf.items())
    if not items or items[0][0] != 0:
        raise ValueError("LOW_SUCCESS_CDF must start with key=0.")
    if abs(items[-1][1] - 1.0) > 1e-9:
        raise ValueError("LOW_SUCCESS_CDF must end with probability 1.0 (tail bucket).")
    prev_c = 0.0
    pmf: List[Tuple[int, float]] = []
    for dt, c in items:
        if c < prev_c - 1e-12:
            raise ValueError("LOW_SUCCESS_CDF must be non-decreasing.")
        p = max(0.0, float(c) - float(prev_c))
        pmf.append((int(dt), float(p)))
        prev_c = float(c)
    # drop zero-prob mass points to keep sampling stable
    pmf = [(dt, p) for dt, p in pmf if p > 0.0]
    # numeric guard: re-normalize
    s = sum(p for _, p in pmf)
    if s <= 0.0:
        # degenerate: always needs the max delta
        return [(int(items[-1][0]), 1.0)]
    return [(dt, p / s) for dt, p in pmf]


def _sample_from_pmf(pmf: List[Tuple[int, float]]) -> int:
    r = float(random.random())
    acc = 0.0
    for val, p in pmf:
        acc += float(p)
        if r <= acc + 1e-12:
            return int(val)
    return int(pmf[-1][0])


def _id_status(h: int, delta_needed: int, temp: int) -> int:
    """
    Deterministic ID判定（INSPECT 无噪声）:
    - temp < TARGET_TEMP: 视为不足 -> TOO_SMALL
    - temp >= TARGET_TEMP:
        * ENOUGH: IN_SPEC
        * LOW: 当 (temp - TARGET_TEMP) >= delta_needed -> IN_SPEC，否则 TOO_SMALL
    """
    t = int(temp)
    if t < int(TARGET_TEMP):
        return int(SleeveIDStatusEnum.TOO_SMALL)
    if int(h) == int(SleeveHiddenH.ENOUGH):
        return int(SleeveIDStatusEnum.IN_SPEC)
    delta_t = int(t) - int(TARGET_TEMP)
    return int(SleeveIDStatusEnum.IN_SPEC) if int(delta_t) >= int(delta_needed) else int(SleeveIDStatusEnum.TOO_SMALL)

# ============================================================
# 需要生成的内容部分（按 tiger_env.py 的风格：返回可调用的 model 函数）

def sleeve_initial_model_gen(
    *,
    p_enough: float = 0.8,
    low_success_cdf: Dict[int, float] | None = None,
    init_temp: int = 16,
    init_loc: int = int(SleeveLocationEnum.HEATER),
) -> InitialModel:
    """Sample latent H (ENOUGH/LOW) and (for LOW) delta_needed; temp/location are deterministic."""
    cdf = dict(low_success_cdf or LOW_SUCCESS_CDF)
    pmf = _delta_needed_pmf_from_cdf(cdf)

    def initial_model(_: SleeveState | None) -> SleeveState:
        h = int(SleeveHiddenH.ENOUGH) if random.random() < float(p_enough) else int(SleeveHiddenH.LOW)
        delta_needed = 0 if h == int(SleeveHiddenH.ENOUGH) else int(_sample_from_pmf(pmf))
        return SleeveState(
            h=int(h),
            delta_needed=int(delta_needed),
            temp=int(init_temp),
            loc=int(init_loc),
            inspected=0,
            inspect_passed=0,
            done=0,
        )

    return initial_model


def sleeve_transition_model_gen() -> TransitionModel:
    """
    Transition:
      - HEAT: temp += 5 (clipped); others unchanged.
      - HEAT_TO_TARGET: temp := max(temp, TARGET_TEMP) (clipped); others unchanged.
      - INSPECT_ID: inspected=1 only at INSPECT station (otherwise no-op)
      - ASSEMBLE: done=1 (always terminal; reward model will decide success/fail)
    """

    def transition_model(state: SleeveState, action: SleeveActions) -> SleeveState:
        s = SleeveState.decode(state)

        if int(s.done) == 1:
            return s

        # ---------------------------
        # Process constraint:
        # Before passing INSPECT (inspected==1 and IN_SPEC),
        # only HEAT / HEAT_TO_TARGET / INSPECT_ID are allowed.
        # Other actions become no-ops (planner should already mask them).
        # ---------------------------
        passed_inspect = int(s.inspect_passed) == 1
        if not bool(passed_inspect):
            if action not in (SleeveActions.HEAT, SleeveActions.HEAT_TO_TARGET, SleeveActions.INSPECT_ID):
                return s

        if action == SleeveActions.HEAT:
            t2 = int(s.temp) + int(HEAT_DELTA)
            t2 = max(int(T_MIN), min(int(T_MAX), int(t2)))
            return SleeveState(
                h=int(s.h),
                delta_needed=int(s.delta_needed),
                temp=int(t2),
                loc=int(s.loc),
                inspected=int(s.inspected),
                inspect_passed=int(s.inspect_passed),
                done=int(s.done),
            )

        if action == SleeveActions.HEAT_TO_TARGET:
            t2 = int(s.temp)
            if int(t2) < int(TARGET_TEMP):
                t2 = int(TARGET_TEMP)
            t2 = max(int(T_MIN), min(int(T_MAX), int(t2)))
            return SleeveState(
                h=int(s.h),
                delta_needed=int(s.delta_needed),
                temp=int(t2),
                loc=int(s.loc),
                inspected=int(s.inspected),
                inspect_passed=int(s.inspect_passed),
                done=int(s.done),
            )

        if action == SleeveActions.INSPECT_ID:
            # Possible at HEATER (in-situ measurement) or INSPECT station.
            if int(s.loc) in (int(SleeveLocationEnum.HEATER), int(SleeveLocationEnum.INSPECT)):
                ok = _id_status(int(s.h), int(s.delta_needed), int(s.temp)) == int(SleeveIDStatusEnum.IN_SPEC)
                return SleeveState(
                    h=int(s.h),
                    delta_needed=int(s.delta_needed),
                    temp=int(s.temp),
                    loc=int(s.loc),
                    inspected=1,
                    inspect_passed=1 if bool(ok) else 0,
                    done=int(s.done),
                )
            return s

        if action == SleeveActions.ASSEMBLE:
            return SleeveState(
                h=int(s.h),
                delta_needed=int(s.delta_needed),
                temp=int(s.temp),
                loc=int(s.loc),
                inspected=int(s.inspected),
                inspect_passed=int(s.inspect_passed),
                done=1,
            )

        return s

    return transition_model


def sleeve_observation_model_gen() -> ObservationModel:
    """
    Observation:
      - Always returns (temp, loc) deterministically.
      - INSPECT_ID at INSPECT station: meas = true id_status (noiseless)
      - otherwise: meas = NONE
    """

    def observation_model(
        state: SleeveState, action: SleeveActions, empty_obs: SleeveObservation
    ) -> SleeveObservation:
        _ = empty_obs
        if action == SleeveActions.INSPECT_ID and int(state.loc) in (
            int(SleeveLocationEnum.HEATER),
            int(SleeveLocationEnum.INSPECT),
        ):
            meas = _id_status(int(state.h), int(state.delta_needed), int(state.temp))
        else:
            meas = int(SleeveIDStatusEnum.NONE)
        return SleeveObservation(meas=int(meas), temp=int(state.temp), loc=int(state.loc))

    return observation_model


def sleeve_reward_model_gen(
    *,
    heat_cost_per_deg: float = 0.02,   # cost per 1℃; per HEAT step cost = 5*heat_cost_per_deg
    # Assemble 代价：用 grasp_cost 表示（ASSEMBLE 成功/失败都会扣除）。
    grasp_cost: float = 50.0,
    place_cost: float = 0.2,
    # Inspect 代价翻倍：原 2 -> 4
    inspect_cost: float = 4.0,
    inspect_fail_penalty: float = 10.0,
    # 为避免“高 assemble 代价导致成功装配回报为负”而让策略拖延不装配，
    # 默认将成功终止奖励提高（实验脚本里仍可通过参数覆盖）。
    assemble_success_reward: float = 200.0,
    assemble_fail_penalty: float = 30.0,
    illegal_assemble_penalty: float = 50.0,
) -> RewardModel:
    """
    Reward returns (reward, done).
    - HEAT: negative proportional to deltaT
    - INSPECT: negative costs
        * If INSPECT_ID fails (inspect_passed==0), apply an additional penalty to discourage risky early inspection.
    - ASSEMBLE:
        * If inspected and InSpec -> +assemble_success_reward (terminal)
        * Else -> -illegal_assemble_penalty (terminal)
      (If you want "inspected but not in spec" to be a softer fail, set illegal_assemble_penalty accordingly.)
    """

    def reward_model(
        state: SleeveState, action: SleeveActions, next_state: SleeveState
    ) -> Tuple[float, bool]:
        if action == SleeveActions.HEAT:
            return -float(heat_cost_per_deg) * float(HEAT_DELTA), False

        if action == SleeveActions.HEAT_TO_TARGET:
            dt = max(0, int(TARGET_TEMP) - int(state.temp))
            return -float(heat_cost_per_deg) * float(dt), False

        if action == SleeveActions.INSPECT_ID:
            # Base inspection cost (always paid when INSPECT_ID is executed).
            # Additional penalty when inspection FAILS (monitor triggers), aligned with RV100 counting.
            penalty = 0.0
            try:
                # If transition performed an inspection, next_state.inspected==1.
                # Failure means inspect_passed==0.
                did_inspect = int(getattr(next_state, "inspected", 0)) == 1
                passed = int(getattr(next_state, "inspect_passed", 0)) == 1
                if bool(did_inspect) and not bool(passed):
                    penalty = float(abs(float(inspect_fail_penalty)))
            except Exception:
                penalty = 0.0
            return -(float(inspect_cost) + float(penalty)), False

        if action == SleeveActions.ASSEMBLE:
            # terminal either way
            passed = int(state.inspect_passed) == 1
            # GRASP is fused into ASSEMBLE: apply its cost here.
            if bool(passed):
                return float(assemble_success_reward) - float(grasp_cost), True
            return -(float(illegal_assemble_penalty) + float(grasp_cost)), True

        # default
        return 0.0, bool(int(next_state.done) == 1)

    return reward_model


# ============================================================
def rollout_policy(state: Any) -> int:
    """
    Rollout heuristic (important for POMCP):
    - Heat to at least ROLLOUT_SAFE_TEMP (e.g. TARGET+15 where LOW pass>=95%) before the first INSPECT_ID.
    - This reduces the chance of an expensive failed inspection, matching the process intuition.
    """
    try:
        s = state
        temp = int(getattr(s, "temp", T_REF))
        loc = int(getattr(s, "loc", int(SleeveLocationEnum.HEATER)))
        inspected = int(getattr(s, "inspected", 0))
        passed = int(getattr(s, "inspect_passed", 0))

        # Before first inspection, heat a bit beyond TARGET to reduce failure probability.
        if inspected == 0 and temp < int(ROLLOUT_SAFE_TEMP):
            return int(SleeveActions.HEAT)
        if inspected == 0:
            # in-situ inspect at heater
            return int(SleeveActions.INSPECT_ID)
        if passed == 0:
            # If we already inspected but didn't pass, keep heating then re-inspect.
            if temp < int(ROLLOUT_SAFE_TEMP):
                return int(SleeveActions.HEAT)
            return int(SleeveActions.INSPECT_ID)
        # After passing INSPECT, directly ASSEMBLE (movement/GRASP is fused into this step).
        return int(SleeveActions.ASSEMBLE)
    except Exception:
        return int(random.choice(list(SleeveActions)))



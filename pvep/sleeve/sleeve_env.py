from __future__ import annotations

import copy
import enum
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from structs import (
    Environment,
    Heuristic,
    InitialModel,
    Observation,
    ObservationModel,
    RewardModel,
    State,
    TransitionModel,
)

from sleeve_pomdp_template import (
    HEAT_DELTA,
    SleeveActions,
    SleeveIDStatusEnum,
    SleeveLocationEnum,
    SleeveObservation,
    SleeveState,
    sleeve_empty_observation,
    sleeve_initial_model_gen,
    sleeve_observation_model_gen,
    sleeve_reward_model_gen,
    sleeve_transition_model_gen,
)


def sleeve_heuristic_gen() -> Heuristic:
    return lambda _s: 0.0


class SleeveEnv(Environment):
    """
    Sleeve (隔套) POMDP environment:
      - HEAT: +5℃ each step
      - INSPECT_ID: noiseless measurement of ID status
      - ASSEMBLE: terminal (success only if inspected and InSpec)
    """

    def __init__(
        self,
        max_steps: int = 200,
        *,
        # Prior for latent H
        p_enough: float = 0.8,
        init_temp: int = 16,
        init_loc: int = int(SleeveLocationEnum.HEATER),
        # Rewards
        heat_cost_per_deg: float = 0.02,
        grasp_cost: float = 50.0,
        place_cost: float = 0.2,
        inspect_cost: float = 4.0,
        inspect_fail_penalty: float = 10.0,
        assemble_success_reward: float = 200.0,
        assemble_fail_penalty: float = 30.0,
        illegal_assemble_penalty: float = 50.0,
        **kwargs: Any,
    ):
        super().__init__(max_steps=max_steps)
        self._initial_model = sleeve_initial_model_gen(
            p_enough=float(p_enough),
            init_temp=int(init_temp),
            init_loc=int(init_loc),
        )
        self._transition_model = sleeve_transition_model_gen()
        self._observation_model = sleeve_observation_model_gen()
        self._reward_model = sleeve_reward_model_gen(
            heat_cost_per_deg=float(heat_cost_per_deg),
            grasp_cost=float(grasp_cost),
            place_cost=float(place_cost),
            inspect_cost=float(inspect_cost),
            inspect_fail_penalty=float(inspect_fail_penalty),
            assemble_success_reward=float(assemble_success_reward),
            assemble_fail_penalty=float(assemble_fail_penalty),
            illegal_assemble_penalty=float(illegal_assemble_penalty),
        )
        self.empty_observation = sleeve_empty_observation()
        self._current_state: SleeveState | None = None

    def reset(self, seed: int = 0) -> SleeveState:
        random.seed(int(seed))
        self.current_step = 0
        self._current_state = SleeveState.decode(self._initial_model(None))
        assert self._current_state is not None, "Initial state must be set."
        return self._current_state

    def step(
        self, action: int
    ) -> Tuple[SleeveObservation, SleeveState, float, bool, bool, Dict[str, Any]]:
        act_enum = SleeveActions(int(action))
        assert self._current_state is not None, "Call reset() first."

        next_state = SleeveState.decode(self._transition_model(self._current_state, act_enum))
        obs = SleeveObservation.decode(self._observation_model(next_state, act_enum, self.empty_observation))
        reward, done = self._reward_model(self._current_state, act_enum, next_state)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # 与 SleeveEnv 不同：这里把 terminal state 也写回，方便外部读取最终状态。
        self._current_state = next_state

        info: Dict[str, Any] = {}
        info["temp"] = int(next_state.temp)
        info["loc"] = int(next_state.loc)
        info["inspected"] = int(next_state.inspected)
        info["meas"] = int(obs.meas)

        return obs, next_state, float(reward), bool(done), bool(truncated), info

    @property
    def current_state(self) -> SleeveState | None:
        return self._current_state

    def visualize_episode(
        self,
        states: List[SleeveState],
        observations: List[SleeveObservation],
        actions: List[SleeveActions],
        episode_num: int,
    ) -> None:
        print(f"\\nEpisode {episode_num} trace:")
        for t, s in enumerate(states):
            act_str = f", action={actions[t].name}" if t < len(actions) else ""
            o_str = f", obs={observations[t]}" if t < len(observations) else ""
            print(f"  t={t}: state={s}{o_str}{act_str}")
        print()


class SleeveEnvFixedLatent(Environment):
    """
    真实实验用环境（固定隐变量 H 与 delta_needed）：

    - 与 SleeveEnv 使用相同的 transition / observation / reward 模型
    - reset() 时不再从先验采样，而是直接使用指定的 (h, delta_needed)

    目的：让每个实验样本有“确定的真实难度”，同时仍可用现有 POMCP+PF wrapper 进行规划与信念更新。
    """

    def __init__(
        self,
        *,
        fixed_h: int,
        fixed_delta_needed: int,
        max_steps: int = 200,
        init_temp: int = 16,
        init_loc: int = int(SleeveLocationEnum.HEATER),
        # Rewards
        heat_cost_per_deg: float = 0.02,
        grasp_cost: float = 50.0,
        place_cost: float = 0.2,
        inspect_cost: float = 4.0,
        inspect_fail_penalty: float = 10.0,
        assemble_success_reward: float = 200.0,
        assemble_fail_penalty: float = 30.0,
        illegal_assemble_penalty: float = 50.0,
        **kwargs: Any,
    ):
        _ = kwargs
        super().__init__(max_steps=max_steps)

        self.fixed_h = int(fixed_h)
        self.fixed_delta_needed = int(fixed_delta_needed)

        self.init_temp = int(init_temp)
        self.init_loc = int(init_loc)

        self._transition_model = sleeve_transition_model_gen()
        self._observation_model = sleeve_observation_model_gen()
        self._reward_model = sleeve_reward_model_gen(
            heat_cost_per_deg=float(heat_cost_per_deg),
            grasp_cost=float(grasp_cost),
            place_cost=float(place_cost),
            inspect_cost=float(inspect_cost),
            inspect_fail_penalty=float(inspect_fail_penalty),
            assemble_success_reward=float(assemble_success_reward),
            assemble_fail_penalty=float(assemble_fail_penalty),
            illegal_assemble_penalty=float(illegal_assemble_penalty),
        )
        self.empty_observation = sleeve_empty_observation()
        self._current_state: SleeveState | None = None
        self.current_step = 0

    def reset(self, seed: int = 0) -> SleeveState:
        random.seed(int(seed))
        self.current_step = 0
        self._current_state = SleeveState(
            h=int(self.fixed_h),
            delta_needed=int(self.fixed_delta_needed),
            temp=int(self.init_temp),
            loc=int(self.init_loc),
            inspected=0,
            inspect_passed=0,
            done=0,
        )
        return SleeveState.decode(self._current_state)

    def step(
        self, action: int
    ) -> Tuple[SleeveObservation, SleeveState, float, bool, bool, Dict[str, Any]]:
        act_enum = SleeveActions(int(action))
        assert self._current_state is not None, "Call reset() first."

        next_state = SleeveState.decode(self._transition_model(self._current_state, act_enum))
        obs = SleeveObservation.decode(self._observation_model(next_state, act_enum, self.empty_observation))
        reward, done = self._reward_model(self._current_state, act_enum, next_state)

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        if not done and not truncated:
            self._current_state = next_state

        info: Dict[str, Any] = {}
        info["temp"] = int(next_state.temp)
        info["loc"] = int(next_state.loc)
        info["inspected"] = int(next_state.inspected)
        info["meas"] = int(obs.meas)
        info["fixed_h"] = int(self.fixed_h)
        info["fixed_delta_needed"] = int(self.fixed_delta_needed)

        return obs, next_state, float(reward), bool(done), bool(truncated), info

    @property
    def current_state(self) -> SleeveState | None:
        return self._current_state

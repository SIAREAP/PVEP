from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from particle_filter import ParticleFilter
from AdhocReasoningEnv import AdhocAgent

import sleeve_pomdp_template as sleeve_tpl
from sleeve_pomdp_template import (
    HEAT_DELTA,
    SleeveActions,
    SleeveIDStatusEnum,
    SleeveLocationEnum,
    SleeveObservation,
    SleeveState,
    obs_from_env,
    sleeve_empty_observation,
    sleeve_initial_model_gen,
    sleeve_observation_model_gen,
    sleeve_reward_model_gen,
    sleeve_transition_model_gen,
    rollout_policy as sleeve_rollout_policy,
    _id_status,  # type: ignore
)


class _StateSetLike:
    def is_final_state(self, s: Any) -> bool:
        return bool(getattr(s, "_done", False))


@dataclass
class SleeveParticle:
    state: SleeveState
    last_obs: SleeveObservation

    def latent(self) -> SleeveState:
        return SleeveState.decode(self.state)


class SleevePFSimEnv:
    """
    Simulatable env used inside POMCP rollouts (generative model).
    Stored as particles in POMCP's search tree.
    """

    def __init__(
        self,
        *,
        transition_fn,
        observe_fn,
        reward_fn,
        particle: SleeveParticle,
        rng: Optional[random.Random] = None,
        pf: Optional[ParticleFilter[SleeveParticle, int, SleeveObservation]] = None,
        rollout_policy_fn=None,
    ) -> None:
        self.transition_fn = transition_fn
        self.observe_fn = observe_fn
        self.reward_fn = reward_fn
        self.particle = particle
        self.rng = rng or random.Random()
        self.pf = pf  # root-only for belief sampling
        self._rollout_policy_fn = rollout_policy_fn or sleeve_rollout_policy

        self.state_set = _StateSetLike()
        self.simulation = True
        self._done = False

        self.state: Dict[str, Any] = {
            "obs": self.get_observation(),
        }

    def _seed_globals_from_rng(self) -> None:
        # sleeve_* functions use global random; seed it from per-sim RNG.
        random.seed(int(self.rng.getrandbits(32)))

    def get_actions_list(self) -> List[int]:
        """
        Action masking to match the real process constraint:
        - Before passing INSPECT (i.e., until inspected==1 and IN_SPEC), only heating actions are allowed.
        - INSPECT_ID allowed at HEATER (in-situ) or INSPECT station.
        - Movement to ASSEMBLY only allowed after passing INSPECT (transition_fn also enforces this).
        """
        s = self.particle.latent()
        if bool(getattr(s, "done", 0)):
            return []

        # NOTE: HEAT_TO_TARGET 是固定的预处理动作，不参与 POMDP 规划动作集合
        heat_actions = [int(SleeveActions.HEAT)]

        loc = int(getattr(s, "loc", int(SleeveLocationEnum.HEATER)))
        temp = int(getattr(s, "temp", 16))
        inspected = int(getattr(s, "inspected", 0))
        passed = int(getattr(s, "inspect_passed", 0)) == 1

        # At HEATER (or INFEED), until reaching TARGET_TEMP we only heat.
        if loc in (int(SleeveLocationEnum.HEATER), int(SleeveLocationEnum.INFEED)):
            if temp < int(sleeve_tpl.TARGET_TEMP):
                return heat_actions
            # Reached target: allow in-situ inspect; policy can choose to delay it if optimal.
            if inspected == 0:
                return heat_actions + [int(SleeveActions.INSPECT_ID)]
            if bool(passed):
                return heat_actions + [int(SleeveActions.ASSEMBLE)]
            return heat_actions + [int(SleeveActions.INSPECT_ID)]

        # At INSPECT: can heat or inspect; only after passing inspect can move on.
        if loc == int(SleeveLocationEnum.INSPECT):
            actions = list(heat_actions) + [int(SleeveActions.INSPECT_ID)]
            if bool(passed):
                actions.append(int(SleeveActions.ASSEMBLE))
            return actions

        # At ASSEMBLY: only after passing inspect can assemble.
        if loc == int(SleeveLocationEnum.ASSEMBLY):
            actions = list(heat_actions)
            if bool(passed):
                actions.append(int(SleeveActions.ASSEMBLE))
            return actions

        return heat_actions

    def get_observation(self) -> SleeveObservation:
        return SleeveObservation.decode(self.particle.last_obs)

    def observation_is_equal(self, obs: Any, _agent: Any = None) -> bool:
        cur = self.get_observation()
        o = obs_from_env(obs)
        return (int(cur.meas), int(cur.temp), int(cur.loc)) == (int(o.meas), int(o.temp), int(o.loc))

    def hash_observation(self, _agent: Any = None) -> int:
        return hash(self.get_observation())

    def hash_state(self, _agent: Any = None) -> int:
        return hash(self.particle.latent())

    def default_policy(self) -> int:
        """POMCP rollout policy."""
        return self._rollout_policy_fn(self.particle.latent())

    def copy(self) -> "SleevePFSimEnv":
        rng2 = random.Random()
        rng2.setstate(self.rng.getstate())
        e = SleevePFSimEnv(
            transition_fn=self.transition_fn,
            observe_fn=self.observe_fn,
            reward_fn=self.reward_fn,
            particle=copy.deepcopy(self.particle),
            rng=rng2,
            pf=None,
            rollout_policy_fn=self._rollout_policy_fn,
        )
        e._done = bool(self._done)
        return e

    def sample_state(self, _agent: Any = None) -> "SleevePFSimEnv":
        if self.pf is None:
            return self.copy()
        p = self.pf.sample()
        sim_rng = random.Random(self.rng.random())
        return SleevePFSimEnv(
            transition_fn=self.transition_fn,
            observe_fn=self.observe_fn,
            reward_fn=self.reward_fn,
            particle=p,
            rng=sim_rng,
            pf=None,
            rollout_policy_fn=self._rollout_policy_fn,
        )

    def step(self, action: int) -> Tuple["SleevePFSimEnv", float, bool, Dict[str, Any]]:
        a = int(action)
        s = self.particle.latent()

        self._seed_globals_from_rng()
        s_next = SleeveState.decode(self.transition_fn(s, SleeveActions(a)))

        self._seed_globals_from_rng()
        o_next = SleeveObservation.decode(self.observe_fn(s_next, SleeveActions(a), sleeve_empty_observation()))

        r, done = self.reward_fn(s, SleeveActions(a), s_next)
        self._done = bool(done)

        self.particle = SleeveParticle(state=s_next, last_obs=o_next)
        self.state["obs"] = self.get_observation()
        return self, float(r), bool(done), {"model_action": a}


def _obs_from_real_env_step(obs: Any) -> SleeveObservation:
    return obs_from_env(obs)


class SleeveRouteAWrapper:
    """
    Route-A wrapper around a real env with .step(action)->(obs,next_state,reward,done,truncated,info):
    - Belief maintained by a generic PF over latent (H, delta_needed).
    - Observation likelihood is analytic & deterministic:
        * INSPECT_ID: noiseless -> filter by id_status consistency
        * otherwise: (temp,loc) deterministic given s_next
    - Provides a POMCP-compatible copy()/step()/observation interface.
    """

    def __init__(
        self,
        real_env: Any,
        *,
        num_particles: int = 500,
        seed: int = 0,
        # Prior
        p_enough: float = 0.8,
        low_success_cdf: Dict[int, float] | None = None,
        init_temp: int = 16,
        init_loc: int = int(SleeveLocationEnum.HEATER),
        # Observation noise (important for ROS-backed real env):
        # - In sim, temp is deterministic and typically matches exactly.
        # - In ROS, sensor/readback can lag, overshoot, or jitter; strict equality will kill all particles.
        # If temp_obs_sigma_c > 0: use Gaussian log-likelihood: logp ∝ -(dt/sigma)^2 / 2
        # Else if temp_obs_tol_c > 0: use hard tolerance band: |dt| <= tol -> 0 else -inf
        temp_obs_sigma_c: float = 0.0,
        temp_obs_tol_c: int = 0,
        # ROS shortcut:
        # If True, derive "inspection passed" purely from observed temperature:
        #   passed := (temp >= TARGET_TEMP + pass_temp_margin_c)
        # This is useful when ROS mode does not provide /dqw/pub_D (diameter) or
        # when the real process defines eligibility-by-temperature.
        derive_pass_from_temp: bool = False,
        pass_temp_margin_c: int = 0,
        # Reward params (keep consistent with real env)
        heat_cost_per_deg: float = 0.02,
        grasp_cost: float = 50.0,
        place_cost: float = 0.2,
        inspect_cost: float = 4.0,
        inspect_fail_penalty: float = 10.0,
        assemble_success_reward: float = 200.0,
        assemble_fail_penalty: float = 30.0,
        illegal_assemble_penalty: float = 50.0,
    ) -> None:
        self._real_env = real_env
        self._rng = random.Random(int(seed))
        self._agent = AdhocAgent(index="A", atype="pomcp")
        self._temp_obs_sigma_c = float(temp_obs_sigma_c)
        self._temp_obs_tol_c = int(temp_obs_tol_c)
        self._derive_pass_from_temp = bool(derive_pass_from_temp)
        self._pass_temp_margin_c = int(pass_temp_margin_c)

        self._init_fn = sleeve_initial_model_gen(
            p_enough=float(p_enough),
            low_success_cdf=dict(low_success_cdf) if low_success_cdf is not None else None,
            init_temp=int(init_temp),
            init_loc=int(init_loc),
        )
        self._transition_fn = sleeve_transition_model_gen()
        self._observe_fn = sleeve_observation_model_gen()
        self._reward_fn = sleeve_reward_model_gen(
            heat_cost_per_deg=float(heat_cost_per_deg),
            grasp_cost=float(grasp_cost),
            place_cost=float(place_cost),
            inspect_cost=float(inspect_cost),
            inspect_fail_penalty=float(inspect_fail_penalty),
            assemble_success_reward=float(assemble_success_reward),
            assemble_fail_penalty=float(assemble_fail_penalty),
            illegal_assemble_penalty=float(illegal_assemble_penalty),
        )

        self._last_real_obs: SleeveObservation = sleeve_empty_observation()

        # -------------------- PF components -------------------- #
        def init_sampler(rng: random.Random) -> SleeveParticle:
            random.seed(int(rng.getrandbits(32)))
            s0 = SleeveState.decode(self._init_fn(None))
            return SleeveParticle(state=s0, last_obs=self._last_real_obs)

        def transition_sampler(s_prev: SleeveParticle, a: int, rng: random.Random) -> SleeveParticle:
            random.seed(int(rng.getrandbits(32)))
            s_next = SleeveState.decode(self._transition_fn(s_prev.latent(), SleeveActions(int(a))))
            # keep last_obs until postprocess sets it
            return SleeveParticle(state=s_next, last_obs=s_prev.last_obs)

        def obs_log_prob(o: SleeveObservation, s_next: SleeveParticle, a: int, _s_prev: Optional[SleeveParticle]) -> float:
            a = int(a)
            o = obs_from_env(o)
            sn = s_next.state

            # Location is treated as deterministic.
            if int(o.loc) != int(sn.loc):
                return float("-inf")

            # Temperature: deterministic in sim, but can be noisy/lagged in ROS.
            dt = int(o.temp) - int(sn.temp)
            sigma = float(self._temp_obs_sigma_c)
            tol = int(self._temp_obs_tol_c)
            if sigma > 0.0:
                # Gaussian log-likelihood up to an additive constant.
                z = float(dt) / float(sigma)
                temp_ll = -0.5 * (z * z)
            elif tol > 0:
                temp_ll = 0.0 if abs(int(dt)) <= int(tol) else float("-inf")
            else:
                # Backward-compatible strict equality.
                temp_ll = 0.0 if int(dt) == 0 else float("-inf")
            if not math.isfinite(float(temp_ll)):
                return float("-inf")

            # Measurement:
            if a == int(SleeveActions.INSPECT_ID):
                # Noiseless at HEATER (in-situ) or INSPECT station; otherwise meas is NONE.
                if int(sn.loc) in (int(SleeveLocationEnum.HEATER), int(SleeveLocationEnum.INSPECT)):
                    if bool(self._derive_pass_from_temp):
                        thr = int(sleeve_tpl.TARGET_TEMP) + int(self._pass_temp_margin_c)
                        true_meas = (
                            int(SleeveIDStatusEnum.IN_SPEC)
                            if int(o.temp) >= int(thr)
                            else int(SleeveIDStatusEnum.TOO_SMALL)
                        )
                    else:
                        true_meas = int(_id_status(int(sn.h), int(sn.delta_needed), int(sn.temp)))
                    return float(temp_ll) if int(o.meas) == int(true_meas) else float("-inf")
                return float(temp_ll) if int(o.meas) == int(SleeveIDStatusEnum.NONE) else float("-inf")
            # Other actions: meas must be NONE
            return float(temp_ll) if int(o.meas) == int(SleeveIDStatusEnum.NONE) else float("-inf")

        def postprocess(s: SleeveParticle, _a: int, o: SleeveObservation) -> SleeveParticle:
            # Keep the real observation for POMCP's observation interface
            s.last_obs = o  # type: ignore[misc]
            # For ROS-backed envs, align particle's observable (temp/loc) with the received observation
            # to prevent drift when the real temperature dynamics differ from the simplistic +5C model.
            try:
                st = SleeveState.decode(s.state)
                inspected = int(st.inspected)
                inspect_passed = int(st.inspect_passed)
                # 折中语义：只有显式 INSPECT_ID 才算 inspected=1。
                if bool(self._derive_pass_from_temp) and int(_a) == int(SleeveActions.INSPECT_ID):
                    try:
                        thr = int(sleeve_tpl.TARGET_TEMP) + int(self._pass_temp_margin_c)
                        if int(getattr(o, "temp", st.temp)) >= int(thr):
                            inspected = 1
                            inspect_passed = 1
                        else:
                            inspected = 1
                            inspect_passed = 0
                    except Exception:
                        pass
                s.state = SleeveState(
                    h=int(st.h),
                    delta_needed=int(st.delta_needed),
                    temp=int(getattr(o, "temp", st.temp)),
                    loc=int(getattr(o, "loc", st.loc)),
                    inspected=int(inspected),
                    inspect_passed=int(inspect_passed),
                    done=int(st.done),
                )
            except Exception:
                pass
            return s

        self._pf: ParticleFilter[SleeveParticle, int, SleeveObservation] = ParticleFilter(
            num_particles=int(num_particles),
            init_sampler=init_sampler,
            transition_sampler=transition_sampler,
            observation_log_prob=obs_log_prob,
            rng=random.Random(int(seed) + 12345),
            postprocess=postprocess,
            fallback_init=init_sampler,
        )
        self._pf.initialize()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_env, name)

    def get_adhoc_agent(self) -> AdhocAgent:
        return self._agent

    def get_observation(self) -> SleeveObservation:
        return SleeveObservation.decode(self._last_real_obs)

    def observation_is_equal(self, obs: Any, _agent: Any = None) -> bool:
        cur = self.get_observation()
        o = obs_from_env(obs)
        return (int(cur.meas), int(cur.temp), int(cur.loc)) == (int(o.meas), int(o.temp), int(o.loc))

    def hash_observation(self, _agent: Any = None) -> int:
        return hash(self.get_observation())

    def copy(self) -> SleevePFSimEnv:
        current_obs = self.get_observation()
        pf_copy = self._pf.copy()
        p0 = pf_copy.sample()
        p0.last_obs = current_obs
        root_rng = random.Random(int(self._rng.random() * 1e9))
        return SleevePFSimEnv(
            transition_fn=self._transition_fn,
            observe_fn=self._observe_fn,
            reward_fn=self._reward_fn,
            particle=p0,
            rng=root_rng,
            pf=pf_copy,
            rollout_policy_fn=sleeve_rollout_policy,
        )

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, _next_state, reward, done, truncated, info = self._real_env.step(int(action))
        _ = truncated

        o_real = _obs_from_real_env_step(obs)
        self._last_real_obs = o_real

        pf_info = self._pf.update(int(action), o_real)

        info = dict(info or {})
        info["pf_log_predictive"] = float(pf_info.log_predictive)
        info["pf_ess"] = float(pf_info.effective_n)
        info["pf_all_impossible"] = bool(pf_info.all_impossible)
        return obs, float(reward), bool(done), info

    def render(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None

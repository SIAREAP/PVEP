# 当前 Sleeve POMDP 模型

当前 POMDP 用于在 LLM 给出初始加热温度后，决定继续加热、检测、装配或安全终止。
`llm+pomdp+no_reflow` 与 `llm+pomdp+reflow` 使用同一个 POMDP；两者只在 LLM
proposal 的生成方式上不同。

## 状态与初始 belief

- 可观测状态：当前温度 `T`、最近一次检测结果、是否终止、已经检测过的温度。
- 隐藏状态：`delta = T_required - T_proposal`。
- `T_start = T_target = T_proposal`，装配过程固定降温 `10 C`。
- `broad` scope 的绝对所需温度候选范围为 `250...440 C`，步长 `5 C`；转换为
  相对 proposal 的 `delta` 后形成隐藏状态集合。
- `narrow` scope 不建立一张新的 proposal-centered 网格，而是从上述同一绝对温度网格中
  只保留落在 `T_proposal +/- 15 C` 内的候选。比如 proposal 为 `351 C` 时，候选为
  `340, 345, 350, 355, 360, 365 C`，对应 delta 为
  `-11, -6, -1, 4, 9, 14 C`。
- 初始 belief 在所有候选绝对所需温度上均匀分布。Reflow 历史不会直接修改 belief。

装配失败概率为：

```text
p_violate = sum_delta b(delta) * I[T - cooling - T_proposal - delta < 0]
```

## 动作与观测

动作集合：

- `HEAT_5`：升温 `5 C`。
- `INSPECT`：温度不变，获得带噪声的尺寸状态观测。
- `ASSEMBLE`：执行装配并终止。
- `SAFE_ABORT`：不装配并安全终止。

检测观测按装配 margin 划分：

| Margin | 观测 |
|---:|---|
| `< 0 C` | `TOO_SMALL` |
| `[0, 10) C` | `BORDERLINE` |
| `>= 10 C` | `IN_SPEC` |

当前两种 POMDP 方法都使用 `constant` 检测模型：上述三类正确观测概率分别为
`0.85 / 0.70 / 0.90`。执行 `INSPECT` 后通过 Bayes 规则更新 `b(delta)`。

## Reward 与风险 gate

| 事件 | Reward |
|---|---:|
| `HEAT_5` | `-5` |
| `INSPECT` | `-10` |
| 成功装配 | `+100 - 50 - 0.5 * overheat_C` |
| 失败装配 | `-50 - 100 = -150` |
| `SAFE_ABORT` | `-500` |

`violation_penalty=100` 把装配失败风险放入期望 reward。同时存在 hard gate：只有
`p_violate <= risk_threshold` 时，`ASSEMBLE` 才会进入 POMCP 的可选动作集合。达到
`450 C` 仍不满足 gate 时只能继续检测或安全终止；单个 episode 最多 `80` 步。

scope 交互实验的所有配置统一使用 `risk_threshold=0.3`。

## POMCP 求解

- 每个真实动作前重新运行 POMCP。
- 搜索深度 `20`，每次规划最多 `200` 次迭代，使用 `200` 个粒子。
- 折扣因子 `0.95`，启用 particle reinvigoration。
- 模拟隐藏状态从当前 belief 采样；执行真实 `INSPECT` 后再更新 belief 并重新规划。

## Reflow 接口

Reflow 为每种材料维护最近 `20` 个历史样例，成功和失败样例都保留，内容包括任务尺寸、
最终温度和装配结果。每种材料的第一个任务没有历史样例。LLM 根据这些样例产生新的
`T_proposal`，随后仍由上述 uniform-belief POMDP 求解；Reflow 本身不是额外的 POMDP
状态或 belief 初始化来源。

scope 交互实验采用 proposal replay：先冻结每个 task/config 的 proposal，再让 broad 与
narrow 复用同一个 proposal 和 `world_seed`。这样 scope 不会通过 PVEP 最终温度反向改变
后续 Reflow 历史，两个 scope 之间唯一变化是初始 belief support。

## 指标定义

- 只有执行 `ASSEMBLE` 且实际 margin `< 0` 时记为 `RVR=1`。
- `INSPECT` 得到 `TOO_SMALL` 不记 RVR。
- `SAFE_ABORT` 的 `Pass=0`、`RVR=0`。
- 过热目前只产生软成本，不会直接判定失败。

实现位置：[run_margin_pvep_demo.py](../run_margin_pvep_demo.py) 和
[run_margin_pvep_tasks.py](../run_margin_pvep_tasks.py)。scope 实验入口为
[run_scope2x2_experiment.py](../run_scope2x2_experiment.py)。

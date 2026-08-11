# Data and code map

## Submitted figures and released evidence

| Manuscript panel | Released evidence | Regeneration entry point |
|---|---|---|
| Fig. 2b, ARIAC corruption score | `results/ariac/Vbinary_sweep_per_order.csv` | `gen_fig2_ariac.py` |
| Fig. 2c, ARIAC inspection burden | `results/ariac/ariac_pertrial_admissibility.csv` | `gen_fig2_ariac.py` |
| Fig. 2d/e, nominal ablation and challenge recovery | `results/ariac/nominal_ablation_per_scenario.csv` | `gen_fig2_ariac.py` |
| Fig. 3b--e, fastening corruption and ablations | `results/tv/final10.csv` | `gen_fig3_tv.py` |
| Fig. 4b/c, rotor setpoint corruption | `results/rotor/table2_perturbation_sweep.csv` | `gen_fig4_rotor.py` |
| Fig. 4d/e, rotor method comparison | `results/rotor/table1_main_5_methods.csv` | `gen_fig4_rotor.py` |
| Fig. 5a, interface coverage | `results/ariac/vocabulary_coverage_variants.csv` | `gen_fig5_coverage_scaling.py` |
| Fig. 5b/c, flat-versus-bounded scaling | `results/ariac/flat_scaling_q200_q2000_raw.csv` | `gen_fig5_coverage_scaling.py` |
| Supplementary scope intervention | `results/rotor/rotor_scope2x2.csv` | `gen_fig6_scope_intervention.py` |
| ARIAC initial admissibility, $Q_0=0.345$ | `results/ariac/ariac_q0_definition_audit.csv` | `recompute_ariac_q0.py` |

`paper/figure_scripts/audit_figure_data.py` independently loads every plotted
table, recomputes aggregates and inferential checks, checks the canonical
20-seed scaling slice, and records SHA-256 hashes for every figure input.

For Fig. 3b/d, the plotted unsafe-episode rate is computed as
`1 - task_pass`, preserving the six-hole episode as the statistical unit.
Panels 3c/e use `total_cost`, the engineered decision cost summed over one
six-hole episode and displayed in arbitrary units.

Figs. 2--4 share a fixed method encoding: Heuristic/grey circle, FM/red cross,
FM + Repair/magenta triangle (Fig. 2 only), PVEP w/o POMDP/orange square,
PVEP w/o SG or PVEP--Binary/green diamond with a dashed trend, and PVEP/blue
star with a solid trend. In continuous panels, small translucent points are
individual scenarios (Fig. 2), episodes (Fig. 3), or trials (Fig. 4), and
light-grey lines follow the same case across settings. Binary panels omit
redundant 0/100 sample bands and show only the aggregate rate, 95% confidence
interval, and exact count. The red dashed line in Fig. 4 denotes the risk limit.

## ARIAC Fig. 2 configuration mapping

| Display label | Released raw configuration/column | Information planning | Replanning | Complete repair package | Nominal strict completion |
|---|---|---:|---|---:|---:|
| FM | `open_loop_vlm_nl` / `open_loop_vlm_nl_completion` | no | no | no | 11/50 |
| FM + Repair | `vlm_nl_re` / `vlm_nl_re_completion` | no | generic | no | 32/50 |
| PVEP w/o POMDP | `vlm_pddl_re` / `vlm_pddl_re_completion` | no | generic | no | 32/50 |
| PVEP w/o SG | `vlm_pddl` / `vlm_pddl_completion` | no | no | no | 11/50 |
| PVEP | `pomdp_our` / `pomdp_our_completion` | POMCP | yes | yes | 50/50 |

Every row retains the common structured executor-state/PDDL applicability
gate. The separate `V_greedy` information-policy ablation retains the full
repair package and is not the displayed `PVEP w/o POMDP` row.

The displayed `Heuristic` baselines retain legacy raw names `Human` (TV) and
`human` (rotor) so that released tables remain byte-stable. They are
engineering rules, not human-participant experiments.

## Additional ARIAC tables

Other files under `results/ariac/` provide trial-level baseline, challenge,
positive-control, and unified 900-second timeout-status records. Lineage-only
workstation paths are de-identified; trial identifiers, model snapshots,
scores, budgets, and failure/status fields are retained.

## Implementation map

- `ariac/competition_tutorials/ariac_pomdp/`: verifier and belief-space
  recovery implementation.
- `ariac/competition_tutorials/ariac_interface.py`: ARIAC simulator interface.
- `ariac/competition_tutorials/eval.py`: proposer and repair prompts.
- `ariac/competition_tutorials/ariac_pomdp/domain.pddl`: action domain.
- `pvep/ibpomcp/`: shared POMCP implementation.
- `pvep/kinova/fasten_pomdp.py`: fastening state/action/observation/cost model.
- `pvep/kinova/screw_hybrid_controller.py`: fastening controller.
- `pvep/sleeve/sleeve_pomdp_template.py`: thermal-domain POMDP.
- `pvep/sleeve/run_scope2x2_experiment.py`: controlled scope intervention.

## Restricted assets

Raw fixture imagery, camera poses, calibration files, and model weights are
not public. Editors or reviewers may request confidential access from the
corresponding author, subject to institutional and data-owner approval, a
data-use agreement, and no public redistribution. Approval cannot be
guaranteed for every request.

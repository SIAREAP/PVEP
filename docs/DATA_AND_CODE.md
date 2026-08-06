# Data and code map

## Primary released analyses

| Manuscript result | Released evidence | Recalculation entry point |
|---|---|---|
| ARIAC controlled corruption sweep | `results/ariac/Vbinary_sweep_per_order.csv` and `ariac_pertrial_admissibility.csv` | `gen_fig2_cross_domain_decoupling.py` |
| ARIAC nominal ablation 22/42/56/100 | `results/ariac/nominal_ablation_per_scenario.csv` | `gen_fig3_mechanism.py` |
| ARIAC initial admissibility Q0=0.345 | `results/ariac/ariac_q0_definition_audit.csv` | `recompute_ariac_q0.py` |
| ARIAC vocabulary/scope necessity | `results/ariac/vocabulary_coverage_variants.csv` | `gen_fig4_necessity.py` |
| ARIAC flat scaling | `results/ariac/flat_scaling_q200_q2000_raw.csv` | `gen_fig4_necessity.py` |
| Fastening corruption and ablations | `results/tv/final10.csv` | Fig. 2/3 scripts |
| Rotor method comparison and corruption | `results/rotor/table1_main_5_methods.csv`, `table2_perturbation_sweep.csv` | Fig. 2/3 scripts |
| Rotor scope intervention | `results/rotor/rotor_scope2x2.csv` | `gen_fig5_scope_intervention.py` |

`paper/figure_scripts/audit_figure_data.py` independently loads the plotted
tables, recomputes the aggregates and inferential checks, and records SHA-256
hashes for every figure input.

## Additional ARIAC tables

The remaining files under `results/ariac/` provide trial-level baseline,
challenge, positive-control, and unified 900-second timeout-status records.
Lineage-only workstation paths are de-identified; trial identifiers, model
snapshots, scores, completion fields, budgets, and failure/status fields are
retained.

## Implementation map

- `ariac/competition_tutorials/ariac_pomdp/`: verifier and belief-space
  recovery implementation.
- `ariac/competition_tutorials/ariac_interface.py`: ARIAC simulator interface.
- `ariac/competition_tutorials/eval.py`: proposer and repair prompts.
- `ariac/competition_tutorials/ariac_pomdp/domain.pddl`: action domain.
- `pvep/ibpomcp/`: shared POMCP implementation.
- `pvep/kinova/fasten_pomdp.py`: fastening state, action, observation, and cost model.
- `pvep/kinova/screw_hybrid_controller.py`: fastening controller.
- `pvep/sleeve/sleeve_pomdp_template.py`: thermal-domain POMDP.
- `pvep/sleeve/run_scope2x2_experiment.py`: controlled scope intervention.

# Data and code map

## Submitted figures and released evidence

| Manuscript panel | Released evidence | Regeneration entry point |
|---|---|---|
| Fig. 2b, ARIAC corruption score | `results/ariac/Vbinary_sweep_per_order.csv` | `gen_fig2_ariac.py` |
| Fig. 2c, ARIAC inspection burden | `results/ariac/ariac_pertrial_admissibility.csv` | `gen_fig2_ariac.py` |
| Fig. 2d/e, method scores and task-condition matrix | `results/ariac/nominal_ablation_per_scenario.csv` | `gen_fig2_ariac.py` |
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
The `112/120` state-decision result is a descriptive hole-level diagnostic;
no Wilson interval assuming independent holes is attached. In the rotor
corruption sweep, the same 90 trial identities repeat across four conditions.
The release therefore reports a per-condition `0/90` Wilson upper endpoint
(about 4.1%) and treats `0/360` only as a descriptive pooled event count.

Figs. 2--4 use consistent method encodings without redundant global legends:
Heuristic/grey circle, FM/red cross, FM + Repair/magenta triangle (Fig. 2
only), PVEP w/o POMDP/orange square, PVEP w/o SG/green diamond, and PVEP/blue
star. PVEP--Binary is visually distinct from PVEP w/o SG: it uses a purple
open circle and dashed trend and is directly labelled in Fig. 2b/c. Method
names in comparison panels are carried by the horizontal axes. Small
translucent points show individual cases; light-grey lines follow matched
cases where retained. Fig. 4e uses jittered trial points without pairing lines
to avoid a dense line web. Binary panels show aggregate rates, 95% Wilson
intervals, and exact counts. Continuous-panel error bars are 95% bootstrap
intervals. The red dashed risk limit in Fig. 4b is labelled directly.

## ARIAC Fig. 2 configuration mapping

| Display label | Released score column | Information planning | Feedback to re-proposal | Mean scenario-normalized score |
|---|---|---:|---|---:|
| FM | `open_loop_vlm_nl_score` | no | none | 51.8% |
| FM + Repair | `vlm_nl_re_score` | no | generic | 77.1% |
| PVEP w/o POMDP | `vlm_pddl_re_score` | no | generic | 69.8% |
| PVEP w/o SG | `our_error_score` (`V_error`) | POMCP | error message, no structured semantic gradient | 68.0% |
| PVEP | `pomdp_our_score` | POMCP | structured semantic-gradient package | 100.0% |

Every row retains the common structured executor-state/PDDL applicability
gate. `PVEP w/o SG` is the released `V_error` condition: it retains POMCP and
verifier error feedback but omits the structured semantic-gradient message.
The released `our_raw_score` field is `V_binary`, shown as PVEP--Binary in
Fig. 2b/c; it is not the w/o-SG row. The separate `V_greedy`
information-policy ablation retains the full repair package and is not the
displayed `PVEP w/o POMDP` row.

The Fig. 2d/e five-method score comparison and the natural-proposal audit reuse the
same scenario identities but come from distinct archived run series and report
different endpoints. They are not repeated estimates of one protocol. The
proposal-source, cache/regeneration, in-loop-backend, time-budget,
crash-adjudication, and metric fields—and explicit `NR` entries where legacy
records are incomplete—are listed in
`docs/PROTOCOL_AND_MODEL_PROVENANCE.md`.

The displayed `Heuristic` baselines retain legacy raw names `Human` (TV) and
`human` (rotor) so that released tables remain byte-stable. They are
engineering rules, not human-participant experiments.

## Additional ARIAC tables

Other files under `results/ariac/` provide trial-level baseline, challenge,
positive-control, and unified 900-second timeout-status records. Lineage-only
workstation paths are de-identified; trial identifiers, model snapshots,
scores, budgets, and failure/status fields are retained.

## Model API configuration and provenance

`ariac/competition_tutorials/eval.py` and the optional thermal LLM proposal
runner are provider-neutral. New foundation-model calls require the provider
endpoint and exact model identifier to be set explicitly through
`PVEP_OPENAI_*` or `SLEEVE_LLM_*` variables; no third-party endpoint or model
alias is selected by default. The exact model identifiers and available
archived run dates are recorded in `docs/PROTOCOL_AND_MODEL_PROVENANCE.md`.
Provider, endpoint, and proxy fields that were not stored with legacy
aggregates are marked `NR`; the release does not infer them from an old code
default. Temperature-zero decoding settings are reported as such, rather than
described as deterministic.

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

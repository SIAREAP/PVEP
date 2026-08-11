# PVEP submitted-figure regeneration

Run from the repository root:

```bash
python3 paper/figure_scripts/generate_all.py
```

The command audits source hashes and regenerates the quantitative panels of
the submitted Figs. 2--5, plus the supplementary scope-intervention figure.
Outputs are written to `paper/figure_scripts/` and copied to `paper/`. The
case-study overview images used in panel a
are publication assets in `paper/`; all plotted numerical values come from
curated trial-level files under `results/`.

## Figure-to-file mapping

- **Fig. 2 — ARIAC kitting (`fig2_ariac.pdf`)**
  - panel a: `paper/ariac_overview.png`
  - panel b: `results/ariac/Vbinary_sweep_per_order.csv`
  - panel c: `results/ariac/ariac_pertrial_admissibility.csv`
  - panels d/e: `results/ariac/nominal_ablation_per_scenario.csv`
  - displayed/raw nominal mapping: FM/`open_loop_vlm_nl`, FM +
    Repair/`vlm_nl_re`, PVEP w/o POMDP/`vlm_pddl_re`, PVEP w/o
    SG/`vlm_pddl`, PVEP/`pomdp_our`
- **Fig. 3 — television fastening (`fig3_tv.pdf`)**
  - panel a: `paper/tv_overview.png`
  - panels b--e: `results/tv/final10.csv`
  - the displayed **Heuristic** row retains raw configuration name `Human`
  - panels b/d plot `1 - task_pass` as the episode-level unsafe rate
  - panels c/e plot the engineered cost summed over one six-hole episode
- **Fig. 4 — rotor fitting (`fig4_rotor.pdf`)**
  - panel a: `paper/rotor_overview.png`
  - panels b/c: `results/rotor/table2_perturbation_sweep.csv`
  - panels d/e: `results/rotor/table1_main_5_methods.csv`
  - the displayed **Heuristic** row retains raw prefix `human`
  - the same 90 trial identities repeat across the four corruption levels;
    `0/360` is descriptive, while the zero-event Wilson upper endpoint is
    computed separately at `n=90`
- **Fig. 5 — interface coverage and bounded scaling
  (`fig5_coverage_scaling.pdf`)**
  - panel a: `results/ariac/vocabulary_coverage_variants.csv`
  - panels b/c: canonical slice of
    `results/ariac/flat_scaling_q200_q2000_raw.csv`, defined by
    `horizon = 8 * task_size + 4` and `latent_variables = task_size`
- **Supplementary scope intervention (`fig6_scope_intervention.pdf`)**
  - `results/rotor/rotor_scope2x2.csv`

## Shared visual encoding in Figs. 2--4

- Heuristic: grey circle; FM: red cross; FM + Repair (Fig. 2 only): magenta
  triangle; PVEP w/o POMDP: orange square; PVEP w/o SG and PVEP--Binary:
  green diamond with a dashed trend; PVEP: blue star with a solid trend.
- In continuous panels, a small translucent point is one scenario (Fig. 2),
  episode (Fig. 3), or trial (Fig. 4); a light-grey line follows that same case
  across settings. Binary panels omit redundant 0/100 sample bands and show
  only the aggregate rate, 95% confidence interval, and exact count. The red
  dashed line in Fig. 4 is the risk limit.

`figure_data_audit.json` records source hashes and the aggregates used by
these figures. To reproduce the repository-frozen ARIAC $Q_0$ result:

```bash
python3 paper/figure_scripts/recompute_ariac_q0.py
```

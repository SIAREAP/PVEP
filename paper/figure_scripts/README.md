# Figure and aggregate regeneration

Run from the repository root:

```bash
python paper/figure_scripts/generate_all.py
python paper/figure_scripts/recompute_ariac_q0.py
```

Data mapping:

- Fig. 2a: `results/ariac/Vbinary_sweep_per_order.csv` and
  `results/ariac/ariac_pertrial_admissibility.csv`
- Fig. 2b: `results/tv/final10.csv`
- Fig. 2c: `results/rotor/table2_perturbation_sweep.csv` with the open-loop
  reference from `results/rotor/table1_main_5_methods.csv`
- Fig. 3a: `results/ariac/nominal_ablation_per_scenario.csv`
- Fig. 3b: `results/tv/final10.csv`
- Fig. 3c: `results/rotor/table1_main_5_methods.csv`
- Fig. 4a: `results/ariac/vocabulary_coverage_variants.csv`
- Fig. 4b-c: `results/ariac/flat_scaling_q200_q2000_raw.csv`
- Fig. 5: `results/rotor/rotor_scope2x2.csv`
- ARIAC Q0: `results/ariac/ariac_q0_definition_audit.csv`

The audit script records input hashes and plotted aggregates in
`figure_data_audit.json`.

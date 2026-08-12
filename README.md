# PVEP reproduction repository

This repository accompanies **Verifier-Enforced Belief Space Planning for
Foundation Model Proposals across Industrial Robot Tasks**. It contains curated
trial-level result tables, deterministic analysis and figure-regeneration
scripts, and reference implementations of the planners, verifiers, and domain
models used in the released analyses.

The repository is synchronized with the author-identified manuscript.

## Reproduce the reported aggregates and figures

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python paper/figure_scripts/recompute_ariac_q0.py
python paper/figure_scripts/generate_all.py
python scripts/generate_manifest.py
python scripts/verify_release.py
```

The Q0 command must report 50 trials, 195 subgoals, 65 first-check passes,
a trial-level macro mean of `0.345380952381`, and a descriptive pooled micro
fraction of `0.333333333333`.

Figure regeneration writes:

- `paper/fig2_ariac.pdf`
- `paper/fig3_tv.pdf`
- `paper/fig4_rotor.pdf`
- `paper/fig5_coverage_scaling.pdf`
- `paper/fig6_scope_intervention.pdf` (Supplementary Information)
- `paper/figure_scripts/figure_data_audit.json`

The representative overview composites used in panel a of Figs. 2--4 are
provided as publication assets; every quantitative panel is rebuilt from the
released tables.

## Repository layout

- `results/`: curated trial-level CSV files and parameter tables.
- `paper/figure_scripts/`: deterministic figure and statistical-analysis code.
- `ariac/`: ROS 2/ARIAC verifier, repair controller, PDDL, and simulator interface.
- `pvep/ibpomcp/`: shared online belief-space planning implementation.
- `pvep/kinova/`: fastening-domain model and controller code.
- `pvep/sleeve/`: thermal interference-fitting model and controller code.
- `docs/DATA_AND_CODE.md`: claim-to-file, configuration, and figure mapping.
- `docs/PROTOCOL_AND_MODEL_PROVENANCE.md`: ARIAC protocol and model-provenance
  ledger, including fields that were not retained in legacy aggregates.
- `scripts/verify_release.py`: aggregate, file-integrity, and release-hygiene checks.
- `MANIFEST.sha256`: cryptographic hashes of released non-generated files.

## Scope and access restrictions

The tables under `results/` are the curated analysis inputs used by the
released scripts. Absolute workstation paths in lineage-only fields have been
replaced with repository-relative paths or `internal://` identifiers; no
numerical result fields were changed by this de-identification.

Raw fixture imagery, camera poses, robot-calibration files, and model weights
are excluded because they are subject to industrial data-sharing restrictions.
The release instead provides de-identified result tables, observation-model
parameters, POMDP configurations, representative publication composites, and
the code needed to reproduce the reported quantitative analyses. Editors or
reviewers may request confidential access from the corresponding author;
access requires institutional and data-owner approval, an executed data-use
agreement, and no public redistribution, and therefore cannot be guaranteed
for every request.

The ARIAC runtime requires an external ROS 2/ARIAC installation. Trial-level
analyses and submitted-figure regeneration require only `requirements.txt`.
Foundation-model calls additionally require explicit provider endpoint and
model values; copy `.env.example` and set `PVEP_OPENAI_BASE_URL`,
`PVEP_OPENAI_MODEL`, and the API key for each new run. The thermal LLM
proposal runner accepts the analogous `SLEEVE_LLM_*` variables. No
third-party endpoint or model alias is selected by default.

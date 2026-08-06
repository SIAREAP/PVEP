# PVEP anonymous reproduction repository

This repository accompanies a double-blind manuscript on
verifier-enforced planning under uncertainty. It contains the released
trial-level result tables, deterministic analysis and figure-regeneration
scripts, and de-identified reference implementations of the core planners,
verifiers, and domain models.

## Reproduce the reported aggregates and figures

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python paper/figure_scripts/recompute_ariac_q0.py
python paper/figure_scripts/generate_all.py
python scripts/verify_release.py
```

The Q0 command must report 50 trials, 195 subgoals, 65 first-check passes,
a trial-level macro mean of `0.345380952381`, and a descriptive pooled micro
fraction of `0.333333333333`.

Figure regeneration writes:

- `paper/fig2_cross_domain_decoupling.pdf`
- `paper/fig3_mechanism.pdf`
- `paper/fig4_necessity.pdf`
- `paper/fig5_scope2x2.pdf`
- `paper/figure_scripts/figure_data_audit.json`

## Repository layout

- `results/`: curated trial-level CSV files and parameter tables.
- `paper/figure_scripts/`: deterministic figure and statistical-analysis code.
- `ariac/`: ROS 2/ARIAC verifier, repair controller, PDDL, and simulator interface.
- `pvep/ibpomcp/`: shared online belief-space planning implementation.
- `pvep/kinova/`: de-identified fastening-domain model and controller code.
- `pvep/sleeve/`: de-identified thermal interference-fitting model and controller code.
- `docs/DATA_AND_CODE.md`: claim-to-file and figure-to-file mapping.
- `scripts/verify_release.py`: aggregate, file-integrity, and anonymity checks.
- `MANIFEST.sha256`: cryptographic hashes of the released code and data
  inputs (regenerated PDFs and aggregate JSON are intentionally excluded).

## Scope of the release

The tables under `results/` are the curated analysis inputs used by the
released scripts. Absolute workstation paths in lineage-only fields have
been replaced with repository-relative paths or `internal://` identifiers;
no numerical result fields were changed by de-identification.

Raw fixture imagery, camera poses, robot calibration files, and model
weights are excluded because they are subject to industrial sharing
restrictions. The release instead provides de-identified result tables,
observation-model parameters, POMDP configurations, and the code needed to
reproduce the reported analyses without those restricted assets.

The ARIAC runtime requires an external ROS 2/ARIAC installation. The
trial-level analyses and all paper figures require only `requirements.txt`.

## Double-blind policy

Please do not attempt to identify the authors during review. Issues and pull
requests should concern reproducibility only. Author and archival citation
metadata will be added after the review period.

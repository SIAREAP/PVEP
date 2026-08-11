from __future__ import annotations

import importlib

import audit_figure_data
import gen_fig5_coverage_scaling
import gen_fig6_scope_intervention


CASE_STUDY_MODULES = (
    "gen_fig2_ariac",
    "gen_fig3_tv",
    "gen_fig4_rotor",
)


def main() -> None:
    audit_figure_data.main()
    for module_name in CASE_STUDY_MODULES:
        importlib.import_module(module_name)
    gen_fig5_coverage_scaling.main()
    gen_fig6_scope_intervention.main()


if __name__ == "__main__":
    main()

from __future__ import annotations

import audit_figure_data
import gen_fig2_cross_domain_decoupling
import gen_fig3_mechanism
import gen_fig4_necessity
import gen_fig5_scope_intervention


def main() -> None:
    audit_figure_data.main()
    gen_fig2_cross_domain_decoupling.main()
    gen_fig3_mechanism.main()
    gen_fig4_necessity.main()
    gen_fig5_scope_intervention.main()


if __name__ == "__main__":
    main()

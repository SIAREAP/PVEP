# ARIAC protocol and model-provenance ledger

This ledger separates ARIAC evaluations that reuse the same 50 scenario
identities but are not repeated estimates of one protocol. It records only
fields supported by the archived tables and run identifiers. `NR` means that a
field was not retained; no provider, endpoint, cache state, or backend role is
inferred from a runtime default.

## Evaluation protocols

| Evaluation | Proposal source | Fresh or frozen | In-loop repair backend | Time budget | Crash and rerun rule | Reported metric |
|---|---|---|---|---:|---|---|
| Fig. 2d/e nominal component-role ablation | Row-specific natural-language or PDDL foundation-model candidate in `results/ariac/nominal_ablation_per_scenario.csv` | NR in the legacy aggregate | Row-specific: none, generic re-proposal, or full PVEP; exact model identifier NR | Target 900 s/scenario; realised per-row budget NR | Attempt-level status is not retained in this aggregate | Strict full-score completion |
| Controlled GPT-5.2 corruption sweep | Nominal GPT-5.2 candidate followed by the stated synthetic corruption and paired seed | Paired proposal/corruption records; cache flag NR | `gpt-5.2-2025-12-11`, with the stated feedback variant | 900 s/scenario | A timeout or in-episode simulator crash in the designated attempt is a failure | Scenario-normalized score, strict/any-score completion, first-check admissibility |
| Controlled Qwen proposal-source check | `qwen3-max` nominal candidate with no synthetic corruption in `results/ariac/additional_proposal_experiments.csv` | NR | Full-loop configuration is indicated by retained run labels; call-level role mapping NR | Target 900 s/scenario; complete attempt lineage NR in the displayed aggregate | NR where attempt lineage is absent | Any-score completion and mean score |
| Separate GLM robustness evaluation | `glm-4.7` nominal candidate in a separately archived robustness series | NR | Full-loop configuration indicated; call-level role mapping NR | NR in the released robustness aggregate | NR in the released robustness aggregate | Any-score completion |
| Natural-proposal audit | Backend-specific nominal proposal with no synthetic corruption | Regeneration/cache state is not retained uniformly across the three legacy series | `V_full` is retained in the timeout-status table; call-level role mapping NR | Target 900 s/scenario; realised GPT attempt budget is NR for some rows | A recorded terminal timeout or in-episode simulator crash in the designated attempt is a failure | Any-score completion and failure composition |

The component-role table and natural-proposal audit therefore must not be
compared as if one replicated the other. They differ at least in archived run
series, endpoint metric, retained configuration labels, and failure accounting,
while some requested protocol fields are unavailable. In particular, the
controlled Qwen check and separate GLM robustness result are not treated as the
natural-proposal Qwen and GLM rates.

For run adjudication, a launch or node failure before a valid episode begins is
rerun and does not define a trial. Where a legacy trial originally used a
shorter limit, one designated 900-second protocol-repair attempt replaces that
shorter attempt. A timeout or simulator crash during the designated episode is
retained as failure. Later diagnostic reruns are flagged and do not replace the
designated outcome. `results/ariac/timeout_900_rerun_status.csv` records the
available timeout and protocol-repair lineage.

## Foundation-model provenance

| Evaluation record | Exact model identifier | Archived run/access evidence | Provider | Base endpoint | Proxy status | Decoding |
|---|---|---|---|---|---|---|
| Controlled GPT corruption | `gpt-5.2-2025-12-11` | Run identifiers dated 2026-07-18 to 2026-07-20 | NR | NR | NR | temperature 0; provider-default top-p; max 4096 tokens; retry budget 3 |
| Natural GPT audit | `gpt-5.2-2025-12-11` | Access date NR in the legacy aggregate | NR | NR | NR | temperature 0; provider-default top-p; max 4096 tokens; retry budget 3 |
| Qwen archived comparisons | `qwen3-max` | Run identifier dated 2026-07-05 | NR | NR | NR | temperature 0; provider-default top-p; max 4096 tokens; retry budget 3 |
| GLM archived comparisons | `glm-4.7` | Run identifier dated 2026-06-21 | NR | NR | NR | temperature 0; provider-default top-p; max 4096 tokens; retry budget 3 |

The code uses OpenAI-compatible clients where applicable, but that client
choice does not identify the actual provider or gateway. Prior hard-coded
third-party endpoints and model aliases have been removed. New runs require
explicit `PVEP_OPENAI_BASE_URL`/`PVEP_OPENAI_MODEL` or `SLEEVE_LLM_*` values,
and those values should be stored with the run output. Temperature-zero
settings are not called deterministic because provider seeds and backend
fingerprints were not fixed uniformly.

The `NR` provider, endpoint, access-date, and proxy fields cannot be recovered
from the released aggregate tables alone. If separately approved billing,
gateway, or run-management records later become available, this ledger can be
amended; the current release does not infer those fields from historical code
defaults.

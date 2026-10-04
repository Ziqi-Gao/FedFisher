# Archived V2 failed-shard completion

This disabled additive proposal resumes the user's unchanged model/signal
experiment after repository restructuring. It does not establish central
registration, admission or scientific completion. Keep all existing task names,
requests, run manifests, results and archive files unchanged.

The fixed historical anchor is
`/data/del6500/FedFisher/archives/model-signal-recovery-v2-4ecb6d5-20261002/closeout.json`,
SHA-256 `1cc83b232fdfc8fc68c772215d5fa4827a6c37ec40aaf523dcdfb968829a6a46`.
It releases the old source for restructuring while explicitly leaving the
experiment incomplete. New-source pilot evidence supplies the new admission
gate; the historical closeout alone never authorizes formal execution.

## Fixed scope

Both new tasks accept exactly `setting`, `cell`, `variant`, `shard_index` and
`shard_count`. No caller may choose seeds, paths, epochs, patience, signal
strength, model size, hardware or runtime. The only setting is `weak_signal_05`;
the only recipes are shallow MLP, E=1, fixed total 30 epochs or validation
patience. All scientific configuration, data, RNG, fitting, evaluation and
noncompletion rules are inherited unchanged from V2.

| Stage | Task | Coverage |
| --- | --- | --- |
| Pilot | `effect-modifier-model-signal-completion-pilot` | Both recipes, all four variants, seed 0, IID and non-IID: 8 requests / 16 units |
| Formal | `effect-modifier-model-signal-completion-shard` | Exactly the archived 24 failed whole shards / 896 units |

Formal keys are fixed in both Python validation and the JSON schema:

- `fedavg_shallow_e1_budget30`, `control`: shards 0 through 15 of 16.
- `fedavg_shallow_e1_budget30`, `site_gaussian_shift`: shard 0 of 16.
- `fedavg_shallow_e1_patience`, `site_gaussian_local_permutation`: shards 9
  through 15 of 16.

Every formal shard owns seeds `shard_index, shard_index+16, ... < 300` and both
splits. The 287 saved complete units within those failed shards remain audit-only;
all 896 units are rerun in independent `_v3_300seeds` namespaces. Pilot outputs
use `_v3_pilot` and never count toward formal coverage. Request IDs start with
`fedfisher-ms-v3-`; a generation changes request IDs, not scientific namespaces.

The retained formal evidence contains 171,898 performance-complete units and
six verified training-cap outcomes. Those six remain scientific noncompletions;
they are neither rerun by this plan nor promoted to successful performance.
The original 200-round cap, patience 10 and improvement threshold `1e-4` stay
fixed. Unknown failures and invalid/nonfinite artifacts continue to fail closed.
Replacement outcomes must retain the same distinction. Whole-cohort terminal
coverage and complete scientific success are different claims.

## Admission, publication and execution

The adapter uses its existing strict protocol-v2 validation, canonical source
guard, centrally assigned affinity and foreground execution. The proposed
`cpu-1core` / 8192-MiB / zero-GPU envelope and runtime fallbacks are provisional
operator review inputs. Actual pilot measurements and archive/pilot-verification
startup cost inform profile review; startup verification is outside fitting
timers. No project code selects resources or changes central state.

`completion_requests.py` constructs pure plans. The compatible command
`scripts/prepare_model_signal_completion_requests.py` delegates source review,
archive checks, fresh pilot admission and immutable publication to
`scheduling/publication/model_signal_completion.py`. Bundles live below
`/scr/del6500/FedFisher/scheduler/outbox/fedfisher-ms-v3-<cohort>-cohort-p<N>/`;
the corresponding inventory is published last under the project scheduler
reports directory. These files are operator-review deliveries, not submissions.

Freeze clean source before publishing the pilot. Only the operator may merge
the disabled proposal, validate and submit its exact eight requests. Pilot
acceptance requires all 16 successful scientific units, actual producing
attempts, paired invariants and exact numerical parity with archived V2 pilot
results, excluding observational timing/memory. The new source-bound proof is
required both before formal publication and before formal runtime initializes
outputs. Old V2 pilot acceptance cannot open the V3 formal gate.

Final acceptance must verify all 24 new shard summaries, immutable terminal
artifacts and actual producers, exact replacement/retention coverage, and paired
data/training/evaluation invariants against historical evidence. Preserve each
unit's actual source and producer. Partial files, an exit code, or a valid model
from a failed old shard alone cannot establish accepted replacement coverage.

Source examples and running fixtures are review inputs only. Development uses
bounded single-thread fixtures below FedFisher scratch; it does not run a real
pilot. Keep canonical source unchanged through the admitted pilot and formal
execution and acceptance.

# One-shot LASSO Agent Contract

This contract is additive to the repository and parent effect-modifier
protocols.  It may not weaken either parent contract.

- All new one-shot computation must use the registered
  `effect-modifier-one-shot-lasso-shard` task through the foreground FedFisher
  ServerScheduler adapter. A job receives one physical CPU core, fixed 8192 MiB
  host memory, one compute thread, one reviewed variant, and exactly one of
  8 or 16 deterministic seed shards per variant.
- ServerScheduler owns CPU affinity, admission, logs, and global concurrency.
  The project must not create Screen sessions or acquire the legacy one-shot
  CPU locks for a centrally managed shard. A shard processes only seeds
  congruent to `shard_index` modulo `shard_count`.
- Delivery is at least once. Process exit is not completion evidence. Continue
  to require the manifest, current provenance, semantic result identity, every
  artifact hash, and the cell done marker before skipping work. Partial or
  invalid cells fail closed without overwrite.
- Pre-cutover one-shot outputs are retained for audit but are non-authoritative
  and may not be resumed. New output uses the fixed
  `stationarity_4000iter_full_scope_server_scheduler_v2r2_300seeds` suffix.

- Preserve the multi-round `dslassocov_*` source and durable results as a
  reference.  Never delete, rename, or overwrite them.
- One-shot families use only the disjoint
  `one_shot_local_logistic_lasso_fedavg_<variant>_` and
  `one_shot_mlp_group_lasso_fedavg_<variant>_` RUN_ID namespaces.
- Every site's training, standardization, and lambda selection must finish
  locally before its one upload.  A training payload contains only `site_idx`, `n_k`, and the final
  `(2 + 2 * covariate_dim)` raw-scale logistic coefficients or complete MLP
  model vector.  Byte accounting includes both int64 metadata fields.
- Never upload gradients, Hessians, Fisher information, individual rows, or a
  site callback.  Never run global CV.  Seal training-site access at upload and
  reject every post-upload attempt to read training rows.
- Server aggregation is the strict `n_k/N` arithmetic mean.  MLP aggregation
  must reuse the existing `run_one_shot_algs.get_fedavg_model` implementation.
  Record training communication as exactly one round and record evaluation
  communication separately.
- Logistic L1 penalizes only the `covariate_dim` `A*X` coefficients.  Its site-local
  epsilon=1/(n_k+1) soft-target separation correction must be manifest-bound
  and audited; it may not be represented as a coefficient penalty.  A legacy
  FISTA false flag may be polished only when its final prox-gradient and KKT
  residuals already pass the unchanged threshold: reset momentum, retain the
  same lambda/tolerance, allow at most 50 deterministic iterations, and audit
  pre/post residuals plus coefficient drift. This rule applies to both
  unpenalized starts and lambda-path fits. Genuinely nonstationary fits must
  still fail closed, and CV must never fall back to a different lambda when the
  originally selected fit fails. Rejections emit canonical, versioned JSON to
  stderr with cell, scheduler attempt, optimizer, and pre/post polish context;
  they must not create a result artifact or completion marker. Formal runs fix
  the primary local-logistic optimizer limit at 4000 iterations. This larger
  limit does not change the tolerance, stationarity threshold, lambda grid, or
  selection rule; every selected fold and final fit must still pass both
  original residual checks. The deterministic seed-285 Gaussian non-IID
  site-8 regression must retain at least one selected fit above 2000 iterations
  and pass without momentum-reset polish. MLP
  explicit-interaction group regularization penalizes only the
  `covariate_dim` first-layer `A*X` input columns, with one L2 group over
  all first hidden units per column.  It must not be described as identifiable
  effect-modifier selection because unpenalized A/X nonlinear paths remain.
  The independent matched zero-penalty CPU reference must remain exactly equal
  to the lambda-zero group path.  Per-weight L1 must be labeled MLP-L1.
- Reuse the exact four-cell SyntheticEffectModifier data and evaluation
  contracts.  Control/local and Gaussian/combined must have identical paired
  training, test, payload, and aggregate-model fingerprints.
- Durable results remain under the standard `/data` output root; scratch and
  runtime files remain under the standard `/scr` tree.  Reject symlinks and
  existing cell artifacts.  The formal CPU scheduler may skip only a cell
  whose manifest, current source/config, done marker, semantic identity, and
  every artifact hash validate; any partial or invalid cell fails closed and
  is never overwritten.
- Legacy formal 300-seed execution was CPU-only and used a local total of
  exactly 32 or 64 workers, split evenly as 8 or 16 deterministic seed shards
  per variant. Every worker used one compute thread. Its locks live only under
  `/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks/` and
  are disjoint from GPU and dsLassoCov lock namespaces.  Scheduler operations
  may inspect and control only the short `ff_os2_<variant>_<shard>of<count>`
  sessions, their own worker processes, and an exact worker's active child
  process during signal cleanup.  Locks are scoped to the exact
  variant and shard, while deterministic disjoint seed assignment prevents
  cell collisions.  They must never wipe, broadly kill, renice, reschedule, or
  otherwise interfere with J-lens, dsLassoCov, an ablation, or any foreign
  experiment. These Screen and lock rules are retained for legacy audit only;
  they do not authorize new local launches.
- Do not add or bypass GPU scheduling scripts.  Do not start 300-seed work
  until no-cache syntax, all old and new tests, and a same-source four-cell
  seed-0 CPU smoke pass and are reviewed.

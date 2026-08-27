# One-shot LASSO Agent Contract

This contract is additive to the repository and parent effect-modifier
protocols.  It may not weaken either parent contract.

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
  and audited; it may not be represented as a coefficient penalty.  MLP explicit-interaction group regularization penalizes only the
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
- Formal 300-seed execution is CPU-only and uses exactly four variant workers,
  each with one compute thread.  Its locks live only under
  `/scr/del6500/FedFisher/effect_modifier_experiment/one_shot_cpu_locks/` and
  are disjoint from GPU and dsLassoCov lock namespaces.  Scheduler operations
  may inspect and control only the four short `ff_os_*300` sessions and their
  own worker processes; they must never wipe, kill, renice, reschedule, or
  otherwise interfere with J-lens, dsLassoCov, an ablation, or any foreign
  experiment.
- Do not add or bypass GPU scheduling scripts.  Do not start 300-seed work
  until no-cache syntax, all old and new tests, and a same-source four-cell
  seed-0 CPU smoke pass and are reviewed.

# Effect-modifier comparison contract

This additive experiment inherits the root and parent protocols. Read their
AGENTS.md files and the Codex context/runbook before work here.

- Preserve the existing SyntheticEffectModifier generator, four variants,
  site capacities, independent test set, and paired data/fit invariants.
  Do not add a standard X-learner: outcome-model methods continue to use
  consistent treatment/control prediction differences and raw-X interventions.
- Homogeneous R methods share effect coefficients across sites but retain
  site-local nuisance models. They estimate probability-scale effects, not
  logits. On the existing binary/label-partition DGP the linear common-effect
  model is a working model, not a claim that all paper assumptions hold.
- Compare all methods on probability-scale treatment-contrast intervention;
  preserve the original logit-scale endpoint for outcome models separately.
  Never fabricate logits or classification accuracy for an effect-only model.
- Resolve cutoff ties by reporting expected Top-k recovery under uniform
  selection among tied features, plus the probability of selecting all true
  signals. Keep legacy index-based tie-breaking only as a labeled diagnostic.
- Real multiround FedAvg must broadcast the aggregate into the next round.
  Epochs, minibatch updates, optimizer iterations and communication rounds are
  distinct. Stop rules, local epochs, selection/refit costs and safety limits
  are manifest-bound. Reaching a safety limit is not convergence.
- Keep the existing dsLassoCov objective and multiround gradient algorithm.
  It is not an X-learner. Preserve old run IDs and artifacts; comparison
  results belong to a new comparison-specific namespace.
- Unpenalized and adaptive-LASSO shared-effect R methods differ only in the
  final effect penalty. Nuisance fitting uses training-only cross-fitting and
  inner selection. Upload only site summary statistics to the effect server.
  Binary constant-response handling, numerical thresholds, adaptive weights,
  selection criteria and all deviations from the reference must be recorded.
- Every method records full fitting/selection and evaluation wall/CPU time;
  retain stage details and logical communication counts/bytes. Do not equate
  single-process runtime with a deployed parallel/network runtime. GPU and CPU
  timing are not directly comparable. Never manufacture missing old timings.
- New scientific execution requires separate reviewed task contracts, disabled
  project-side registration proposals, fixtures, and operator-admitted pilots.
  Existing task names must not be repurposed to launch these experiments.
  No direct scientific CLI, background worker, resource query or central write.
- Use the sole project scheduler adapter, strict protocol-v2 validation,
  deterministic seed shards and allocation-provided one-thread CPU execution.
  Pilot/formal parameters express science only, never resources or paths.
- Require a format-v2 run manifest, current-source provenance, canonical
  configuration/identity hashes, artifact digests and verified cell markers.
  Missing may run; partial/invalid fails closed; validated may skip. Reject
  symlinks and overwrites. Failed attempts must never become success markers.
- Durable outputs: /data/del6500/FedFisher/effect_modifier_experiment/outputs/.
  Tests, caches and temporary output: /scr/del6500/FedFisher/ only.
  Bounded numerical/contract tests are development validation, not permission
  for a pilot or a formal experiment. Record changing evidence only in
  docs/codex/SCHEDULER_HANDOFF.md.

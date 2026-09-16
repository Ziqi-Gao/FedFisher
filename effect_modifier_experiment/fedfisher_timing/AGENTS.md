# Standalone FedFisher timing protocol

Read the root/parent policies and Codex context/runbook first. This additive
CPU recipe reruns historical diagonal/KFAC Fisher methods with real timing.

- Preserve old scientific directories and old missing-time records. A new time
  belongs only to its new fitted model, seed, data/configuration and execution.
- Keep the original data generator, SGD recipe, quantization, Fisher weighting,
  server optimizer and validation-accuracy selection. The original global
  training row order fixes validation sampling. Selection overlaps local
  training, as in the historical method; never relabel it held-out validation.
- Standalone execution computes the selected Fisher only. Isolate its random
  draws from later local SGD so diag/KFAC use matching local fits. This is a
  new explicit RNG recipe, not bitwise reproduction of the historical GPU path.
- Measure serial single-core CPU wall/process time with the common MethodTimer.
  Include every fit, compression and validation-selection cost. Components are
  nested detail and must not be added to their containing stages a second time.
  Never use a component sum or queue duration as measured standalone runtime.
- Use the common comparison evaluator for probability and logit raw-X scores,
  five paired permutations and explicit tie handling. Do not overwrite or mix
  old endpoint observations. Methods receive no test labels during fitting.
- Fixed science-only task parameters, source-bound immutable artifacts and
  current-source/semantic/hash/done checks are mandatory. Partial output fails
  closed. The sole scheduler adapter remains the only execution entrypoint.
- New tasks require disabled proposals, fixtures and operator registration.
  Publish only the 16 seed-zero pilot requests initially; the 9,600-cell formal
  cohort requires accepted pilot results, paired invariants and resource review.
- Code belongs here; durable observations under the project data root; test
  caches and outbox below scratch. Keep source frozen throughout admitted runs.
  Add changing evidence only to the handoff before freeze or scratch afterward.

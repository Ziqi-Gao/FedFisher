# One-shot effect methods and common-protocol experiments

Inherit the root and parent operating protocols. This additive subtree
implements the user's FedFisher-EM / DR-Sieve specification and an independent
SCMV2 dataset. The parent's legacy DGP defaults remain unchanged.

- Keep data generation, observed-data types, methods, evaluation, result
  validation and scheduler composition separate. Reuse established model and
  numerical components without changing historical method definitions.
- Methods receive observed rows only. True propensity, potential-outcome
  means, CATE, signal masks and variable roles belong to evaluator data.
- All new-protocol methods share the same outcome-independent 70/15/15 T/V/H
  indices, raw covariate coordinates, target site weights and perturbations.
  Historical results with different training populations are not comparable.
- Servers accept serialized summary payloads, never hospital callbacks, raw
  validation data, training rows, or evaluator truth. Keep any legacy
  validation-selected Fisher diagnostic exclusively in simulation evaluation.
- Distinguish external mean-absolute perturbation scores from H-based half
  squared perturbation scores. Every approximate score uses the same H rows,
  perturbations and target weights as its exact reference.
- Diagonal strict Fisher aggregation and score-metric damping have independent
  contracts. Do not silently alter the aggregation objective with damping.
- DR nuisance fitting uses only T, with training-only nested cross-fitting;
  V selects the effect penalty from uploaded risk summaries. H scores the
  selected T model. Do not clip its predictions while claiming exact scores.
- Count serialized basis directions, matrices, metadata and model vectors in
  actual upload bytes. Protocol/model broadcasts are separate. Local/server
  compute timing does not measure deployed network latency.
- Tests may use small numerical fixtures with one CPU thread and scratch
  caches. They do not authorize scientific pilots or full seed cohorts.
- New task names require disabled proposals, strict schemas, fixtures and
  operator-admitted staged pilots. Do not launch through historical task names.
  Compression pilots require successful full-gradient linearization evidence.
- Keep provenance-bound result namespaces separate from causal_comparison v1
  and all historical namespaces. Reject partial output and overwrite.
- Null scenarios report no recovery AUROC/Top-k, and no method claims causal
  validity, FDR control or novelty merely by passing software validation.

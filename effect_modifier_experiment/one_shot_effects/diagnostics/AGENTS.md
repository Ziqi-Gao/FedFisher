# Checkpoint replay diagnostic protocol

Inherit the root, experiment and one_shot_effects agent protocols. This subtree
implements the user's bounded diagnostic plan, separately from method fitting.

- The first cohort is exactly four Fisher replays and one SCMV2 DR-path replay,
  all seed 0. No training, nuisance refit, penalty reselection, compression or
  additional seeds are part of these recipes.
- Bind the accepted original pilot's manifest, result, checkpoint and payload
  hashes separately from the current diagnostic source provenance. Never relax
  the original artifact store's current-source validation to permit replay.
- Regenerate from the frozen original configuration and check all T/V/H, test
  and original five-repeat permutation fingerprints before diagnostic inference.
  An input mismatch fails closed; original artifacts remain read-only.
- Fixed H subsamples and random directions must not depend on predictions,
  variable roles or outcomes. Reuse local derivatives across the parameter path.
- Audit the actual original pilot function through
  `fisher_summary.tau_and_directional_derivatives` on the same fixed rows and
  normalized directions. Compare its tau and derivatives with the independent
  implementation and stable central differences; source replay equality alone
  cannot certify the original implementation.
- Require stable finite-difference coverage separately for control log-odds,
  treated log-odds and CATE in each required direction. Missing coverage is
  `inconclusive`, not a pass. A zero actual aggregation direction is explicitly
  inapplicable; a zero random direction is inconclusive. Zero directions never
  contribute usable checks. Persist and semantically revalidate these decisions.
- Separate binary log-odds, probability effects, direct CATE Taylor expansion
  and sigmoid-of-linear-logit diagnostics. The latter is not a replacement for
  the deployed quadratic summary contract.
- Name prediction averaging and parameter averaging separately. SCMV2 truth is
  evaluator-only; legacy has no CATE truth under this diagnostic contract.
- Keep the original DR validation selection and report the entire saved path.
  Test curves cannot select lambda or epochs, and nonzero groups are not
  significant discoveries or evidence of FDR control.
- At most four later training diagnostics are conditional on replay findings,
  need separately reviewed recipes and scheduler admission, and are not created
  or implicitly authorized for execution by a replay result.
- Preserve the two .25 full-H approximation gates. Failed or inconclusive
  gradient checks stop progression to method experiments. Numerical gates do
  not independently establish acceptable CATE performance or scientific novelty.
- No direct scientific CLI. Tiny one-thread numerical fixtures are permitted;
  real replay inference and data generation require the registered foreground
  adapter. Reports may read completed results without fitting or inference.

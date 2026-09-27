# Model-size and signal-strength ablation admission

This additive, disabled proposal covers two binary experiments. `small_mlp`
uses a single hidden layer of eight neurons at the original signal strength
2.0. `weak_signal_1` and `weak_signal_05` change the effect-modifier interaction
strength to 1.0 and 0.5, respectively, and cover small, shallow and deep MLPs.
Each architecture retains eight FedAvg recipes (E=1/5/10/30, each with patience
or a total 30-epoch budget) and the original Diag/KFAC FedFisher methods.
Each weak-signal setting also includes one iterative logistic dsLasso cell;
dsLasso has no MLP architecture. See the experiment package README for the
fixed scientific definitions. Existing task names and namespaces are unchanged.

The sole foreground scheduler adapter accepts exactly `setting`, `cell`,
`variant`, `shard_index`, and `shard_count`. Schemas and runtime validation reject
unreviewed settings, invalid setting/cell pairs, free hyperparameters, paths,
hardware hints and invalid shards. Requests omit resources and execution profile.
Resource selection, admission and submission belong to ServerScheduler.

The pilot has 72 setting/method combinations across four original variants:
288 requests, each processing seed 0 with IID and non-IID splits, for 576 units.
The pilot schema fixes shard index/count to 0/1. The complete planned formal
cohort covers seeds 0–299 and both splits, for 172,800 units in 2,304 requests
with eight shards or 4,608 requests with sixteen shards. The setting and full
scientific configuration are manifest-bound; shard choice cannot change a
scientific run identity. `ablation-plan.json` records the complete coverage.

The proposed one-core, 8192-MiB CPU profile and 86400/604800-second fallback
runtime values are unmeasured review inputs. Development fixtures do not prove
full-size runtime or memory suitability. Review/register and admit the pilot
first. Formal publication and execution require actual same-source pilot
acceptance, complete scientific artifacts, paired invariants, measured costs,
and operator profile/admission review. Process exit zero alone is insufficient.

The two source examples illustrate request syntax only. They are not an outbox
bundle, central registration or submitted jobs. This code-delivery phase does
not publish requests or execute experiments. Any later request publication must
use the project scratch outbox after a clean canonical source freeze, with
immutable inventory and provenance. Keep that source unchanged through admitted
execution and acceptance; preserve historical outputs and failed attempts.

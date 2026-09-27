# Model/signal recovery v2

This disabled additive proposal preserves the v1 numerical recipes, including
the 200-round FedAvg patience limit. The new execution contract distinguishes
scientific success from a strictly validated maximum-round noncompletion. A
noncompletion retains its diagnostic model and full training audit, has no
formal intervention score, and does not stop subsequent units. Unknown errors
still fail the request. Existing v1 tasks and artifacts remain unchanged.

The fixed source closeout is recorded in `recovery_protocol.SOURCE_CLOSEOUT`.
Only the 129 fully completed, exit-zero v1 shards (4,838 units) are eligible for
retention. The two scientific failures, the exit-unknown shard, and the 4,476
unadmitted requests receive new identities and complete replacement shards:
4,479 requests, 167,962 units. Their union with retained results covers all
172,800 formal units. Old partial successes are preserved for audit only.

The new pilot contains 288 seed-zero baseline requests plus two cap diagnostics
for `small_mlp/fedavg_small_e1_patience` under the Gaussian and combined variants.
Each diagnostic processes exactly seeds 9, 25, and 41 in order, for both splits.
This totals 290 requests and 588 units. Acceptance checks the two known seed-25
non-IID caps, continuation after them, all six known nonzero-seed successes,
and parity with prior successful numerical results. A zero process exit alone
does not establish scientific completion.

Project publication uses `scripts/prepare_model_signal_recovery_requests.py`.
It requires clean finalized source and revalidates the fixed closeout before
and after publication. Formal publication additionally requires actual
same-source recovery pilot acceptance. Bundles are immutable nested directories
under `/scr/del6500/FedFisher/scheduler/outbox/`; inventories are published last
under `/scr/del6500/FedFisher/scheduler/reports/`. Requests omit all hardware
hints. An outbox bundle is not a submitted job.

The operator must add and enable the two new task contracts, review their
provisional CPU profile and runtime estimates, and admit the pilot first.
Formal execution follows actual pilot acceptance and exact replacement coverage
review. Keep canonical source frozen from pilot admission through formal
completion. Do not reset the old controller, retry old failed IDs, overwrite
old outputs, or interpret noncompletion as convergence.

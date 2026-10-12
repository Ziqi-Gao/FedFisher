# Scheduler request fixtures

These protocol-v2 request examples are grouped by scientific recipe. The
`original-lasso/` and `comparison/` directories cover the original CPU contracts;
versioned directories retain each later recipe's examples. `survival-v1/formal/`
keeps formal examples separate from that recipe's pilot examples. See the
[scheduler contract index](../../../effect_modifier_experiment/server_scheduler/README.md)
for the matching schemas, disabled proposals and fixed plans.

Request examples describe scientific parameters; they are not published outbox
deliveries or submitted jobs. The separate [running-manifest fixtures](../server_scheduler/)
model the scheduler allocation and attempt envelope used by adapter tests.
Neither kind of fixture establishes live admission or scientific completion.

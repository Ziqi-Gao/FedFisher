# Shared scientific mechanisms

| Module | Responsibility |
| --- | --- |
| `measurement.py` | Existing method-stage timing and communication/payload observations |
| `fingerprints.py` | Existing binary array and model fingerprint formats |
| `recovery_metrics.py` | Existing tie-aware recovery ranking arithmetic |

These modules preserve their existing calculations. Their callers retain the
experiment-specific contrast scale, truth definition, stopping rule, scientific
recipe and result namespace. Measurements observe computation rather than
selecting resources or managing output paths. Compatibility imports remain at
the former comparison-module locations.

Artifact/provenance/publication mechanisms belong to
[infrastructure](../infrastructure/README.md). Experiment composition remains
in the owning experiment. See [architecture](../../docs/architecture.md).

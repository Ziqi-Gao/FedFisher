"""Publication must authenticate its source before inspecting or writing work."""
from contextlib import ExitStack
from importlib import import_module
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.scheduling.publication import require_canonical_source
from tests.paths import PROJECT_ROOT


class RequestPublicationSourceTests(unittest.TestCase):
    def test_foreign_publication_code_fails_before_provenance_or_git_lookup(self):
        scratch = Path("/scr/del6500/FedFisher/repository-refactor-20261002/publication-source-fixtures")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            foreign = Path(directory) / "foreign.py"
            foreign.write_text("# unrelated source must never publish canonical provenance\n")
            for name in ("original", "comparison", "effects_replay", "fedfisher_timing",
                         "model_signal", "model_signal_recovery", "survival", "survival_recovery"):
                publisher = import_module("effect_modifier_experiment.scheduling.publication." + name)
                entry = "git_snapshot" if name == "original" else (
                    "current_provenance" if name in ("survival", "survival_recovery") else "_provenance")
                provider = "_git_text" if name == "original" else "compute_provenance"
                with self.subTest(name=name), ExitStack() as stack:
                    stack.enter_context(mock.patch.object(publisher, "__file__", str(foreign)))
                    observed = stack.enter_context(mock.patch.object(publisher, provider))
                    with self.assertRaisesRegex(ValueError, "canonical registered source"):
                        getattr(publisher, entry)()
                    observed.assert_not_called()

    def test_checkout_source_validation_is_explicit_and_preserves_symlink_rejection(self):
        publisher = import_module("effect_modifier_experiment.scheduling.publication.comparison")
        require_canonical_source(publisher.__file__, PROJECT_ROOT)
        scratch = Path("/scr/del6500/FedFisher/repository-refactor-20261002/publication-source-fixtures")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            alias = Path(directory) / Path(publisher.__file__).name
            alias.symlink_to(publisher.__file__)
            with self.assertRaisesRegex(ValueError, "canonical registered source"):
                require_canonical_source(alias, PROJECT_ROOT)

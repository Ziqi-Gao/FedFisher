"""Golden scientific request identity/coverage captured before the layout change."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from tests.paths import PROJECT_ROOT

FIXTURE = PROJECT_ROOT / 'tests/fixtures/repository_request_parity.json'


class RequestParityTests(unittest.TestCase):
    def test_every_reviewed_request_recipe_preserves_its_original_bytes(self):
        fixture = json.loads(FIXTURE.read_text())
        modules = {'original': 'effect_modifier_experiment.requests', 'comparison': 'effect_modifier_experiment.causal_comparison.requests', 'effects_replay': 'effect_modifier_experiment.one_shot_effects.diagnostics.requests', 'fedfisher_timing': 'effect_modifier_experiment.fedfisher_timing.requests', 'model_signal': 'effect_modifier_experiment.model_signal_ablation.requests', 'model_signal_recovery': 'effect_modifier_experiment.model_signal_ablation.recovery_requests', 'survival': 'effect_modifier_experiment.survival.requests', 'survival_recovery': 'effect_modifier_experiment.survival.recovery_requests'}
        old = importlib.import_module(modules['model_signal'])
        rows = []
        for spec in old.build_requests(cohort='formal'):
            request = spec['request']
            rows.append(dict(job_id=request['job_id'], task=request['task'], parameters=request['parameters'],
                configuration=spec['configuration'], configuration_sha256=spec['configuration_sha256'],
                run_id=spec['run_id'], seeds=spec['seed_sequence'], splits=spec['splits']))
        closeout = dict(retained_request_ids=fixture['retained_job_ids'], old_inventory={'requests': rows})
        for label, expected in fixture['scenarios'].items():
            name, arguments = label.split(':', 1)
            kwargs = dict(cohort='formal', closeout=closeout) if label == 'model_signal_recovery:formal' else json.loads(arguments)
            builder = importlib.import_module(modules[name])
            with self.subTest(scenario=label):
                specs = builder.build_requests(**kwargs)
                payload = json.dumps(specs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
                self.assertEqual(len(specs), expected['count'])
                self.assertEqual(hashlib.sha256(payload).hexdigest(), expected['sha256'])


if __name__ == '__main__':
    unittest.main()

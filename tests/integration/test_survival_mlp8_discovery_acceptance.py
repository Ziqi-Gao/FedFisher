"""V2 acceptance composition: real immutable receipts, mocked numerical boundary.

Numerical ranking validation has its own analytical fixtures. These tests use
the existing scratch-only producer fixture to check that V2 evidence is routed
through its own store/proof and cannot be replaced by prediction-only V1.
"""
from copy import deepcopy
import json
import unittest
from unittest import mock

from tests.integration import test_survival_mlp8_acceptance as fixtures
from effect_modifier_experiment.survival import acceptance as historical
from effect_modifier_experiment.survival_mlp8 import acceptance as gate, artifacts
from effect_modifier_experiment.survival_mlp8 import discovery_protocol as recipe
from effect_modifier_experiment.survival_mlp8.requests import build_requests


class DiscoveryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = owner = fixtures.SurvivalMLP8AcceptanceTests()
        self.addCleanup(owner.doCleanups)
        owner.setUp()
        # The base helper replaces only the numerical store. Route its same
        # checksum boundary through V2, leaving central producer checks real.
        owner.stack.enter_context(mock.patch.object(artifacts, 'DiscoveryArtifactStore', gate.ArtifactStore))
        audit, requests = [], []
        specs = build_requests(discovery=True)
        for old, spec in zip(owner.specs, specs):
            config, raw = spec['configuration'], spec['request']
            parent = owner.root / 'outputs' / config['run_id']
            (owner.root / 'outputs' / old['run_id']).rename(parent)
            fixtures.write_json(parent / 'run_manifest.env', {'fixture_config': config})
            path = gate.OUTBOX_ROOT / 'discovery' / spec['filename']
            fixtures.write_json(path, raw)
            requests.append(owner.request_record(spec, path))
            for seed in spec['seed_sequence']:
                key = config['scenario'], seed
                directory = parent / ('seed_%03d' % seed)
                owner.directories[key] = directory
                fixtures.write_json(directory / 'started.json', dict(execution=fixtures.execution(raw['job_id'])))
                result = json.loads((directory / 'result.json').read_text())
                result['discovery'] = {tag: dict(raw_x_sha256='1' * 64,
                    contrast_truth_sha256='2' * 64, permutation_sha256='3' * 64,
                    recovery=dict(probability_all_signals_selected=.5, auroc=.75)) for tag in recipe.TAGS}
                result['timing']['stages']['discovery_evaluation'] = dict(wall_seconds=3., cpu_seconds=2.)
                fixtures.write_json(directory / 'result.json', result)
                owner.rehash(directory)
            owner.write_completed_job(raw)
            fixtures.write_json(historical.CENTRAL_RETRIES / (raw['job_id'] + '.json'),
                dict(schema_version=2, job_id=raw['job_id'], attempts_started=1, max_attempts=3,
                     next_attempt_at=None, last_failure_reason=None, updated_at='fixture'))
            audit.extend([dict(schema_version=2, job_id=raw['job_id'], event='job_attempt_started', attempt=1),
                dict(schema_version=2, job_id=raw['job_id'], event='job_transition', project='FedFisher',
                     source='pending', destination='running')])
        historical.CENTRAL_AUDIT.write_text(''.join(json.dumps(row) + '\n' for row in audit))
        owner.inventory = owner.inventory_for('pilot', specs, requests)
        owner.inventory.update(bundle='fedfisher-survival-mlp8-v2-pilot', task=recipe.PILOT_TASK)
        fixtures.write_json(owner.inventory_path, owner.inventory)

    def collect(self):
        return gate.collect_acceptance(self.fixture.inventory_path, discovery=True)

    def test_fresh_v2_proof_contains_all_recovery_and_cannot_be_v1_proof(self):
        proof = self.collect()
        self.assertEqual(proof['protocol'], recipe.PROTOCOL)
        self.assertEqual((proof['validated_request_count'], proof['validated_unit_count']), (2, 4))
        self.assertFalse(proof['inference_performed'])
        for unit in proof['units']:
            recovery = unit['observations']['modifier_recovery']
            self.assertEqual(set(recovery), set(recipe.TAGS))
            self.assertTrue(all(row['probability_all_signals_selected'] == .5 for row in recovery.values()))
            self.assertEqual(unit['measured_stage_wall_seconds'], 15.)
        path = gate.publish_acceptance(proof, discovery=True)
        self.assertEqual(gate.verify_pilot_acceptance(path, discovery=True), proof)
        self.assertNotEqual(path, gate.default_acceptance_path())
        with self.assertRaisesRegex(ValueError, 'invalid same-source'):
            gate.verify_pilot_acceptance(path)
        with self.assertRaisesRegex(ValueError, 'publication scope/source mismatch'):
            gate.publish_acceptance(proof)

    def test_cross_method_and_cross_scenario_permutation_pairing(self):
        for key, field in [(('iid', 0), 'raw_x_sha256'), (('covariate_shift', 49), 'permutation_sha256')]:
            path = self.fixture.directories[key] / 'result.json'
            original = json.loads(path.read_text())
            self.fixture.change_result(key, lambda result: result['discovery']['pooled_site_local_pv'].update({field: 'f' * 64}))
            with self.assertRaisesRegex(ValueError, 'do not share test rows, truth and permutations'):
                self.collect()
            fixtures.write_json(path, original)
            self.fixture.rehash(self.fixture.directories[key])

    def test_saved_ranking_change_invalidates_fresh_proof(self):
        path = gate.publish_acceptance(self.collect(), discovery=True)
        self.fixture.change_result(('iid', 0), lambda result:
            result['discovery'][recipe.TAGS[0]]['recovery'].update(auroc=.6))
        with self.assertRaisesRegex(ValueError, 'fresh verification'):
            gate.verify_pilot_acceptance(path, discovery=True)

    def test_old_bundle_and_scope_are_not_v2_evidence(self):
        original = deepcopy(self.fixture.inventory)
        fixtures.write_json(self.fixture.inventory_path, dict(original, bundle='fedfisher-survival-mlp8-v1-pilot'))
        with self.assertRaisesRegex(ValueError, 'fixed coverage or frozen source'):
            self.collect()
        fixtures.write_json(self.fixture.inventory_path, original)
        proof = self.collect()
        proof['scope'] = 'survival_mlp8_pilot_scientific_acceptance'
        fixtures.write_json(gate.default_acceptance_path(discovery=True), proof)
        with self.assertRaisesRegex(ValueError, 'invalid same-source'):
            gate.verify_pilot_acceptance(gate.default_acceptance_path(discovery=True), discovery=True)

"""Exact immutable request coverage and source/input gates, using scratch only."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.scheduling.publication import survival_pooled as builder
from tests.integration.request_fixtures import bind_review_sources

SCRATCH = Path('/scr/del6500/FedFisher/survival-pooled-validation/requests')


def fake_provenance():
    return dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                source_tree_sha256='2' * 64, provenance_sha256='3' * 64)


class SurvivalPooledRequestTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        bind_review_sources(self.stack, builder)
        self.stack.enter_context(mock.patch.object(builder, 'OUTBOX_ROOT', self.root / 'outbox'))
        self.stack.enter_context(mock.patch.object(builder, 'REPORT_ROOT', self.root / 'reports'))
        self.stack.enter_context(mock.patch.object(builder, 'current_provenance', return_value=fake_provenance()))
        self.stack.enter_context(mock.patch.object(builder, 'historical_binding', return_value={'fixture': True}))
        # Publication fixtures exercise exact request bytes and filesystem safety;
        # independent input/acceptance tests exercise real scientific bindings.
        self.stack.enter_context(mock.patch.object(builder, 'review_sources', return_value={'fixture': True}))

    def test_pilot_six_requests_twelve_units_with_no_resource_or_seed_choices(self):
        specs = builder.build_requests()
        self.assertEqual(len(specs), 6)
        self.assertEqual(sum(len(s['seed_sequence']) for s in specs), 12)
        for s in specs:
            self.assertEqual(set(s['request']), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
            self.assertEqual(set(s['request']['parameters']), {'family', 'scenario'})
            self.assertEqual(s['seed_sequence'], [0, 49])
            self.assertEqual(s['configuration']['seeds'], [0, 49])

    def test_formal_96_requests_exactly_1800_units(self):
        specs = builder.build_requests(cohort='formal')
        self.assertEqual(len(specs), 96)
        coverage, configs = {}, {}
        for s in specs:
            p = s['request']['parameters']
            key = p['family'], p['scenario']
            coverage.setdefault(key, []).extend(s['seed_sequence'])
            configs.setdefault(key, set()).add(s['configuration_sha256'])
        self.assertEqual(len(coverage), 6)
        self.assertEqual(sum(map(len, coverage.values())), 1800)
        for key in coverage:
            self.assertEqual(sorted(coverage[key]), list(range(300)))
            self.assertEqual(len(configs[key]), 1)
        self.assertEqual(len(builder.build_requests(cohort='formal', shard_count=8)), 48)

    def test_generations_are_disjoint_and_bad_scope_is_rejected(self):
        first, second = builder.build_requests(1), builder.build_requests(2)
        self.assertTrue({s['request']['job_id'] for s in first}.isdisjoint(
            {s['request']['job_id'] for s in second}))
        for kwargs in ({'generation': 0}, {'generation': True}, {'cohort': 'unknown'},
                       {'shard_count': 8}, {'cohort': 'formal', 'shard_count': True},
                       {'cohort': 'formal', 'shard_count': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                builder.build_requests(**kwargs)

    def test_inventory_last_actual_bytes_and_historical_binding(self):
        original = builder.atomic_create
        targets = []

        def observe(path, data):
            targets.append(path)
            return original(path, data)

        with mock.patch.object(builder, 'atomic_create', side_effect=observe):
            result = builder.write_requests()
        self.assertEqual(len(targets), 7)
        self.assertEqual(targets[-1], Path(result['inventory_path']))
        self.assertEqual(result['request_count'], 6)
        self.assertEqual(result['scientific_cell_count'], 12)
        self.assertEqual(result['pooled_fit_count'], 24)
        self.assertFalse(result['submitted'])
        self.assertFalse(result['formal_plan']['publication_authorized'])
        self.assertEqual(result['historical_inputs'], {'fixture': True})
        for row in result['requests']:
            data = Path(row['path']).read_bytes()
            self.assertEqual(builder.sha256(data), row['sha256'])
            self.assertEqual(json.loads(data)['parameters'], row['parameters'])
        self.assertEqual(json.loads(targets[-1].read_text()), result)

    def test_dirty_source_and_invalid_inputs_fail_before_writes(self):
        for target in ('current_provenance', 'historical_binding'):
            with mock.patch.object(builder, target, side_effect=ValueError('invalid prerequisite')):
                with self.assertRaisesRegex(ValueError, 'invalid prerequisite'):
                    builder.write_requests()
            self.assertFalse((self.root / 'outbox').exists())

    def test_existing_partial_or_finished_bundle_is_preserved(self):
        result = builder.write_requests()
        before = Path(result['inventory_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        self.assertEqual(Path(result['inventory_path']).read_bytes(), before)

    def test_source_or_historical_input_mutation_leaves_no_ready_inventory(self):
        for target, values in (
                ('current_provenance', [fake_provenance(), dict(fake_provenance(), source_tree_sha256='4' * 64)]),
                ('historical_binding', [{'fixture': True}, {'fixture': False}])):
            with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
                root = Path(directory)
                with mock.patch.object(builder, 'OUTBOX_ROOT', root / 'outbox'), mock.patch.object(
                        builder, 'REPORT_ROOT', root / 'reports'), mock.patch.object(builder, target, side_effect=values):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        builder.write_requests()
                self.assertTrue((root / 'outbox' / builder.bundle_name()).exists())
                self.assertFalse((root / 'reports' / builder.inventory_name()).exists())

    def test_formal_requires_same_source_pilot_acceptance_before_publication(self):
        with self.assertRaisesRegex(ValueError, 'acceptance first'):
            builder.write_requests(cohort='formal')
        self.assertFalse((self.root / 'outbox').exists())
        proof = {'path': 'fixture-only', 'sha256': '5' * 64, 'evidence': {'provenance': fake_provenance()}}
        with mock.patch.object(builder, 'acceptance_binding', return_value=proof) as check:
            result = builder.write_requests(cohort='formal', pilot_acceptance='fixture-only')
        self.assertEqual(check.call_count, 2)
        self.assertEqual(result['pilot_acceptance'], proof)
        self.assertEqual(result['request_count'], 96)
        self.assertEqual(result['scientific_cell_count'], 1800)


if __name__ == '__main__':
    unittest.main()

"""Immutable request publication fixtures; never submit or run scientific work."""

from tests.paths import PROJECT_ROOT

from tests.integration.request_fixtures import bind_review_sources

from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = PROJECT_ROOT
SPEC = importlib.util.spec_from_file_location('prepare_survival_requests', ROOT / 'effect_modifier_experiment/scheduling/publication/survival.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)
SCRATCH = Path('/scr/del6500/FedFisher/survival-validation/requests')


def fake_provenance():
    return dict(git_commit='1' * 40, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                source_tree_sha256='2' * 64, provenance_sha256='3' * 64)


class SurvivalRequestTests(unittest.TestCase):
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

    def test_pilot_exact_coverage_without_resource_or_seed_parameters(self):
        requests = builder.build_requests()
        self.assertEqual(len(requests), 10)
        self.assertEqual(len({r['request']['job_id'] for r in requests}), 10)
        for spec in requests:
            request = spec['request']
            self.assertEqual(set(request), {'schema_version', 'project', 'task', 'job_id', 'parameters', 'priority'})
            self.assertEqual(set(request['parameters']), {'cell', 'scenario'})
            self.assertEqual(spec['seed_sequence'], [0])
            self.assertEqual(spec['configuration']['seed'], 0)

    def test_formal_160_shards_cover_exactly_3000_scientific_cells(self):
        specs = builder.build_requests(cohort='formal')
        self.assertEqual(len(specs), 160)
        coverage = {}
        configurations = {}
        for spec in specs:
            p = spec['request']['parameters']
            self.assertEqual(set(p), {'cell', 'scenario', 'shard_index', 'shard_count'})
            key = p['cell'], p['scenario']
            coverage.setdefault(key, []).extend(spec['seed_sequence'])
            configurations.setdefault(key, set()).add(spec['configuration_sha256'])
        self.assertEqual(len(coverage), 10)
        for key, seeds in coverage.items():
            self.assertEqual(sorted(seeds), list(range(300)))
            self.assertEqual(len(configurations[key]), 1)
        self.assertEqual(sum(map(len, coverage.values())), 3000)
        self.assertEqual(len(builder.build_requests(cohort='formal', shard_count=8)), 80)

    def test_identifiers_generations_and_strict_parameters(self):
        one, two = builder.build_requests(1), builder.build_requests(2)
        self.assertTrue({r['request']['job_id'] for r in one}.isdisjoint({r['request']['job_id'] for r in two}))
        for kwargs in ({'generation': 0}, {'generation': True}, {'cohort': 'unknown'},
                       {'shard_count': 8}, {'cohort': 'formal', 'shard_count': True},
                       {'cohort': 'formal', 'shard_count': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                builder.build_requests(**kwargs)

    def test_pilot_inventory_is_published_last_and_matches_materialized_bytes(self):
        real_create = builder.atomic_create
        targets = []

        def observe(path, data):
            targets.append(path)
            return real_create(path, data)

        with mock.patch.object(builder, 'atomic_create', side_effect=observe):
            result = builder.write_requests()
        self.assertEqual(len(targets), 11)
        self.assertEqual(targets[-1], Path(result['inventory_path']))
        self.assertFalse(result['submitted'])
        self.assertFalse(result['formal_plan']['publication_authorized'])
        self.assertEqual(result['scientific_cell_count'], 10)
        self.assertEqual(result['provenance'], fake_provenance())
        for record in result['requests']:
            path = Path(record['path'])
            self.assertEqual(path.parent.name, builder.bundle_name())
            body = path.read_bytes()
            request = json.loads(body)
            self.assertEqual(builder.sha256(body), record['sha256'])
            self.assertEqual(request['job_id'], record['job_id'])
            self.assertEqual(request['parameters'], record['parameters'])
        self.assertEqual(json.loads(targets[-1].read_text()), result)

    def test_dirty_source_blocks_before_any_directory_is_created(self):
        with mock.patch.object(builder, 'current_provenance', side_effect=ValueError('dirty source')):
            with self.assertRaisesRegex(ValueError, 'dirty source'):
                builder.write_requests()
        self.assertFalse((self.root / 'outbox').exists())
        self.assertFalse((self.root / 'reports').exists())

    def test_actual_provenance_helper_rejects_dirty_or_invalid_source(self):
        self.stack.close()
        with ExitStack() as sources:
            bind_review_sources(sources, builder)
            with mock.patch.object(builder, 'compute_provenance', return_value=dict(fake_provenance(), git_dirty='1')):
                with self.assertRaisesRegex(ValueError, 'clean source'):
                    builder.current_provenance()
            with mock.patch.object(builder, 'compute_provenance', return_value=dict(fake_provenance(), git_commit='bad')):
                with self.assertRaisesRegex(ValueError, 'commit'):
                    builder.current_provenance()

    def test_existing_or_partial_bundle_is_preserved(self):
        first = builder.write_requests()
        content = Path(first['inventory_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        self.assertEqual(Path(first['inventory_path']).read_bytes(), content)

    def test_symlink_and_outside_root_refused(self):
        target = self.root / 'real'
        target.mkdir()
        alias = self.root / 'alias'
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            builder.safe_path(alias / 'request.json')
        with self.assertRaises(ValueError):
            builder.safe_path(Path('/home/del6500/projects/FedFisher/unsubmitted.json'))

    def test_source_mutation_during_write_leaves_no_ready_inventory(self):
        with mock.patch.object(builder, 'current_provenance', side_effect=[
                fake_provenance(), dict(fake_provenance(), source_tree_sha256='4' * 64)]):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                builder.write_requests()
        self.assertTrue((self.root / 'outbox' / builder.bundle_name()).exists())
        self.assertFalse((self.root / 'reports' / builder.inventory_name()).exists())

    def test_formal_requests_cannot_be_written_without_actual_pilot_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'acceptance first'):
            builder.write_requests(cohort='formal')
        self.assertFalse((self.root / 'outbox').exists())

    def test_formal_acceptance_is_revalidated_and_bound_to_inventory(self):
        # This is a writer fixture; the acceptance validator has independent
        # fabricated central/artifact tests and is never replaced in production.
        proof = dict(path='fixture-only', sha256='5' * 64, evidence={'provenance': fake_provenance()})
        with mock.patch.object(builder, 'acceptance_binding', return_value=proof) as check:
            result = builder.write_requests(cohort='formal', pilot_acceptance='fixture-only')
        self.assertEqual(check.call_count, 2)
        self.assertEqual(result['pilot_acceptance'], proof)
        self.assertEqual(result['request_count'], 160)
        self.assertEqual(result['scientific_cell_count'], 3000)
        self.assertFalse(result['submitted'])

    def test_changed_acceptance_leaves_no_formal_ready_inventory(self):
        with mock.patch.object(builder, 'acceptance_binding', side_effect=[{'proof': 1}, {'proof': 2}]):
            with self.assertRaisesRegex(ValueError, 'pilot proof changed'):
                builder.write_requests(cohort='formal', pilot_acceptance='fixture-only')
        self.assertFalse((self.root / 'reports' / builder.inventory_name(cohort='formal')).exists())


if __name__ == '__main__':
    unittest.main()

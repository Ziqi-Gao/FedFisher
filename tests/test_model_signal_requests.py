"""Request-only fixtures confined to disposable project scratch directories."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prepare_model_signal_requests_fixture',
    ROOT / 'scripts/prepare_model_signal_requests.py')
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def provenance(**changes):
    result = dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                  source_tree_sha256='c' * 64, provenance_sha256='d' * 64)
    result.update(changes)
    return result


class ModelSignalRequestTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-validation/requests')
        builder.safe_path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.outbox, self.reports = self.root / 'outbox', self.root / 'reports'
        self.stack.enter_context(mock.patch.object(builder, 'OUTBOX_ROOT', self.outbox))
        self.stack.enter_context(mock.patch.object(builder, 'REPORT_ROOT', self.reports))
        self.current = self.stack.enter_context(mock.patch.object(builder, 'compute_provenance', return_value=provenance()))
        self.stack.enter_context(mock.patch.object(builder, '_source_hashes', return_value={'fixture': 'reviewed'}))

    def test_exact_matrix_and_disjoint_formal_seeds(self):
        for cohort, shards, count, units in (('pilot', None, 288, 576),
                ('formal', 8, 2304, 172800), ('formal', 16, 4608, 172800)):
            specs = builder.build_requests(1, cohort, shards)
            self.assertEqual(len(specs), count)
            self.assertEqual(len({s['filename'] for s in specs}), count)
            groups = {}
            for item in specs:
                request, p = item['request'], item['request']['parameters']
                self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
                self.assertEqual(set(p), {'setting', 'cell', 'variant', 'shard_index', 'shard_count'})
                self.assertLessEqual(len(request['job_id']), 128)
                self.assertTrue(request['job_id'].startswith('fedfisher-ms-'))
                self.assertEqual(item['request_sha256'], builder.sha256(builder.encoded(request)))
                self.assertEqual(item['configuration']['setting'], p['setting'])
                groups.setdefault((p['setting'], p['cell'], p['variant']), []).extend(item['seed_sequence'])
            self.assertEqual(len(groups), 288)
            self.assertTrue(all(sorted(seeds) == ([0] if cohort == 'pilot' else list(range(300)))
                                for seeds in groups.values()))
            self.assertEqual(sum(len(s['seed_sequence']) * len(s['splits']) for s in specs), units)
            self.assertEqual(sum(key[1] == 'dslassocov' for key in groups), 8)
        self.assertFalse(self.outbox.exists())

    def test_invalid_generation_scope_and_sharding_are_rejected(self):
        for generation in (0, -1, True, '1', None):
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                builder.build_requests(generation)
        for shards in (1, 4, 32, True):
            with self.subTest(shards=shards), self.assertRaises(ValueError):
                builder.build_requests(1, 'formal', shards)
        with self.assertRaises(ValueError):
            builder.build_requests(1, 'pilot', 8)
        changed = deepcopy(builder.protocol.SETTINGS)
        changed['small_mlp']['signal_strength'] = 1.0
        with mock.patch.object(builder.protocol, 'SETTINGS', changed), self.assertRaisesRegex(ValueError, 'setting changed'):
            builder.build_requests()
        with mock.patch.object(builder.protocol, 'seed_sequence', return_value=[1]), self.assertRaisesRegex(ValueError, 'seed'):
            builder.build_requests()

    def test_publishes_inventory_last_then_refuses_overwrite(self):
        created, original = [], builder.atomic_create
        def record(path, data):
            created.append(path)
            original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=record):
            inventory = builder.write_requests()
        path = self.reports / builder.inventory_name()
        self.assertEqual(created[-1], path)
        self.assertEqual(len(created), 289)
        self.assertEqual(json.loads(path.read_text()), inventory)
        self.assertEqual(inventory['scientific_cell_count'], 576)
        self.assertEqual(inventory['formal_plan']['scientific_cell_count'], 172800)
        self.assertFalse(inventory['submitted'])
        self.assertFalse(inventory['formal_plan']['publication_authorized'])
        self.assertEqual(list(self.outbox.glob('*.json')), [])
        with self.assertRaises(FileExistsError):
            builder.write_requests()

    def test_dirty_source_and_missing_pilot_gate_fail_before_writes(self):
        self.current.return_value = provenance(git_dirty='1')
        with self.assertRaisesRegex(ValueError, 'dirty'):
            builder.write_requests()
        self.current.return_value = provenance()
        with self.assertRaisesRegex(ValueError, 'requires complete'):
            builder.write_requests(cohort='formal')
        self.assertFalse(self.outbox.exists())
        self.assertFalse(self.reports.exists())

    def test_source_race_preserves_partial_bundle_without_inventory(self):
        self.current.side_effect = [provenance(), provenance(source_tree_sha256='e' * 64)]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            builder.write_requests()
        self.assertEqual(len(list((self.outbox / builder.bundle_name()).iterdir())), 288)
        self.assertFalse((self.reports / builder.inventory_name()).exists())

    def test_partial_write_is_not_overwritten_and_symlink_is_refused(self):
        original, calls = builder.atomic_create, []
        def fail(path, data):
            calls.append(path)
            if len(calls) == 2:
                raise OSError('fixture interruption')
            original(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=fail), self.assertRaises(OSError):
            builder.write_requests()
        with self.assertRaises(FileExistsError):
            builder.write_requests()
        self.assertEqual(len(list((self.outbox / builder.bundle_name()).iterdir())), 1)
        link = self.root / 'alias'
        link.symlink_to(self.outbox, target_is_directory=True)
        with mock.patch.object(builder, 'OUTBOX_ROOT', link), self.assertRaisesRegex(ValueError, 'symlink'):
            builder.write_requests(2)

    def test_formal_publication_revalidates_real_gate_boundary(self):
        from effect_modifier_experiment.model_signal_ablation import acceptance
        proof = dict(provenance=provenance(), passed=True)
        path = self.root / 'fixture-proof.json'
        path.write_text(json.dumps(proof))
        with mock.patch.object(acceptance, 'verify_pilot_acceptance', return_value=proof) as verify:
            bound = builder._acceptance_binding('formal', path, provenance())
            self.assertEqual(bound['evidence'], proof)
            self.assertEqual(bound['sha256'], builder.sha256(path.read_bytes()))
            verify.return_value = dict(proof, provenance=provenance(git_commit='f' * 40))
            with self.assertRaisesRegex(ValueError, 'source provenance'):
                builder._acceptance_binding('formal', path, provenance())
            verify.side_effect = ValueError('incomplete evidence')
            with self.assertRaisesRegex(ValueError, 'incomplete evidence'):
                builder.write_requests(cohort='formal', pilot_acceptance=path)
        self.assertFalse(self.outbox.exists())


if __name__ == '__main__':
    unittest.main()

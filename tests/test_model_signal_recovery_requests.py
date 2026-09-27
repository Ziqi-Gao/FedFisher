"""Recovery publication gates and exact retained/replacement coverage."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('recovery_request_fixture', 'prepare_model_signal_recovery_requests.py')
old_builder = load('original_request_fixture', 'prepare_model_signal_requests.py')


def provenance(**changes):
    result = dict(git_commit='a' * 40, git_dirty='0', git_status_sha256='b' * 64,
                  source_tree_sha256='c' * 64, provenance_sha256='d' * 64)
    result.update(changes)
    return result


class RecoveryRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for item in old_builder.build_requests(1, 'formal', 16):
            req = item['request']
            rows.append(dict(job_id=req['job_id'], task=req['task'], parameters=req['parameters'],
                path=str(old_builder.OUTBOX_ROOT / old_builder.bundle_name(1, 'formal') / item['filename']),
                sha256=item['request_sha256'], configuration=item['configuration'],
                configuration_sha256=item['configuration_sha256'], run_id=item['run_id'],
                seeds=item['seed_sequence'], splits=item['splits']))
        cls.closed = dict(binding=dict(builder.protocol.SOURCE_CLOSEOUT),
                          retained_request_ids=[row['job_id'] for row in rows[:129]],
                          old_inventory=dict(requests=rows))

    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/model-signal-recovery-validation/requests')
        builder.safe_path(scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        builder.safe_path(scratch)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch)))
        self.outbox, self.reports = self.root / 'outbox', self.root / 'reports'
        self.stack.enter_context(mock.patch.object(builder, 'OUTBOX_ROOT', self.outbox))
        self.stack.enter_context(mock.patch.object(builder, 'REPORT_ROOT', self.reports))
        self.current = self.stack.enter_context(mock.patch.object(builder, 'compute_provenance', return_value=provenance()))
        self.closeout = self.stack.enter_context(mock.patch.object(builder, '_closeout', return_value=self.closed))
        self.stack.enter_context(mock.patch.object(builder, '_source_hashes', return_value={'fixture': 'reviewed'}))

    def test_pilot_covers_all_baselines_and_continuation_diagnostics(self):
        specs = builder.build_requests()
        self.assertEqual(len(specs), 290)
        self.assertEqual(sum(len(x['seed_sequence']) * 2 for x in specs), 588)
        diagnostics = [x for x in specs if x['request']['parameters']['case'] == 'cap_diagnostic']
        self.assertEqual(len(diagnostics), 2)
        for item in diagnostics:
            self.assertEqual(item['seed_sequence'], [9, 25, 41])
            self.assertEqual(item['configuration']['seeds'], [9, 41])
            self.assertTrue(item['run_id'].endswith('_v2_diagnostic'))
        baselines = [x for x in specs if x not in diagnostics]
        self.assertEqual(len({x['run_id'] for x in baselines}), 288)
        self.assertTrue(all(x['seed_sequence'] == [0] and x['run_id'].endswith('_v2_pilot') for x in baselines))
        for item in specs:
            request = item['request']
            self.assertEqual(set(request), {'schema_version', 'job_id', 'project', 'task', 'parameters', 'priority'})
            self.assertLessEqual(len(request['job_id']), 128)
            self.assertTrue(request['job_id'].startswith('fedfisher-ms-v2-'))
            self.assertEqual(item['request_sha256'], builder.sha256(builder.encoded(request)))

    def test_exact_whole_shard_replacement_union_has_no_gaps_or_duplicates(self):
        specs = builder.build_requests(cohort='formal', closeout=self.closed)
        self.assertEqual(len(specs), 4479)
        self.assertEqual(sum(len(x['seed_sequence']) * 2 for x in specs), 167962)
        retained = set(self.closed['retained_request_ids'])
        units = set()
        for row in self.closed['old_inventory']['requests']:
            if row['job_id'] in retained:
                p = row['parameters']
                units.update((p['setting'], p['cell'], p['variant'], seed, split)
                             for seed in row['seeds'] for split in row['splits'])
        self.assertEqual(len(units), 4838)
        old_keys = {builder._parameters_key(row['parameters']): row
                    for row in self.closed['old_inventory']['requests']}
        for item in specs:
            p = item['request']['parameters']
            old = old_keys[builder._parameters_key(p)]
            self.assertNotIn(old['job_id'], retained)
            self.assertNotEqual(item['request']['job_id'], old['job_id'])
            self.assertNotEqual(item['run_id'], old['run_id'])
            self.assertEqual(set(p), {'setting', 'cell', 'variant', 'shard_index', 'shard_count'})
            for seed in item['seed_sequence']:
                for split in item['splits']:
                    key = (p['setting'], p['cell'], p['variant'], seed, split)
                    self.assertNotIn(key, units)
                    units.add(key)
        self.assertEqual(len(units), 172800)
        self.assertFalse(self.outbox.exists())

    def test_recovery_rejects_changed_or_incomplete_retention_contracts(self):
        candidates = [None, dict(self.closed, retained_request_ids=self.closed['retained_request_ids'][:128]),
            dict(self.closed, retained_request_ids=self.closed['retained_request_ids'][:128] + ['foreign']),
            dict(self.closed, retained_request_ids=self.closed['retained_request_ids'][:128] +
                 self.closed['retained_request_ids'][:1])]
        changed = deepcopy(self.closed)
        changed['old_inventory']['requests'][0]['configuration']['method']['optimizer']['max_rounds'] = 201
        candidates.append(changed)
        changed = deepcopy(self.closed)
        changed['old_inventory']['requests'][0]['seeds'] = [0]
        candidates.append(changed)
        changed = deepcopy(self.closed)
        changed['old_inventory']['requests'][0] = changed['old_inventory']['requests'][1]
        candidates.append(changed)
        for candidate in candidates:
            with self.assertRaises(ValueError):
                builder.build_requests(cohort='formal', closeout=candidate)
        for generation in (0, True, -1, '1', None, 10 ** 80):
            with self.assertRaises(ValueError):
                builder.build_requests(generation)
        with mock.patch.object(builder.protocol, 'seed_sequence', return_value=(1,)), self.assertRaisesRegex(ValueError, 'seed'):
            builder.build_requests()

    def test_new_formal_recipe_cannot_change_old_scientific_configuration(self):
        original_config = builder.protocol.run_config
        def changed(task, parameters):
            config = original_config(task, parameters)
            config['dgp']['signal_strength'] = .125
            return config
        with mock.patch.object(builder.protocol, 'run_config', side_effect=changed), self.assertRaisesRegex(ValueError, 'scientific recipe'):
            builder.build_requests(cohort='formal', closeout=self.closed)

    def test_pilot_is_nested_immutable_and_inventory_is_published_last(self):
        paths, create = [], builder.atomic_create
        def record(path, data):
            paths.append(path)
            create(path, data)
        with mock.patch.object(builder, 'atomic_create', side_effect=record):
            inventory = builder.write_requests()
        self.assertEqual(len(paths), 291)
        self.assertEqual(paths[-1], self.reports / builder.inventory_name())
        self.assertEqual(inventory['request_count'], 290)
        self.assertEqual(inventory['scientific_cell_count'], 588)
        self.assertEqual(inventory['closeout'], self.closed['binding'])
        self.assertEqual(json.loads(paths[-1].read_text()), inventory)
        self.assertFalse(inventory['submitted'])
        self.assertFalse(inventory['formal_plan']['publication_authorized'])
        self.assertEqual(list(self.outbox.glob('*.json')), [])
        self.assertEqual(self.closeout.call_count, 2)
        with self.assertRaises(FileExistsError):
            builder.write_requests()

    def test_dirty_source_and_failed_closeout_never_publish(self):
        self.current.return_value = provenance(git_dirty='1')
        with self.assertRaisesRegex(ValueError, 'dirty'):
            builder.write_requests()
        self.current.return_value = provenance()
        self.closeout.side_effect = ValueError('closeout binding invalid')
        with self.assertRaisesRegex(ValueError, 'closeout binding'):
            builder.write_requests()
        self.assertFalse(self.outbox.exists())

    def test_source_or_closeout_drift_cannot_publish_inventory(self):
        self.current.side_effect = [provenance(), provenance(git_commit='e' * 40)]
        with self.assertRaisesRegex(ValueError, 'source changed'):
            builder.write_requests()
        self.assertFalse((self.reports / builder.inventory_name()).exists())
        self.current.side_effect = None
        self.closeout.side_effect = [self.closed, dict(self.closed, binding={'path': 'changed', 'sha256': 'e' * 64})]
        with self.assertRaisesRegex(ValueError, 'closeout changed'):
            builder.write_requests(generation=2)
        self.assertFalse((self.reports / builder.inventory_name(2)).exists())

    def test_formal_gate_requires_current_source_closeout_and_all_290_pilots(self):
        from effect_modifier_experiment.model_signal_ablation import recovery_acceptance
        proof = dict(provenance=provenance(), closeout=self.closed['binding'], passed=True,
            formal_eligible=True, validated_request_count=290, validated_unit_count=588)
        path = self.root / 'pilot-proof.json'
        path.write_text(json.dumps(proof))
        with mock.patch.object(recovery_acceptance, 'verify_pilot_acceptance', return_value=proof) as verify:
            binding = builder._acceptance_binding('formal', path, provenance(), self.closed)
            self.assertEqual(binding['sha256'], builder.sha256(path.read_bytes()))
            for change in (dict(provenance=provenance(git_commit='f' * 40)), dict(closeout={}),
                           dict(passed=False), dict(formal_eligible=False),
                           dict(validated_request_count=288), dict(validated_unit_count=576)):
                verify.return_value = dict(proof, **change)
                with self.assertRaisesRegex(ValueError, 'acceptance differs'):
                    builder._acceptance_binding('formal', path, provenance(), self.closed)
        with self.assertRaisesRegex(ValueError, 'requires complete'):
            builder._acceptance_binding('formal', None, provenance(), self.closed)
        with self.assertRaisesRegex(ValueError, 'only valid for formal'):
            builder._acceptance_binding('pilot', path, provenance(), self.closed)


if __name__ == '__main__':
    unittest.main()

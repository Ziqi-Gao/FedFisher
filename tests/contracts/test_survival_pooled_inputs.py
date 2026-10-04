"""Tiny file-only historical-input fixtures: no training, prediction or DGP."""
from contextlib import ExitStack
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from effect_modifier_experiment.infrastructure.provenance import PROVENANCE_KEYS, canonical_text
from effect_modifier_experiment.survival_pooled import inputs

SCRATCH = Path('/scr/del6500/FedFisher/survival-pooled-validation/inputs')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.write_text(canonical(value))
    return binding(path)


def provenance(commit):
    value = dict(git_commit=commit, git_dirty='0', git_status_sha256=hashlib.sha256(b'').hexdigest(),
                 source_tree_sha256='3' * 64)
    value['provenance_sha256'] = hashlib.sha256(canonical_text(value, PROVENANCE_KEYS).encode()).hexdigest()
    return value


class SurvivalPooledInputTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        inputs._index.cache_clear()
        inputs._HASH_CACHE.clear()
        self.addCleanup(inputs._index.cache_clear)
        self.addCleanup(inputs._HASH_CACHE.clear)
        self.methods = ('dslasso', 'fedfisher_diag_shallow', 'fedfisher_kfac_shallow')
        self.stack.enter_context(mock.patch.object(inputs, 'METHODS', self.methods))
        self.stack.enter_context(mock.patch.object(inputs, 'SCENARIOS', ('iid',)))
        self.stack.enter_context(mock.patch.object(inputs, 'SEEDS', (0,)))
        self.stack.enter_context(mock.patch.object(inputs, 'PROTECTED_SOURCES', {}))
        central = self.root / 'central'
        central.mkdir()
        self.stack.enter_context(mock.patch.object(inputs, 'CENTRAL_JOBS', central))
        self.audit = self.root / 'audit.jsonl'
        self.audit.write_bytes(b'')
        self.stack.enter_context(mock.patch.object(inputs, 'CENTRAL_AUDIT', self.audit))
        accepted_provenance = provenance('1' * 40)
        old_provenance = provenance('2' * 40)
        self.units = {}
        self.csv_rows = []
        for cell in self.methods:
            prov = accepted_provenance if cell == 'dslasso' else old_provenance
            self.units[cell] = self.make_unit(cell, prov, central)
        old = dict(fedfisher_passed=True, fedfisher_validated_units=2400, source_release_gate_passed=True,
            provenance=old_provenance,
            runs=[dict(config=self.units[c]['config'], manifest=self.units[c]['manifest'])
                  for c in self.methods if c != 'dslasso'],
            validated_units=[self.units[c]['old_unit'] for c in self.methods if c != 'dslasso'],
            central_jobs={self.units[c]['job_id']: dict(job=self.units[c]['job']['job'],
                producer=self.units[c]['producer'], state='completed') for c in self.methods if c != 'dslasso'})
        old_binding = write_json(self.root / 'closeout.json', old)
        accepted = dict(scope='survival_dslasso_integrated_formal_acceptance', epoch=3, passed=True,
            validated_unit_count=600, archived_fedfisher_units=2400, selected_recipe='budget_and_step',
            original_closeout=old_binding, provenance=accepted_provenance,
            units=[self.units['dslasso']['r3_unit']], central_jobs=[self.units['dslasso']['job']])
        accepted_binding = write_json(self.root / 'acceptance.json', accepted)
        csv_path = self.root / 'seed-results.csv'
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.csv_rows[0]))
            writer.writeheader()
            writer.writerows(self.csv_rows)
        for key, value in [('ACCEPTANCE', accepted_binding), ('ORIGINAL_CLOSEOUT', old_binding),
                           ('SEED_RESULTS', binding(csv_path))]:
            self.stack.enter_context(mock.patch.object(inputs, key, value))

    def make_unit(self, cell, prov, central):
        run_id = 'fixture_' + cell + '_iid_300seeds'
        run = self.root / run_id
        path = run / 'seed_000'
        path.mkdir(parents=True)
        config = dict(cell=cell, scenario='iid', run_id=run_id, seeds=list(range(300)),
                      dgp=dict(covariate_dim=2, num_sites=1), method=dict(optimizer={}))
        if cell == 'dslasso':
            config['recovery_recipe'] = 'budget_and_step'
        else:
            config['method']['optimizer'] = dict(architecture='shallow')
        manifest = dict(format_version='2', run_id=run_id, config_json=canonical(config),
                        config_sha256=hashlib.sha256(canonical(config).encode()).hexdigest(), **prov)
        manifest_path = run / 'run_manifest.env'
        manifest_path.write_text(''.join('%s=%s\n' % item for item in manifest.items()))
        identity = dict(run_id=run_id, config_sha256=manifest['config_sha256'],
                        provenance_sha256=prov['provenance_sha256'], seed=0, cell=cell, scenario='iid')
        if cell == 'dslasso':
            identity['recipe'] = 'budget_and_step'
        job_id = 'fixture_' + cell
        execution = dict(job_id=job_id, attempt=1)
        started = write_json(path / 'started.json', dict(identity=identity, execution=execution))
        result = write_json(path / 'result.json', dict(identity=identity, status='complete', fit={},
            fingerprints=dict(training='4' * 64, evaluation='5' * 64, evaluation_truth='6' * 64)))
        stream = io.BytesIO()
        if cell == 'dslasso':
            np.savez(stream, intercept=np.array(0., dtype=np.float64), coefficients=np.zeros(5))
        else:
            # Parameter shape only; these arrays are never fitted or evaluated.
            np.savez(stream, vector=np.zeros(2497), upload_000=np.frombuffer(b'fixture', dtype=np.uint8))
        model = path / 'model.npz'
        model.write_bytes(stream.getvalue())
        artifacts = {'started.json': started, 'result.json': result, 'model.npz': binding(model)}
        artifacts['done.json'] = write_json(path / 'done.json', dict(identity=identity,
            artifacts={name: part['sha256'] for name, part in artifacts.items()}))
        log = self.root / (job_id + '.log')
        log.write_bytes(b'')
        retry = write_json(self.root / (job_id + '.retry.json'), dict(job_id=job_id, attempts_started=1))
        audits = {}
        for name in ('attempt_started', 'running_transition'):
            record = dict(job_id=job_id, event=name, schema_version=2)
            data = (canonical(record) + '\n').encode()
            offset = self.audit.stat().st_size
            with self.audit.open('ab') as handle:
                handle.write(data)
            audits[name] = dict(path=str(self.audit), offset=offset, length=len(data),
                                sha256=hashlib.sha256(data).hexdigest(), record=record)
        producer = dict(execution_receipt=execution, retry=retry,
                        logs=dict(stdout_log=binding(log), stderr_log=binding(log)), audit=audits)
        job_path = central / (job_id + '.json')
        job = dict(job=write_json(job_path, dict(project='FedFisher', state='completed', exit_code=0,
                                                job_id=job_id)), logs=producer['logs'])
        self.csv_rows.append(dict(method=cell, scenario='iid', seed=0, run_id=run_id,
            result_path=result['path'], result_sha256=result['sha256'], producer_commit=prov['git_commit'],
            provenance_sha256=prov['provenance_sha256']))
        return dict(config=config, manifest=binding(manifest_path), artifacts=artifacts,
            producer=producer, job=job, job_id=job_id,
            r3_unit=dict(seed=0, parameters=dict(scenario='iid'), manifest=binding(manifest_path),
                         artifacts=list(artifacts.values()), producer_evidence=producer),
            old_unit=dict(seed=0, cell=cell, scenario='iid', identity=identity, artifacts=artifacts,
                          execution_receipt=execution))

    def test_binding_is_pure_copy_and_index_has_exact_coverage(self):
        first = inputs.historical_binding()
        first['acceptance']['sha256'] = '0' * 64
        self.assertNotEqual(first, inputs.historical_binding())
        report = inputs.validate_historical_inputs()
        self.assertEqual(report['scientific_unit_count'], 3)
        self.assertIs(report['inference_performed'], False)

    def test_load_linear_calls_archived_validator_without_inference(self):
        from effect_modifier_experiment.survival import recovery
        from effect_modifier_experiment.survival.dslasso import SurvivalLinearModel
        with mock.patch.object(recovery, 'validate_result') as validate, \
                mock.patch.object(SurvivalLinearModel, 'predict_eta', side_effect=AssertionError('no inference')):
            loaded = inputs.load_historical('linear', 'iid', 0)
        validate.assert_called_once()
        self.assertEqual(set(loaded), {'dslasso'})
        self.assertEqual(loaded['dslasso']['binding']['producer_commit'], '1' * 40)
        self.assertEqual(loaded['dslasso']['binding']['identity']['recipe'], 'budget_and_step')
        self.assertTrue(inputs.validate_selected_binding(loaded['dslasso']['binding']))

    def test_load_neural_preserves_both_actual_producers_without_inference(self):
        from effect_modifier_experiment.survival import artifacts
        from effect_modifier_experiment.survival.fisher import FisherFit
        with mock.patch.object(artifacts, 'validate_result') as validate, \
                mock.patch.object(FisherFit, 'predict_eta', side_effect=AssertionError('no inference')):
            loaded = inputs.load_historical('shallow', 'iid', 0)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(set(loaded), {'fedfisher_diag', 'fedfisher_kfac'})
        self.assertEqual(loaded['fedfisher_diag']['binding']['producer_commit'], '2' * 40)

    def test_changed_anchor_is_rejected_after_cached_index(self):
        inputs.validate_historical_inputs()
        Path(inputs.ACCEPTANCE['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            inputs.validate_historical_inputs()

    def test_changed_selected_model_fails_before_semantic_validation(self):
        inputs.validate_historical_inputs()
        model = Path(self.units['dslasso']['artifacts']['model.npz']['path'])
        model.write_bytes(b'mutated')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            inputs.load_historical('linear', 'iid', 0)

    def test_completed_directory_with_extra_failure_is_rejected(self):
        directory = Path(self.units['dslasso']['artifacts']['result.json']['path']).parent
        (directory / 'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            inputs.load_historical('linear', 'iid', 0)

    def test_changed_producer_audit_fails_even_with_other_artifact_hashes_valid(self):
        inputs.validate_historical_inputs()
        self.audit.write_bytes(b'changed audit')
        with self.assertRaisesRegex(ValueError, 'audit changed'):
            inputs.load_historical('linear', 'iid', 0)

    def test_symlink_is_rejected_in_bound_input(self):
        path = Path(inputs.SEED_RESULTS['path'])
        backup = path.with_suffix('.original')
        path.rename(backup)
        path.symlink_to(backup)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            inputs.validate_historical_inputs()

    def test_semantic_validator_failure_is_not_promoted_to_loaded_model(self):
        from effect_modifier_experiment.survival import recovery
        with mock.patch.object(recovery, 'validate_result', side_effect=ValueError('invalid saved fit')):
            with self.assertRaisesRegex(ValueError, 'invalid saved fit'):
                inputs.load_historical('linear', 'iid', 0)

    def test_selected_receipt_cannot_claim_other_identity_or_producer(self):
        from effect_modifier_experiment.survival import recovery
        with mock.patch.object(recovery, 'validate_result'):
            receipt = inputs.load_historical('linear', 'iid', 0)['dslasso']['binding']
        bad = deepcopy(receipt)
        bad['producer_commit'] = '9' * 40
        with self.assertRaisesRegex(ValueError, 'receipt differs'):
            inputs.validate_selected_binding(bad)

    def test_selected_fingerprints_read_bound_result_without_restoring_model(self):
        from effect_modifier_experiment.survival import recovery
        with mock.patch.object(recovery, 'validate_result'):
            receipt = inputs.load_historical('linear', 'iid', 0)['dslasso']['binding']
        with mock.patch.object(inputs, '_load_unit', side_effect=AssertionError('no restoration')):
            self.assertEqual(inputs.selected_fingerprints(receipt),
                dict(training='4' * 64, evaluation='5' * 64, evaluation_truth='6' * 64))
        result = Path(receipt['artifacts']['result.json']['path'])
        result.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            inputs.selected_fingerprints(receipt)

    def test_unknown_requests_fail_closed(self):
        for args in [('unknown', 'iid', 0), ('linear', 'other', 0), ('linear', 'iid', True),
                     ('linear', 'iid', -1), ('linear', 'iid', 300)]:
            with self.assertRaisesRegex(ValueError, 'unknown historical'):
                inputs.load_historical(*args)


if __name__ == '__main__':
    unittest.main()

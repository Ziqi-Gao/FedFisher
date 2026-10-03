"""Replay identity and pre-inference refusal fixtures; no pilot execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from effect_modifier_experiment.dslassocov_run_contract import canonical_text
from effect_modifier_experiment.one_shot_effects import artifacts as original_artifacts
from effect_modifier_experiment.one_shot_effects import protocol as original_protocol
from effect_modifier_experiment.one_shot_effects.runner import generate_data, _data_evidence
from effect_modifier_experiment.one_shot_effects.evaluation import array_hash
from effect_modifier_experiment.one_shot_effects.interventions import make_permutations, permutation_hash
from effect_modifier_experiment.one_shot_effects.diagnostics import artifacts, protocol, runner, sources
from effect_modifier_experiment.one_shot_effects.wire import canonical

SCRATCH = Path('/scr/del6500/FedFisher/effects-replay-validation')


class ReplaySourceTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=SCRATCH, prefix='source-contract-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = protocol.run_config(protocol.PILOT_TASK, protocol.cohort_plan()[0])
        self.original_generation_source_files = copy.deepcopy(self.config['original_generation_source_files'])
        # Real fixture files keep these hash/path checks independent of the
        # active checkout and the scientific recipe's immutable archived hashes.
        source_root = self.root/'generation-source'
        for relative in self.original_generation_source_files:
            path = source_root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('bounded generation source fixture '+relative)
        self.config['original_generation_source_files'] = {
            relative: hashlib.sha256((source_root/relative).read_bytes()).hexdigest()
            for relative in self.original_generation_source_files
        }
        path_type = sources.Path
        def source_path(*parts, **kwargs):
            if parts == ('/home/del6500/projects/FedFisher',) and not kwargs:
                return source_root
            return path_type(*parts, **kwargs)
        patch = mock.patch.object(sources, 'Path', side_effect=source_path)
        patch.start(); self.addCleanup(patch.stop)
        self.binding = copy.deepcopy(next(iter(self.config['sources'].values())))
        directory = self.root/Path(self.binding['directory']).name
        (directory/'seed_000').mkdir(parents=True)
        for relative in self.binding['files']:
            path = directory/relative
            path.write_text('bounded source fixture '+relative)
        self.binding.update(directory=str(directory), files={relative: hashlib.sha256((directory/relative).read_bytes()).hexdigest()
                            for relative in self.binding['files']})
        self.config['sources'] = {'fixture': self.binding}
        self.report = self.root/'accepted.json'
        self.report.write_text(json.dumps(dict(status='scientifically_validated',
            source_provenance=self.config['original_source_provenance'],
            frozen_input_hashes={str(directory/p): h for p,h in self.binding['files'].items()})))
        self.config['accepted_report'] = dict(path=str(self.report), sha256=hashlib.sha256(self.report.read_bytes()).hexdigest())

    def test_bound_report_and_every_artifact_are_verified_readonly(self):
        before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.root.rglob('*') if p.is_file()}
        sources.verify_source_bindings(self.config)
        self.assertEqual(before, {p: (p.stat().st_mtime_ns,p.read_bytes()) for p in before})
        for name in self.binding['files']:
            path = Path(self.binding['directory'])/name
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                sources.verify_source_bindings(self.config)
            path.write_bytes(original)

    def test_rehashing_checkpoint_without_accepted_binding_is_rejected(self):
        name = 'seed_000/model.npz'
        path = Path(self.binding['directory'])/name
        path.write_bytes(b'changed')
        self.binding['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, 'hash'):
            sources.verify_source_bindings(self.config)

    def test_archived_generation_hashes_reject_changed_source_before_artifact_checks(self):
        stale = copy.deepcopy(self.config)
        stale['original_generation_source_files'] = self.original_generation_source_files
        self.assertNotEqual(stale['original_generation_source_files'],
                            self.config['original_generation_source_files'])
        with mock.patch.object(sources, '_sha', wraps=sources._sha) as sha:
            with self.assertRaisesRegex(ValueError, 'original generation code changed'):
                sources.verify_source_bindings(stale)
        sha.assert_called_once_with(self.report)
        self.assertEqual(self.original_generation_source_files,
            protocol.run_config(protocol.PILOT_TASK, protocol.cohort_plan()[0])['original_generation_source_files'])

    def test_original_symlink_extra_and_report_change_are_rejected(self):
        path = Path(self.binding['directory'])/'seed_000/result.json'
        original = path.read_bytes()
        alternate = self.root/'alternate'
        alternate.write_bytes(original)
        path.unlink()
        path.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            sources.verify_source_bindings(self.config)
        path.unlink()
        path.write_bytes(original)
        self.report.write_text(self.report.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'report changed'):
            sources.verify_source_bindings(self.config)


class ReplayDataGateTests(unittest.TestCase):
    def setUp(self):
        # Deliberately tiny data fixture, not the pilot's DGP dimensions.
        self.config = original_protocol.run_config(original_protocol.PILOT_TASK,
            dict(recipe='linearity_shallow', dataset='scmv2_base'))
        self.config['dgp'].update(num_sites=2, covariate_dim=5, site_min_samples=30, site_max_samples=30, num_test=20)
        self.data = generate_data(self.config)
        e = self.config['evaluation']
        self.plans = [make_permutations(len(s.holdout), 5, e['holdout_seed_offset']+1000003*s.site_id, 5)
                      for s in self.data.sites]
        test_hashes = [permutation_hash(make_permutations(int(np.sum(self.data.evaluation.site_ids == i)), 5,
            e['test_seed_offset']+1000003*i, 5)) for i in range(2)]
        primary = dict(holdout=dict(permutation_sha256=[permutation_hash(p) for p in self.plans]),
            test=dict(permutation_sha256=test_hashes, raw_x_sha256=array_hash(self.data.evaluation.rows.x)))
        self.sources = dict(linearity=dict(config=self.config,
            result=dict(data_fingerprints=_data_evidence(self.data),models=dict(primary=primary))))

    def test_exact_data_passes_and_each_split_or_plan_mismatch_fails(self):
        self.assertTrue(sources.validate_data_replay(self.data,self.plans,self.sources)['passed'])
        for field in ('train','validation','holdout','partition'):
            changed = copy.deepcopy(self.sources)
            changed['linearity']['result']['data_fingerprints']['sites'][0][field] = '0'*64
            with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
                sources.validate_data_replay(self.data,self.plans,changed)
        bad = list(self.plans)
        bad[0] = make_permutations(len(self.data.sites[0].holdout),5,777,5)
        with self.assertRaisesRegex(ValueError, 'permutation'):
            sources.validate_data_replay(self.data,bad,self.sources)

    def test_mismatch_stops_before_diagnostic_dispatch(self):
        config = protocol.run_config(protocol.PILOT_TASK, protocol.cohort_plan()[0])
        with mock.patch.object(runner,'load_sources',return_value=self.sources), \
             mock.patch.object(runner,'generate_data',return_value=self.data), \
             mock.patch.object(runner,'validate_data_replay',side_effect=ValueError('fingerprint mismatch')), \
             mock.patch.object(runner,'verify_source_bindings') as final_check:
            with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):
                runner.compute_replay(config)
            final_check.assert_not_called()

    def test_site_assignment_swap_preserving_counts_is_rejected(self):
        ids = self.data.evaluation.site_ids.copy()
        i,j = np.flatnonzero(ids==0)[0],np.flatnonzero(ids==1)[0]
        ids[i],ids[j] = ids[j],ids[i]
        altered = replace(self.data,evaluation=replace(self.data.evaluation,site_ids=ids))
        with self.assertRaisesRegex(ValueError,'site assignment'):
            sources.validate_data_replay(altered,self.plans,self.sources)

    def test_changed_consistent_CATE_truth_is_rejected_before_inference(self):
        truth = self.data.evaluation.truth
        altered_truth = replace(truth,tau=-truth.tau,mu_0=truth.mu_1,mu_1=truth.mu_0)
        altered = replace(self.data,evaluation=replace(self.data.evaluation,truth=altered_truth))
        with self.assertRaisesRegex(ValueError,'truth differs'):
            sources.validate_data_replay(altered,self.plans,self.sources)


class ReplayArtifactTests(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=SCRATCH,prefix='replay-artifact-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.provenance = dict(git_commit='a'*40,git_dirty='0',git_status_sha256=hashlib.sha256(b'').hexdigest(),
                              source_tree_sha256='b'*64)
        self.provenance['provenance_sha256'] = hashlib.sha256(canonical_text(
            self.provenance,original_artifacts.PROVENANCE_KEYS).encode()).hexdigest()
        patch = mock.patch.object(original_artifacts,'current_provenance',return_value=self.provenance)
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(artifacts,'verify_source_bindings')
        patch.start(); self.addCleanup(patch.stop)
        def semantic(result,config,identity):
            if result.get('identity') != identity or result.get('status') != 'complete' or result.get('value') != 17:
                raise ValueError('fixture semantic mismatch')
        patch = mock.patch.object(artifacts,'validate_result',side_effect=semantic)
        patch.start(); self.addCleanup(patch.stop)
        self.config = protocol.run_config(protocol.PILOT_TASK,protocol.cohort_plan()[0])
        self.store = artifacts.ReplayArtifactStore(self.config,self.provenance,self.root/self.config['run_id'])
        self.result = dict(status='complete',identity=self.store.identity,value=17)
        self.execution = dict(job_id='fedfisher-effects-replay-fixture',attempt=1,execution_profile='cpu-1core',
            estimated_runtime_seconds=99,allocation=dict(cpu_cores=1,memory_mib=8192,gpu_count=0,
                gpu_memory_mib=0,gpu_utilization_pct=0,exclusive_gpu=True))

    def finish(self):
        self.store.initialize(); self.store.start(self.execution); self.store.complete(self.result)

    def test_json_only_completion_no_checkpoint_or_new_upload(self):
        self.finish()
        self.assertEqual(self.store.status(),'validated')
        self.assertEqual({p.name for p in self.store.unit_dir.iterdir()},{'started.json','result.json','done.json'})
        self.assertEqual(self.store.manifest['protocol'],protocol.PROTOCOL)
        with self.assertRaises(ValueError): self.store.complete(self.result)

    def test_partial_failure_is_preserved_and_never_resumed(self):
        self.store.initialize(); self.store.start(self.execution); self.store.fail(ValueError('refuse mismatch'))
        with self.assertRaisesRegex(ValueError,'partial'): self.store.status()
        with self.assertRaises(ValueError): self.store.complete(self.result)
        self.assertTrue((self.store.unit_dir/'failure.json').is_file())

    def test_rehashed_invalid_result_cannot_be_completion(self):
        self.finish()
        result = dict(self.result,value=18)
        p = self.store.unit_dir/'result.json'; p.write_text(canonical(result))
        d = self.store.unit_dir/'done.json'; marker=json.loads(d.read_text())
        marker['artifacts']['result.json']=hashlib.sha256(p.read_bytes()).hexdigest(); d.write_text(canonical(marker))
        with self.assertRaisesRegex(ValueError,'semantic'): self.store.status()

    def test_changed_new_source_and_original_bindings_refuse_skip(self):
        self.finish()
        with mock.patch.object(original_artifacts,'current_provenance',return_value=dict(self.provenance,git_commit='c'*40)):
            with self.assertRaisesRegex(ValueError,'source provenance'): self.store.status()
        with mock.patch.object(artifacts,'verify_source_bindings',side_effect=ValueError('original changed')):
            with self.assertRaisesRegex(ValueError,'original changed'): self.store.status()

    def test_rehashed_empty_execution_receipt_is_rejected(self):
        self.finish()
        path=self.store.unit_dir/'started.json'; value=json.loads(path.read_text());value['execution']={}
        path.write_text(canonical(value))
        marker_path=self.store.unit_dir/'done.json'; marker=json.loads(marker_path.read_text())
        marker['artifacts']['started.json']=hashlib.sha256(path.read_bytes()).hexdigest()
        marker_path.write_text(canonical(marker))
        with self.assertRaisesRegex(ValueError,'execution receipt'):self.store.status()


if __name__ == '__main__':
    unittest.main()

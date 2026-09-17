"""Shared producer proofs preserve experiment-specific evidence policies."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure import evidence
from effect_modifier_experiment.fedfisher_timing import acceptance as timing
from effect_modifier_experiment.survival import acceptance as survival


class SharedEvidenceTests(unittest.TestCase):
    def setUp(self):
        scratch = Path('/scr/del6500/FedFisher/infrastructure-validation/evidence')
        evidence.ordinary_path(scratch, exists=False)
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.job_id = 'shared-evidence-fixture'
        self.audit = self.root / 'audit.jsonl'
        self.retry = self.root / (self.job_id + '.json')
        self.receipt = dict(job_id=self.job_id, execution_profile='cpu-1core', attempt=1,
            allocation=dict(cpu_cores=1, memory_mib=8192, gpu_count=0, gpu_memory_mib=0,
                            gpu_utilization_pct=0, exclusive_gpu=False),
            estimated_runtime_seconds=10.)
        self.job = {name: self.receipt[name] for name in
                    ('job_id', 'execution_profile', 'allocation', 'estimated_runtime_seconds')}
        self.job['estimated_runtime_seconds'] = 20.
        for attempt in (1, 2):
            for kind in ('stdout', 'stderr'):
                path = self.root / ('%s.attempt-%03d.%s.log' % (self.job_id, attempt, kind))
                path.write_bytes(b'')
                if attempt == 2:
                    self.job[kind + '_log'] = str(path)
        self.retry.write_text(json.dumps(dict(schema_version=2, job_id=self.job_id,
            attempts_started=2, max_attempts=3, next_attempt_at=None,
            last_failure_reason='cpu_interrupted', updated_at='fixture')))
        self.records = [dict(schema_version=2, event='job_attempt_started', job_id=self.job_id,
                             attempt=1),
                        dict(schema_version=2, event='job_transition', job_id=self.job_id,
                             project='FedFisher', source='pending', destination='running')]
        self.audit_lines = [json.dumps(record).encode() + b'\n' for record in self.records]
        self.audit.write_bytes(b''.join(self.audit_lines))
        for module in (timing, survival):
            # The two registered tasks deliberately require different job-ID
            # prefixes. Their existing protocol suites test those policies;
            # this fixture isolates a byte-identical shared producer proof.
            self.stack.enter_context(mock.patch.object(module, 'validate_execution', return_value=None))
            for name, value in (('CENTRAL_LOGS', self.root), ('CENTRAL_RETRIES', self.root),
                                ('CENTRAL_AUDIT', self.audit)):
                self.stack.enter_context(mock.patch.object(module, name, value))

    def proof(self, module):
        return module._producer_evidence(dict(job_id=self.job_id), self.job, self.receipt)

    @staticmethod
    def binding(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_both_experiments_keep_exact_producer_evidence_shape_and_hashes(self):
        audit = {}
        offset = 0
        for name, line, record in zip(('attempt_started', 'running_transition'),
                                      self.audit_lines, self.records):
            audit[name] = dict(path=str(self.audit), offset=offset, length=len(line),
                              sha256=hashlib.sha256(line).hexdigest(), record=record)
            offset += len(line)
        expected = dict(producer_attempt=1, final_completed_attempt=2,
            resumed_from_earlier_attempt=True, execution_receipt=self.receipt,
            retry=self.binding(self.retry),
            logs={kind + '_log': self.binding(self.root / (self.job_id + '.attempt-001.' + kind + '.log'))
                  for kind in ('stdout', 'stderr')}, audit=audit,
            allocation_runtime_evidence='hash_bound_started_receipt_validated_against_fixed_task_profile; '
                'central_audit_and_logs_prove_attempt; original_running_snapshot_not_archived')
        self.assertEqual(self.proof(timing), expected)
        self.assertEqual(self.proof(survival), expected)

    def test_survival_retains_strict_producer_logs_without_changing_timing_policy(self):
        producer_log = self.root / (self.job_id + '.attempt-001.stdout.log')
        producer_log.write_text('RuntimeError: fixture failure\n')
        # The timing adapter historically binds producer log bytes only. Its
        # final-job stderr review is separate and is not broadened or rewritten.
        self.assertEqual(self.proof(timing)['logs']['stdout_log'], self.binding(producer_log))
        with self.assertRaisesRegex(ValueError, 'unexplained failure'):
            self.proof(survival)

    def test_survival_retains_strict_json_in_retry_and_audit_readers(self):
        original = self.retry.read_text()
        self.retry.write_text(original[:-1] + ', "attempts_started": 2}')
        self.assertEqual(self.proof(timing)['final_completed_attempt'], 2)
        with self.assertRaisesRegex(ValueError, 'duplicate key'):
            self.proof(survival)
        self.retry.write_text(original)
        line = self.audit_lines[0].rstrip()
        self.audit.write_bytes(line[:-1] + b', "attempt": 1}\n' + self.audit_lines[1])
        self.assertEqual(self.proof(timing)['producer_attempt'], 1)
        with self.assertRaisesRegex(ValueError, 'duplicate key'):
            self.proof(survival)

    def test_common_reader_rejects_symlinked_evidence_and_parent_paths(self):
        link = self.root / 'linked.json'
        link.symlink_to(self.retry)
        for module in (timing, survival):
            with self.subTest(module=module.__name__), self.assertRaises(ValueError):
                module._binding(link)
        with self.assertRaises(ValueError):
            evidence.file_binding(self.root / '..' / self.root.name / self.retry.name)


if __name__ == '__main__':
    unittest.main()

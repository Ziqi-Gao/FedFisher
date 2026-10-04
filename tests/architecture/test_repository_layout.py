"""Architectural boundaries and compatibility after extracting command logic."""
from tests.paths import PROJECT_ROOT
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

from effect_modifier_experiment.scheduling import adapter, catalog, dispatch

ROOT = PROJECT_ROOT
MOVED = {
    'fedfisher_server_scheduler_adapter': 'effect_modifier_experiment.scheduling.adapter',
    'check_effect_modifier_dslassocov_invariants': 'effect_modifier_experiment.dslassocov.invariants',
    'check_effect_modifier_one_shot_lasso_invariants': 'effect_modifier_experiment.one_shot_lasso.invariants',
    'prepare_fedfisher_scheduler_requests': 'effect_modifier_experiment.scheduling.publication.original',
    'prepare_comparison_pilot_requests': 'effect_modifier_experiment.scheduling.publication.comparison',
    'prepare_effects_replay_requests': 'effect_modifier_experiment.scheduling.publication.effects_replay',
    'prepare_fedfisher_timing_requests': 'effect_modifier_experiment.scheduling.publication.fedfisher_timing',
    'prepare_model_signal_requests': 'effect_modifier_experiment.scheduling.publication.model_signal',
    'prepare_model_signal_recovery_requests': 'effect_modifier_experiment.scheduling.publication.model_signal_recovery',
    'prepare_model_signal_completion_requests': 'effect_modifier_experiment.scheduling.publication.model_signal_completion',
    'prepare_survival_requests': 'effect_modifier_experiment.scheduling.publication.survival',
    'prepare_survival_recovery_requests': 'effect_modifier_experiment.scheduling.publication.survival_recovery',
    'make_synthetic_compare_visualizations': 'reporting.binary.comparison',
    'plot_synthetic_original': 'reporting.binary.original',
    'plot_synthetic_alpha_sweep': 'reporting.binary.alpha_sweep',
    'summarize_synthetic_original': 'reporting.binary.summary',
    'make_completed_noniid_visualizations': 'reporting.effect_modifier.completed',
    'make_completed_noniid_time_cost': 'reporting.effect_modifier.time_cost',
    'make_effect_modifier_visualizations': 'reporting.effect_modifier.factorial',
    'make_one_shot_lasso_fedfisher_comparison': 'reporting.effect_modifier.lasso_fisher',
    'make_unified_noniid_comparison': 'reporting.effect_modifier.unified',
    'report_causal_comparison': 'reporting.effect_modifier.causal_comparison',
    'report_effects_replay': 'reporting.effect_modifier.effects_replay',
    'report_fedfisher_timing': 'reporting.effect_modifier.fedfisher_timing',
    'summarize_effect_modifier_ablations': 'reporting.effect_modifier.ablations',
}

PLANS = {'original': 'effect_modifier_experiment.requests', 'comparison': 'effect_modifier_experiment.causal_comparison.requests', 'effects_replay': 'effect_modifier_experiment.one_shot_effects.diagnostics.requests', 'fedfisher_timing': 'effect_modifier_experiment.fedfisher_timing.requests', 'model_signal': 'effect_modifier_experiment.model_signal_ablation.requests', 'model_signal_recovery': 'effect_modifier_experiment.model_signal_ablation.recovery_requests', 'survival': 'effect_modifier_experiment.survival.requests', 'survival_recovery': 'effect_modifier_experiment.survival.recovery_requests'}


PLANS['model_signal_completion'] = 'effect_modifier_experiment.model_signal_ablation.completion_requests'


class RepositoryLayoutTests(unittest.TestCase):
    def test_legacy_imports_share_implementation_and_module_state(self):
        for old, new in MOVED.items():
            with self.subTest(command=old):
                implementation = importlib.import_module(new)
                compatibility = importlib.import_module('scripts.' + old)
                self.assertIs(compatibility, implementation)
                with mock.patch.object(compatibility, '_compatibility_probe', object(), create=True):
                    self.assertIs(implementation._compatibility_probe, compatibility._compatibility_probe)

    def test_libraries_do_not_import_command_wrappers_or_modify_search_paths(self):
        for directory in ('effect_modifier_experiment', 'reporting'):
            for path in (ROOT / directory).rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith('scripts'), str(path))
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(a.name.startswith('scripts') for a in node.names), str(path))
                    if (directory == 'reporting' or 'scheduling' in path.parts) and isinstance(node, ast.Call):
                        self.assertNotIn(ast.unparse(node.func), ('sys.path.insert', 'sys.path.append'), str(path))

    def test_shared_layers_and_experiment_runner_dependencies_remain_separate(self):
        owners = {'causal_comparison', 'fedfisher_timing', 'model_signal_ablation',
                  'one_shot_effects', 'survival'}
        base = ROOT / 'effect_modifier_experiment'
        for path in base.rglob('*.py'):
            parts = path.relative_to(ROOT).with_suffix('').parts
            parent = '.'.join(parts[:-1])
            owner = parts[1] if len(parts) > 2 else None
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = '.' * node.level + (node.module or '')
                    names = [importlib.util.resolve_name(name, parent) if node.level else name]
                for name in names:
                    if owner in {'infrastructure', 'components'} and name.startswith('effect_modifier_experiment.'):
                        self.assertIn(name.split('.')[1], {'infrastructure', 'components'}, str(path))
                    if owner in owners and name.startswith('effect_modifier_experiment.') and '.runner' in name:
                        self.assertEqual(name.split('.')[1], owner, str(path))

    def test_task_catalog_covers_every_running_fixture_without_profile_drift(self):
        fixtures = [json.loads(p.read_text()) for p in (ROOT / 'tests/fixtures/server_scheduler').glob('*.json')]
        self.assertEqual({row['task'] for row in fixtures}, catalog.ALLOWED_TASKS)
        self.assertEqual(len(catalog.ALLOWED_TASKS), 24)
        for row in fixtures:
            expected = catalog.TASK_PROFILES[row['task']]
            self.assertEqual(row['allocation']['memory_mib'], expected['memory_mib'])
            self.assertEqual(row['execution_profile'], expected['execution_profile'])

    def test_dispatch_sets_runtime_before_loading_any_runner(self):
        for task, contract in catalog.TASK_CONTRACTS.items():
            observed = []
            runner = mock.Mock(return_value=23)
            module = mock.Mock(**{contract.runner_name: runner})
            def load(name):
                self.assertEqual(observed, ['runtime'])
                self.assertEqual(name, contract.runner_module)
                observed.append('import')
                return module
            with self.subTest(task=task), mock.patch.object(adapter, '_safe_runtime',
                    side_effect=lambda *args: observed.append('runtime')), \
                    mock.patch.object(dispatch, 'import_module', side_effect=load):
                job = dict(task=task, job_id='fixture', allocation={'cpu_cores': 1})
                self.assertEqual(adapter.execute_task(job), 23)
                runner.assert_called_once_with(job)

    def test_adapter_import_and_all_protocol_routes_remain_lightweight(self):
        code = '''
import importlib, sys
from effect_modifier_experiment.scheduling import adapter, catalog, dispatch
for contract in catalog.TASK_CONTRACTS.values():
    if contract.protocol_module:
        importlib.import_module(contract.protocol_module)
for name in sys.modules:
    assert name.split('.')[0] not in {'numpy', 'torch', 'nngeometry', 'data', 'models'}, name
    assert not name.endswith('.runner'), name
'''
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_experiment_plans_import_without_science_or_publication(self):
        code = '''import importlib, sys
modules = %r
for name in modules:
    importlib.import_module(name)
for name in sys.modules:
    assert name.split('.')[0] not in {'numpy', 'torch', 'nngeometry'}, name
    assert not name.endswith(('.runner', '.acceptance', '.recovery_acceptance')), name
    assert not name.startswith('effect_modifier_experiment.scheduling.publication'), name
''' % list(PLANS.values())
        subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_review_inventory_binds_extracted_implementations(self):
        for name in ('comparison', 'effects_replay', 'fedfisher_timing', 'model_signal',
                     'model_signal_recovery', 'model_signal_completion', 'survival', 'survival_recovery'):
            module = importlib.import_module('effect_modifier_experiment.scheduling.publication.' + name)
            with self.subTest(builder=name):
                self.assertEqual(module.REVIEW_SOURCES['request_publication'],
                    catalog.PROJECT_ROOT / Path(module.__file__).relative_to(ROOT))
                plan = importlib.import_module(module.build_requests.__module__)
                self.assertEqual(module.REVIEW_SOURCES['request_plan'],
                    catalog.PROJECT_ROOT / Path(plan.__file__).relative_to(ROOT))
                for label, path in catalog.adapter_review_sources().items():
                    self.assertEqual(module.REVIEW_SOURCES[label], path)
                    self.assertTrue((ROOT / path.relative_to(catalog.PROJECT_ROOT)).is_file())

    def test_compatible_cli_help_runs_outside_repository(self):
        # Only argparse entrypoints: some historical renderers have no help mode.
        names = [name for name in MOVED if name.startswith('prepare_')]
        names += ['report_effects_replay', 'report_fedfisher_timing']
        if ROOT == catalog.PROJECT_ROOT:
            names += ['fedfisher_server_scheduler_adapter']
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment.pop('PYTHONPATH', None)
        for name in names:
            with self.subTest(command=name):
                completed = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / (name + '.py')), '--help'],
                    cwd='/scr/del6500/FedFisher', env=environment, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertIn('usage:', completed.stdout)
                self.assertIn('--help', completed.stdout)

    def test_invalid_manifest_cli_fails_before_runtime_or_scientific_imports(self):
        completed = subprocess.run([sys.executable, '-B', '-c',
                'from effect_modifier_experiment.scheduling.adapter import main; raise SystemExit(main())',
                '--job-manifest', 'relative.json'], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('job manifest path must be absolute', completed.stderr)

    def test_scientific_entrypoint_rejects_review_worktrees(self):
        if ROOT == catalog.PROJECT_ROOT:
            source = (ROOT / 'scripts/fedfisher_server_scheduler_adapter.py').read_text()
            self.assertIn('entrypoint is outside canonical tree', source)
            return
        completed = subprocess.run([sys.executable, '-B',
            str(ROOT / 'scripts/fedfisher_server_scheduler_adapter.py'), '--help'], cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(completed.returncode, 2)
        self.assertIn('entrypoint is outside canonical tree', completed.stderr)


if __name__ == '__main__':
    unittest.main()

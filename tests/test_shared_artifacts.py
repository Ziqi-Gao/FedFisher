"""Compatibility and safety checks for the shared, stdlib-only artifact layer."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from effect_modifier_experiment.infrastructure import artifacts, provenance


TEST_ROOT = Path('/scr/del6500/FedFisher/infrastructure-validation')


class SharedArtifactTests(unittest.TestCase):
    def setUp(self):
        artifacts.safe_path(TEST_ROOT)
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_shared_imports_do_not_load_scientific_packages_or_experiments(self):
        script = '''
import sys
from effect_modifier_experiment.infrastructure import artifacts, evidence, provenance, requests
for name in sys.modules:
    assert name.split('.')[0] not in {'numpy', 'torch', 'models', 'data'}, name
    assert not name.startswith(('effect_modifier_experiment.causal_comparison',
        'effect_modifier_experiment.dslassocov', 'effect_modifier_experiment.survival',
        'effect_modifier_experiment.fedfisher_timing')), name
'''
        subprocess.run([sys.executable, '-B', '-c', script], check=True,
                       cwd=provenance.PROJECT_ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_survival_artifacts_import_has_no_sibling_experiment_dependency(self):
        script = '''
import sys
from effect_modifier_experiment.survival import artifacts
assert 'effect_modifier_experiment.infrastructure.artifacts' in sys.modules
assert 'effect_modifier_experiment.infrastructure.provenance' in sys.modules
for name in sys.modules:
    assert not name.startswith(('effect_modifier_experiment.causal_comparison',
        'effect_modifier_experiment.dslassocov',
        'effect_modifier_experiment.fedfisher_timing')), name
'''
        subprocess.run([sys.executable, '-B', '-c', script], check=True,
                       cwd=provenance.PROJECT_ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_old_public_imports_are_compatibility_aliases(self):
        from effect_modifier_experiment import dslassocov_run_contract as legacy
        from effect_modifier_experiment.causal_comparison import artifacts as comparison
        from effect_modifier_experiment.survival import artifacts as survival
        from effect_modifier_experiment.fedfisher_timing import artifacts as timing
        for name in ('canonical_text', 'sha256_text', 'sha256_file', 'env_text',
                     'parse_env_file', 'atomic_create_text', 'compute_provenance',
                     'compute_source_tree_sha256'):
            self.assertIs(getattr(legacy, name), getattr(provenance, name))
        for name in ('canonical', 'digest', 'safe_path', 'read_json', 'create_bytes'):
            for module in (comparison, survival, timing):
                self.assertIs(getattr(module, name), getattr(artifacts, name))

    def test_canonical_json_preserves_historical_bytes_and_rejects_nonfinite(self):
        self.assertEqual(artifacts.canonical({'z': [True, None], 'a': '生存'}),
                         '{"a":"\\u751f\\u5b58","z":[true,null]}')
        self.assertEqual(artifacts.digest({'b': 2, 'a': 1}),
                         '43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777')
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                artifacts.canonical({'nested': [value]})

    def test_json_rejects_duplicate_keys_and_nonfinite_constants(self):
        for index, value in enumerate(('{"a":1,"a":2}', '{"x":{"a":1,"a":2}}',
                                       '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}')):
            path = self.root/('%d.json' % index)
            path.write_text(value)
            with self.assertRaises(ValueError):
                artifacts.read_json(path)
        path = self.root/'valid.json'
        path.write_text('{"nested":[1,true,null]}')
        self.assertEqual(artifacts.read_json(path), {'nested': [1, True, None]})

    def test_path_policy_rejects_escape_root_and_symlink_components(self):
        for path in (Path('relative/file'), artifacts.OUTPUT_ROOT, artifacts.SCRATCH_ROOT,
                     self.root/'..'/'escape', provenance.PROJECT_ROOT/'output.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                artifacts.safe_path(path)
        directory = self.root/'directory'
        directory.mkdir()
        link = self.root/'link'
        link.symlink_to(directory, target_is_directory=True)
        dangling = self.root/'dangling'
        dangling.symlink_to(self.root/'missing')
        for path in (link, link/'result.json', dangling):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'symlink'):
                artifacts.safe_path(path)
        self.assertEqual(artifacts.safe_path(directory/'new.json'), directory/'new.json')

    def test_exclusive_publication_preserves_existing_bytes_and_cleans_temporary(self):
        for name, publish, data, replacement in (
                ('bytes', artifacts.create_bytes, b'original\x00', b'replacement'),
                ('text', provenance.atomic_create_text, 'original\n', 'replacement')):
            path = self.root/name
            publish(path, data)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                publish(path, replacement)
            self.assertEqual(path.read_bytes(), before)
        self.assertEqual({p.name for p in self.root.iterdir()}, {'bytes', 'text'})

    def test_env_roundtrip_and_old_validation_rules(self):
        values = {'first': 'a=b', 'second': '2'}
        text = 'first=a=b\nsecond=2\n'
        self.assertEqual(provenance.canonical_text(values, ('first', 'second')), text)
        self.assertEqual(provenance.env_text(values.items()), text)
        path = self.root/'manifest.env'
        provenance.atomic_create_text(path, '# comment\n\n'+text)
        self.assertEqual(provenance.parse_env_file(path), values)
        for bad in ('first=1\nfirst=2\n', '=value\n', 'not key value\n'):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                provenance.parse_env_file(path)
        for values, keys in (({}, ('missing',)), ({'a': ''}, ('a',)),
                             ({'a': 'line\nbreak'}, ('a',))):
            with self.assertRaises(ValueError):
                provenance.canonical_text(values, keys)

    def _source_fixture(self):
        (self.root/'a.txt').write_bytes(b'alpha\n')
        (self.root/'z.bin').write_bytes(b'\x00\xff\x01')
        (self.root/'link').symlink_to('a.txt')
        return b'z.bin\0link\0a.txt\0'

    def test_source_and_provenance_golden_hashes_preserve_existing_artifact_contract(self):
        # Golden values captured from the pre-extraction contract in 322b739.
        listing = self._source_fixture()
        golden_tree = 'f26154bfd75380f7771e2819dfde125ab9be5f0d9ebe1f0d60b4e6b9147b5951'
        cases = (
            (b'', '0', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
             'e79719476f3e14c414dc1650ac6c00f37622b6b27244f512b6abbadcc85ea4c4'),
            (b' M a.txt\0?? z.bin\0', '1',
             'a6ba427ea9bdaed160437b32d3aea6f6df896147f00bce33b8264c5669011e6c',
             '11df04854960e694f4b3f6dbab304c0bb9bff7a4f5249a7ee58e6c61b86930da'),
        )
        for status, dirty, status_hash, proof_hash in cases:
            def git(args, cwd=None):
                self.assertEqual(cwd, self.root)
                if args[1] == 'rev-parse':
                    return b'1'*40+b'\n'
                if args[1] == 'status':
                    return status
                self.assertEqual(args, ['git', 'ls-files', '--cached', '--others',
                                        '--exclude-standard', '-z'])
                return listing
            with mock.patch.object(provenance, '_run', side_effect=git):
                self.assertEqual(provenance.compute_provenance(self.root), dict(
                    git_commit='1'*40, git_dirty=dirty, git_status_sha256=status_hash,
                    source_tree_sha256=golden_tree, provenance_sha256=proof_hash))

    def test_source_hash_binds_symlink_target_without_following_it(self):
        listing = self._source_fixture()
        with mock.patch.object(provenance, '_run', return_value=listing):
            before = provenance.compute_source_tree_sha256(self.root)
            (self.root/'link').unlink()
            (self.root/'link').symlink_to('missing-target')
            self.assertNotEqual(provenance.compute_source_tree_sha256(self.root), before)
            (self.root/'directory').mkdir()
            with mock.patch.object(provenance, '_run', return_value=b'directory\0'):
                with self.assertRaisesRegex(ValueError, 'unsupported entry'):
                    provenance.compute_source_tree_sha256(self.root)


if __name__ == '__main__':
    unittest.main()

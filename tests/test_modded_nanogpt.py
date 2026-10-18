import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

adapter = module('modded_nanogpt', ROOT / 'webui/tools/modded_nanogpt.py')
server = module('server', ROOT / 'webui/server.py')
TABLE = '''## World record history
# | Record time | Description | Date | Log | Contributors
--- | --- | --- | --- | --- | ---
1 | 45 minutes | Baseline | 05/28/24 | none | @one
2 | 3 minutes | [Muon](https://example.com/post) | 01/01/25 | [log](records/track_1_short/second/log.txt) | @two
2 | 3.1 minutes | Re-timing | 02/01/25 | none | @two
3 | 3.05 minutes | FP8 head | 03/01/25 | none | @three
## Rules
1 | 1 minute | Must not be parsed | 03/01/25 | none | @four
'''


class HistoryTests(unittest.TestCase):
    def test_parse_units_and_retiming(self):
        records = adapter.parse_records(TABLE)
        self.assertEqual([r['seconds'] for r in records], [2700, 180, 186, 183])
        self.assertEqual([r['kind'] for r in records], ['record', 'record', 'retime', 'record'])
        self.assertEqual(records[1]['description'], 'Muon')
        self.assertNotEqual(records[1]['comparison_group'], records[2]['comparison_group'])
        self.assertEqual(records[2]['comparison_group'], records[3]['comparison_group'])
        self.assertEqual(adapter.duration('39.9 seconds'), 39.9)
        with self.assertRaises(ValueError):
            adapter.duration('unknown')
        with self.assertRaises(ValueError):
            adapter.parse_records('not the expected upstream table')

    def test_only_extract_python_source_and_safe_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = 'import torch\nvalue = 1\n'
            (root / 'log.txt').write_text('=' * 30 + '\n' + code + '=' * 30 + '\nstep:10 val_loss:3.28\n')
            self.assertEqual(adapter.snapshot(root, {'log_paths': ['log.txt']})[0], code)
            (root / 'log.txt').write_text('step:10 val_loss:3.28\n')
            self.assertEqual(adapter.snapshot(root, {'log_paths': ['log.txt']}), (None, None))
            self.assertIsNone(adapter.local_file(root, '../escape'))
        self.assertIsNone(adapter.source_url('javascript:alert(1)', 'abc'))
        self.assertIn('/blob/abc/records/', adapter.source_url('records/log.txt', 'abc'))

    def test_import_is_repeatable_and_timing_reset_reaches_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, output = root / 'upstream', root / 'history'
            repo.mkdir()
            (repo / 'README.md').write_text(TABLE)
            (repo / 'LICENSE').write_text('MIT test fixture')
            log = repo / 'records/track_1_short/second/log.txt'
            log.parent.mkdir(parents=True)
            log.write_text('=' * 30 + '\nimport torch\nx = 1\n' + '=' * 30 + '\ntraining output\n')
            for args in [['init', '-q'], ['add', '.'], ['-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'fixture']]:
                subprocess.run(['git', '-C', str(repo), *args], check=True)
            adapter.import_history(repo, output)
            def hashes():
                return {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in output.rglob('*') if p.is_file()}
            before = hashes()
            adapter.import_history(repo, output)
            self.assertEqual(before, hashes())
            state = server.build_state(str(output))
            exps = state['experiments']
            self.assertEqual(state['metric'], 'training_seconds')
            self.assertEqual(len(exps), 4)
            self.assertEqual(exps[2]['status'], 'retime')
            self.assertTrue(exps[2]['comparison_reset'])
            self.assertIsNone(exps[2]['best_before'])
            self.assertEqual(exps[2]['best_after'], 186)
            self.assertEqual(exps[3]['delta_best'], -3)
            self.assertEqual(state['summary']['best_val_loss'], 183)
            self.assertEqual(server.project_card('id', {'path': str(output)})['best'], 183)
            self.assertFalse(server.managed(str(output)))
            (output / 'workspace.json').write_text(json.dumps({'design': 'fixed'}))
            self.assertFalse(server.managed(str(output)))
            (output / 'workspace.json').unlink()
            self.assertIn('multi-file trainer', server.agents_state(str(output))['execution_note'])
            # Working-tree edits must not be attributed to the pinned upstream revision.
            log.write_text('=' * 30 + '\nimport torch\nx = 999\n' + '=' * 30 + '\n')
            adapter.import_history(repo, output)
            self.assertFalse((output / 'runs' / exps[1]['run_id'] / 'train.py').exists())
            ordinary = root / 'existing-project'
            ordinary.mkdir()
            (ordinary / 'train.py').write_text('do_not_overwrite = True')
            with self.assertRaises(ValueError):
                adapter.import_history(repo, ordinary)

    def test_legacy_project_has_no_comparison_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'results.tsv').write_text('commit\tval_loss\tmemory_gb\tstatus\trun_id\tdescription\n'
                                            '\t0.2\t3\tkeep\t-\tbaseline\n\t0.18\t3\tkeep\t-\tbetter\n')
            state = server.build_state(str(root))
            self.assertEqual(state['summary']['best_val_loss'], 0.18)
            self.assertFalse(any(e['comparison_reset'] for e in state['experiments']))


if __name__ == '__main__':
    unittest.main()

import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    'research_runner', Path(__file__).resolve().parents[1] / 'webui/research_runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def parameters(**changes):
    return {**dict(batch_size=2, vocab_size=100, context_length=8, num_layers=1,
                d_model=16, num_heads=2, d_ff=32, t=4, t_c=4, t_w=1,
                eval_every=2, eval_batches=2, save_every=4, seed=7,
                lr_max=.001, lr_min=.0001, clip_threshold=1., device='cpu'), **changes}


@contextlib.contextmanager
def fake_trainer(root, *, finish=True, final_validation=True, dataset=None):
    """Exercise the adapter itself without importing Torch, W&B, or student code."""
    source = root / 'copy'
    entry = source / 'cs336_basics/pretrain.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('# A local fake entry point; no training runs.\n')
    package = types.ModuleType('cs336_basics')
    package.__file__ = str(entry.parent / '__init__.py')
    package.__path__ = [str(entry.parent)]
    model_module = types.ModuleType('cs336_basics.gpt')
    model_module.__file__ = str(entry.parent / 'gpt.py')
    package.gpt = model_module
    train = source / 'train.npy'
    valid = source / 'valid.npy'
    train.write_bytes(b'fake train data')
    valid.write_bytes(b'fake validation data')
    torch = types.ModuleType('torch')
    torch.rng = 0
    torch.manual_seed = lambda seed: setattr(torch, 'rng', seed)
    torch.get_rng_state = lambda: torch.rng
    torch.set_rng_state = lambda value: setattr(torch, 'rng', value)
    torch.random = types.SimpleNamespace(default_generator=types.SimpleNamespace(manual_seed=torch.manual_seed))
    numpy = types.ModuleType('numpy')
    numpy.random = types.SimpleNamespace(seed=lambda seed: None)
    samples = []
    restored = []

    def loader(data, batch_size, context_length, device):
        samples.append((Path(data.filename).name, torch.rng))
        torch.rng += 91
        return None, None

    def config(**kwargs):
        return types.SimpleNamespace(dataset=str(dataset or train), val_dataset=str(valid), **kwargs)

    namespace = dict(__file__=str(entry), data_loader=loader, hyperparams=config, torch=torch,
                     SimpleNamespace=types.SimpleNamespace, Path=Path, restored=restored,
                     should_finish=finish, final_validation=final_validation)
    exec('''
def main(cfg):
    # Simulate model initialization consuming a different amount of randomness.
    torch.rng += 37
    initial_rng = torch.rng
    for step in range(1, cfg.t + 1):
        data_loader(SimpleNamespace(filename=cfg.dataset), cfg.batch_size, cfg.context_length, cfg.device)
        restored.append(torch.rng == initial_rng)
        if step % cfg.eval_every == 0 and (step != cfg.t or final_validation):
            for _ in range(cfg.eval_batches):
                data_loader(SimpleNamespace(filename=cfg.val_dataset), cfg.batch_size, cfg.context_length, cfg.device)
                restored.append(torch.rng == initial_rng)
            run.log({'val_loss': 4.0 / step}, step=step)
    checkpoint = Path('checkpoints') / run.name
    checkpoint.mkdir(parents=True)
    (checkpoint / 'model.pt').write_text('fake checkpoint')
    if should_finish:
        run.finish()
''', namespace)
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    try:
        with patch.dict(sys.modules, {'torch': torch, 'numpy': numpy, 'cs336_basics': package,
                                      'cs336_basics.gpt': model_module}), \
             patch.object(runner.importlib.util, 'find_spec', return_value=types.SimpleNamespace(origin=str(entry))), \
             patch.object(runner.runpy, 'run_module', return_value=namespace), \
             patch.dict(os.environ, {}, clear=False), contextlib.redirect_stdout(io.StringIO()) as stdout:
            yield types.SimpleNamespace(source=source, samples=samples, restored=restored,
                                        stdout=stdout, namespace=namespace, torch=torch,
                                        model_module=model_module)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path


class ParameterTests(unittest.TestCase):
    def test_valid_parameters_and_zero_warmup(self):
        p = parameters()
        self.assertEqual(runner.validate(p), p)
        self.assertEqual(runner.validate({**p, 't_w': 0})['t_w'], 0)

    def test_rejects_invalid_evaluation_and_model_shapes(self):
        cases = [dict(t=3), dict(t_c=1), dict(t_w=5), dict(d_model=15),
                 dict(d_model=6, num_heads=2), dict(lr_min=.1), dict(device='unknown')]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate({**parameters(), **change})

    def test_rejects_nonfinite_boolean_and_noninteger_parameters(self):
        for key, value in [('seed', True), ('t', 4.0), ('eval_batches', 0),
                           ('lr_max', math.nan), ('lr_min', math.inf),
                           ('clip_threshold', -1), ('lr_max', True)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runner.validate({**parameters(), key: value})
        with self.assertRaises(ValueError):
            runner.validate([])


class RecorderTests(unittest.TestCase):
    def test_records_final_metric_and_elapsed_time(self):
        recorder = runner.Recorder('/tmp/run-123/metrics.json', 4)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            recorder.log({'val_loss': 1.25, 'loss': 2, 'invalid': math.nan, 'flag': True}, step=4)
        recorder.finish()
        result = recorder.result({'seed': 7})
        self.assertEqual(result['val_loss'], 1.25)
        self.assertEqual(result['num_steps'], 4)
        self.assertGreaterEqual(result['training_seconds'], 0)
        self.assertEqual(recorder.name, 'turing-run-123')
        self.assertEqual(json.loads(output.getvalue()),
                         {'type': 'progress', 'step': 4, 'val_loss': 1.25, 'loss': 2.0})

    def test_rejects_incomplete_stale_or_invalid_metrics(self):
        for finished, step, value in [(False, 4, 1.0), (True, 2, 1.0),
                                      (True, 4, math.nan), (True, 4, math.inf)]:
            with self.subTest(finished=finished, step=step, value=value):
                recorder = runner.Recorder('/tmp/run/metrics.json', 4)
                with contextlib.redirect_stdout(io.StringIO()):
                    recorder.log({'val_loss': value}, step=step)
                if finished:
                    recorder.finish()
                with self.assertRaisesRegex(ValueError, 'final-step validation'):
                    recorder.result({})


class AdapterTests(unittest.TestCase):
    def test_adapter_captures_metric_and_keeps_sampling_streams_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'artifacts/experiment-1/metrics.json'
            with fake_trainer(root) as fake:
                before = (fake.source / 'cs336_basics/pretrain.py').read_bytes()
                runner.cs336(fake.source, parameters(), output)
                self.assertEqual(json.loads(output.read_text())['val_loss'], 1.0)
                self.assertEqual([seed for name, seed in fake.samples if name == 'train.npy'], [7, 8, 9, 10])
                self.assertEqual([seed for name, seed in fake.samples if name == 'valid.npy'],
                                 [1_000_007, 1_000_008, 1_000_007, 1_000_008])
                self.assertTrue(all(fake.restored))
                self.assertEqual(before, (fake.source / 'cs336_basics/pretrain.py').read_bytes())
                self.assertTrue((output.parent / 'checkpoints/turing-experiment-1/model.pt').is_file())
                self.assertFalse((fake.source / 'checkpoints').exists())
                self.assertIn('val_loss: 1.00000000', fake.stdout.getvalue())
                self.assertFalse(output.with_suffix('.tmp').exists())

    def test_adapter_never_writes_comparable_score_for_incomplete_run(self):
        for changes in [dict(finish=False), dict(final_validation=False)]:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / 'artifacts/run/metrics.json'
                with fake_trainer(root, **changes) as fake:
                    with self.assertRaisesRegex(ValueError, 'final-step validation'):
                        runner.cs336(fake.source, parameters(), output)
                    self.assertFalse(output.exists())

    def test_adapter_rejects_dataset_outside_copy_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / 'original.npy'
            original.write_bytes(b'original data')
            with fake_trainer(root, dataset=original) as fake:
                with self.assertRaisesRegex(ValueError, 'dataset must exist inside'):
                    runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])
                self.assertEqual(original.read_bytes(), b'original data')

    def test_adapter_rejects_entrypoint_resolving_to_another_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root) as fake:
                with patch.object(runner.importlib.util, 'find_spec',
                                  return_value=types.SimpleNamespace(origin='/other/project/pretrain.py')):
                    with self.assertRaisesRegex(ValueError, 'outside the selected source'):
                        runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])

    def test_adapter_rejects_cached_model_module_from_original_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root) as fake:
                fake.model_module.__file__ = str(root / 'original/cs336_basics/gpt.py')
                with self.assertRaisesRegex(ValueError, 'Model code resolves outside'):
                    runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])

    def test_adapter_rejects_symlink_entrypoint_outside_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root) as fake:
                original = root / 'original-pretrain.py'
                original.write_text('raise RuntimeError("must not run")\n')
                entry = fake.source / 'cs336_basics/pretrain.py'
                entry.unlink()
                entry.symlink_to(original)
                with self.assertRaisesRegex(ValueError, 'entry point'):
                    runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])

    def test_adapter_checks_other_imported_package_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root) as fake:
                tokenizer = types.ModuleType('cs336_basics.tokenizer')
                tokenizer.__file__ = str(root / 'original/tokenizer.py')
                with patch.dict(sys.modules, {'cs336_basics.tokenizer': tokenizer}):
                    with self.assertRaisesRegex(ValueError, 'Model code resolves outside'):
                        runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])

    def test_adapter_rejects_training_on_validation_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root, dataset=root / 'copy/valid.npy') as fake:
                with self.assertRaisesRegex(ValueError, 'datasets must be different'):
                    runner.cs336(fake.source, parameters(), root / 'metrics.json')
                self.assertEqual(fake.samples, [])

    def test_loader_failure_restores_rng_and_does_not_record_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with fake_trainer(root) as fake:
                def broken_loader(*args):
                    fake.torch.rng = 12345
                    raise RuntimeError('dataset failed')

                fake.namespace['data_loader'] = broken_loader
                output = root / 'artifacts/run/metrics.json'
                with self.assertRaisesRegex(RuntimeError, 'dataset failed'):
                    runner.cs336(fake.source, parameters(), output)
                self.assertEqual(fake.torch.rng, 7 + 37)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()

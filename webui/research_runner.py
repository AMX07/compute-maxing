"""Run an existing trainer with fixed data/evaluation and capture metrics for Turing.

The CS336 adapter invokes existing functions without modifying assignment source.
It uses local metrics instead of opening a W&B session.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import sys
import time


def validate(parameters):
    if not isinstance(parameters, dict):
        raise ValueError('Parameters must be an object.')
    integers = ('batch_size', 'vocab_size', 'context_length', 'num_layers', 'd_model', 'num_heads', 'd_ff',
                't', 't_c', 't_w', 'eval_every', 'eval_batches', 'save_every', 'seed')
    for key in integers:
        value = parameters.get(key)
        if type(value) is not int or value < (0 if key in ('seed', 't_w') else 1):
            raise ValueError(f'{key} must be a positive integer (seed/warmup may be zero).')
    for key in ('lr_max', 'lr_min', 'clip_threshold'):
        value = parameters.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    if parameters['lr_min'] > parameters['lr_max']:
        raise ValueError('Minimum learning rate must not exceed peak learning rate.')
    if parameters['d_model'] % parameters['num_heads'] or (parameters['d_model'] // parameters['num_heads']) % 2:
        raise ValueError('Model width must divide into even-sized attention heads.')
    if parameters['t'] % parameters['eval_every']:
        raise ValueError('The final training step must include validation.')
    if parameters['t_c'] <= parameters['t_w'] or parameters['t_w'] > parameters['t']:
        raise ValueError('Warmup must end before the cosine schedule and training end.')
    if parameters.get('device') not in ('cpu', 'mps', 'cuda'):
        raise ValueError('Unsupported device.')
    return parameters


class Recorder:
    def __init__(self, output, steps):
        self.output = Path(output)
        self.name = 'turing-' + self.output.parent.name
        self.steps = steps
        self.last_validation = None
        self.finished = False
        self.started = time.monotonic()

    def log(self, values, step=None):
        clean = {key: float(value) for key, value in values.items()
                 if type(value) in (int, float) and math.isfinite(value)}
        if 'val_loss' in clean:
            self.last_validation = (step, clean['val_loss'])
        print(json.dumps({'type': 'progress', 'step': step, **clean}), flush=True)

    def finish(self):
        self.finished = True

    def result(self, parameters):
        if not self.finished or self.last_validation is None or self.last_validation[0] != self.steps:
            raise ValueError('Trainer did not complete final-step validation; no comparable score was produced.')
        return {'val_loss': self.last_validation[1], 'num_steps': self.steps,
                'training_seconds': round(time.monotonic() - self.started, 3),
                'parameters': parameters, 'evaluation': 'Fixed sampled windows; final-step cross-entropy',
                'adapter': 'cs336-pretrain-v1'}

    def __repr__(self):
        return f'Turing run {self.name}'


def cs336(source, parameters, output):
    p = validate(parameters)
    source = Path(source).resolve()
    entry = source / 'cs336_basics/pretrain.py'
    if not entry.is_file() or not entry.resolve().is_relative_to(source):
        raise ValueError('The copied pretraining entry point is missing.')
    # Import the copied package, even if an older editable install is on sys.path.
    sys.path.insert(0, str(source))
    spec = importlib.util.find_spec('cs336_basics.pretrain')
    if spec is None or Path(spec.origin).resolve() != entry.resolve():
        raise ValueError('The training module resolves outside the selected source directory.')
    os.environ['WANDB_MODE'] = 'offline'
    import numpy as np
    import torch
    random.seed(p['seed'])
    np.random.seed(p['seed'])
    torch.manual_seed(p['seed'])
    namespace = runpy.run_module('cs336_basics.pretrain', run_name='turing_research_target')
    main = namespace['main']
    for name, module in tuple(sys.modules.items()):
        if (name == 'cs336_basics' or name.startswith('cs336_basics.')) and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise ValueError('Model code resolves outside the selected source directory.')
    globals_ = main.__globals__
    config = namespace['hyperparams'](**{k: v for k, v in p.items() if k != 'seed'})
    for name in ('dataset', 'val_dataset'):
        path = Path(getattr(config, name)).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'{name} must exist inside the copied project.')
    if Path(config.dataset).resolve() == Path(config.val_dataset).resolve():
        raise ValueError('Training and validation datasets must be different.')
    loader = globals_['data_loader']
    counts = {'train': 0, 'val': 0}
    validation_path = Path(config.val_dataset).resolve()

    def seeded_loader(dataset, batch_size, context_length, device):
        validation = Path(dataset.filename).resolve() == validation_path
        kind = 'val' if validation else 'train'
        # Both streams are independent of model initialization and of one another.
        index = counts[kind] % config.eval_batches if validation else counts[kind]
        seed = p['seed'] + (1_000_000 if validation else 0) + index
        counts[kind] += 1
        rng = torch.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(seed)
            return loader(dataset, batch_size, context_length, device)
        finally:
            torch.set_rng_state(rng)

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(output, config.t)
    globals_['run'] = recorder
    globals_['data_loader'] = seeded_loader
    # main writes relative checkpoint paths; confine them to this experiment's artifacts.
    os.chdir(output.parent)
    main(config)
    result = recorder.result(p)
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(output)
    print(f"val_loss: {result['val_loss']:.8f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True, choices=['cs336'])
    parser.add_argument('--source', required=True)
    parser.add_argument('--parameters', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cs336(args.source, json.loads(Path(args.parameters).read_text()), args.output)


if __name__ == '__main__':
    main()

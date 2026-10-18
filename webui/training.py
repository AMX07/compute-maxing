"""Persistent, single-run training jobs for the local Turing UI (standard library only)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import threading
import uuid

ACTIVE = {'queued', 'running', 'stopping'}
ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def defaults(root):
    root = Path(root).resolve()
    if (root / 'import.json').exists() and (root.parent / 'train_gpt.py').exists():
        return dict(profile='modded-nanogpt', target='ssh', host='', directory='~/modded-nanogpt',
                    command='uv run --no-project torchrun --standalone --nproc_per_node=8 train_gpt.py', timeout_minutes=30)
    profile = read(root / 'project.json', {})
    command = profile.get('train_command') or 'uv run train.py'
    if root.parent == ROOT / 'workspaces' and not (root / 'pyproject.toml').exists():
        image = os.environ.get('TURING_IMAGE_PROJECT', str(ROOT.parent / 'autoresearch-image'))
        command = shlex.join(['uv', 'run', '--project', image, 'train.py'])
    remote = profile.get('remote') or {}
    return dict(profile='project', target='ssh' if remote.get('host') else 'local', host=remote.get('host') or '',
                directory=remote.get('dir') if remote.get('host') else str(root), command=command, timeout_minutes=30)


def validate(config):
    if not isinstance(config, dict):
        raise ValueError('Training settings must be an object.')
    c = {k: str(config.get(k, '')).strip() for k in ('profile', 'target', 'host', 'directory', 'command')}
    if c['profile'] not in ('project', 'modded-nanogpt', 'custom') or c['target'] not in ('local', 'ssh'):
        raise ValueError('Choose a training profile and a local or SSH target.')
    if any('\x00' in v or '\n' in v or '\r' in v or len(v) > 4096 for v in c.values()):
        raise ValueError('Settings must be single-line values shorter than 4096 characters.')
    if not c['directory'] or not c['command']:
        raise ValueError('Working directory and training command are required.')
    if c['target'] == 'ssh' and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.@-]*', c['host']):
        raise ValueError('Enter an SSH host alias or user@hostname. Configure ports and keys in ~/.ssh/config.')
    if c['target'] == 'ssh' and not c['directory'].startswith(('/', '~/')):
        raise ValueError('Use an absolute remote directory or a path starting with ~/.')
    try:
        c['timeout_minutes'] = int(config.get('timeout_minutes', 30))
        argv = shlex.split(c['command'])
    except (ValueError, TypeError):
        raise ValueError('Enter a whole-number timeout and a command with balanced quotes.') from None
    if not 1 <= c['timeout_minutes'] <= 1440 or not argv:
        raise ValueError('Timeout must be between 1 and 1440 minutes; a command is required.')
    if c['target'] == 'local':
        c['directory'] = str(Path(c['directory']).expanduser().resolve())
    return c


def remote_directory(path):
    return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith('~/') else shlex.quote(path)


def ssh_command(c, argv):
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', c['host'],
            'cd -- ' + remote_directory(c['directory']) + ' && exec ' + shlex.join(argv)]


# This probe only inspects the environment; it never imports the trainer or fetches datasets/kernels.
PROBE = r'''
import glob, importlib, json, os, shutil, sys
c = json.loads(sys.argv[1]); checks = []
def add(name, ok, detail):
    checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))
add('Working directory', os.path.isdir('.'), os.getcwd())
add('Training executable', bool(shutil.which(c['argv'][0])), c['argv'][0])
for arg in c['argv'][1:]:
    if arg.endswith('.py') and not arg.startswith('-'):
        add('Training script', os.path.isfile(arg), arg)
if c['profile'] == 'project' and os.path.isfile('inference.py') and os.path.isfile('prepare.py'):
    try:
        import mlx.core
        add('MLX', True, 'Installed')
    except Exception as e:
        add('MLX', False, str(e))
    cache = os.path.expanduser('~/.cache/autoresearch-image')
    add('Image dataset', all(os.path.isfile(os.path.join(cache, f)) for f in ('train.npz', 'test.npz')), cache)
if c['profile'] == 'modded-nanogpt':
    add('Trainer source', os.path.isfile('train_gpt.py') and os.path.isdir('track_1_short'), 'train_gpt.py + track_1_short/')
    try:
        import torch
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        add('8 H100 GPUs', len(names) >= 8 and all('H100' in n for n in names[:8]), ', '.join(names) or 'No CUDA GPUs detected')
        add('PyTorch version', torch.__version__.split('+')[0].startswith('2.10.'), torch.__version__)
    except Exception as e:
        add('PyTorch / CUDA', False, str(e))
    for name in ('kernels', 'tiktoken', 'numpy', 'huggingface_hub'):
        try:
            importlib.import_module(name); add(name, True, 'Installed')
        except Exception as e:
            add(name, False, str(e))
    data = os.path.join(os.environ.get('DATA_PATH', ''), 'data/fineweb10B')
    add('FineWeb training data', bool(glob.glob(data + '/fineweb_train_*.bin')), data)
    add('FineWeb validation data', bool(glob.glob(data + '/fineweb_val_*.bin')), data)
print(json.dumps({'checks': checks, 'ready': all(x['ok'] for x in checks)}))
'''


def check(config):
    c = validate(config)
    command = shlex.split(c['command'])
    payload = json.dumps({'profile': c['profile'], 'argv': command})
    # Match uv's project selection on both local and SSH targets, without installing packages.
    options = [] if command[:2] == ['uv', 'run'] and '--no-project' not in command else ['--no-project']
    if command[:2] == ['uv', 'run'] and '--project' in command:
        i = command.index('--project')
        if i + 1 < len(command):
            options = ['--project', command[i + 1]]
    argv = ['uv', 'run', *options, '--no-sync', 'python', '-c', PROBE, payload]
    if c['target'] == 'ssh':
        argv, cwd = ssh_command(c, argv), None
    else:
        cwd = c['directory']
        if not Path(cwd).is_dir():
            return {'ready': False, 'checks': [{'name': 'Working directory', 'ok': False, 'detail': f'Not found: {cwd}'}]}
    try:
        result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=25)
        if result.returncode:
            raise ValueError((result.stderr or result.stdout)[-2000:] or f'Exit code {result.returncode}')
        report = json.loads(result.stdout.strip().splitlines()[-1])
        return {'ready': bool(report['ready']), 'checks': report['checks']}
    except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired) as e:
        return {'ready': False, 'checks': [{'name': 'Environment check', 'ok': False, 'detail': str(e)}]}


def storage(root):
    return Path(root) / 'training'


def job_state(folder):
    state = read(folder / 'state.json', {})
    if state.get('status') in ACTIVE:
        pid = read(folder / 'worker.json', {}).get('pid')
        if pid:
            try:
                os.kill(int(pid), 0)
            except ProcessLookupError:
                state = {**state, 'status': 'interrupted', 'message': 'The job worker exited without a final status. Check the log and remote host before restarting.'}
            except PermissionError:
                pass
        elif time.time() - state.get('started_at', 0) > 30:
            state = {**state, 'status': 'interrupted', 'message': 'The job worker did not start.'}
    return state


def state(root):
    base = storage(root)
    folders = (base / 'runs').glob('*') if (base / 'runs').exists() else []
    jobs = sorted((s for folder in folders if (s := job_state(folder))),
                  key=lambda s: s.get('started_at', 0), reverse=True)[:20]
    latest = jobs[0] if jobs else None
    log = ''
    if latest:
        try:
            with (base / 'runs' / latest['id'] / 'run.log').open('rb') as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - 20000))
                log = f.read().decode('utf-8', errors='replace')
        except OSError:
            pass
    return {'config': read(base / 'config.json') or defaults(root), 'source_directory': str(Path(root).parent if (Path(root) / 'import.json').exists() else Path(root)), 'jobs': jobs, 'latest': latest, 'log': log}


def save(root, config):
    c = validate(config)
    write(storage(root) / 'config.json', c)
    return c


def start(root, config):
    c = validate(config)
    base = storage(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(job['status'] in ACTIVE for job in state(root)['jobs']):
            raise ValueError('A training job is already active for this project.')
        report = check(c)
        if not report['ready']:
            return {'started': False, 'preflight': report}
        save(root, c)
        jid = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        folder = base / 'runs' / jid
        folder.mkdir(parents=True)
        write(folder / 'state.json', {'id': jid, 'status': 'queued', 'started_at': time.time(), 'config': c})
        with (folder / 'worker.log').open('w') as log:
            try:
                proc = subprocess.Popen(['uv', 'run', '--no-project', '--python', sys.executable, 'python', '-u',
                                         str(Path(__file__).resolve()), '--worker', str(folder.resolve())],
                                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                write(folder / 'worker.json', {'pid': proc.pid})
                threading.Thread(target=proc.wait, daemon=True).start()
            except OSError as e:
                write(folder / 'state.json', {'id': jid, 'status': 'failed', 'started_at': time.time(), 'message': str(e), 'config': c})
                raise ValueError(str(e)) from e
        return {'started': True, 'job': jid, 'preflight': report}


def stop(root, jid):
    if not re.fullmatch(r'\d{8}-\d{6}-[0-9a-f]{8}', str(jid)):
        raise ValueError('Invalid training job ID.')
    folder = storage(root) / 'runs' / jid
    if not folder.is_dir():
        raise ValueError('Training job not found.')
    if job_state(folder).get('status') in ACTIVE:
        (folder / 'STOP').touch()
    return {'ok': True}


# Runs on the remote host. Keeping control on stdin lets a requested stop (or a lost SSH
# connection) terminate the whole torchrun process group, rather than just the SSH client.
REMOTE_WORKER = r'''
import json, os, signal, subprocess, sys, threading, time
argv = json.loads(sys.argv[1]); cancel = threading.Event()
def control():
    sys.stdin.readline()
    cancel.set()
threading.Thread(target=control, daemon=True).start()
p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
try:
    while p.poll() is None and not cancel.wait(.2):
        pass
    if cancel.is_set() and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
        try: p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL); p.wait()
        sys.exit(130)
    sys.exit(p.wait())
finally:
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGKILL)
'''


def worker(folder):
    folder = Path(folder)
    s = read(folder / 'state.json')
    c = s['config']
    argv = shlex.split(c['command'])
    remote = c['target'] == 'ssh'
    if remote:
        argv = ssh_command(c, ['uv', 'run', '--no-project', 'python', '-u', '-c', REMOTE_WORKER, json.dumps(argv)])
    proc = None
    status, message = 'failed', ''
    try:
        with (folder / 'run.log').open('w') as log:
            log.write(f"Target: {c['target']} {c['host'] if remote else 'this computer'}\nDirectory: {c['directory']}\nCommand: {c['command']}\n\n")
            log.flush()
            proc = subprocess.Popen(argv, cwd=None if remote else c['directory'], stdin=subprocess.PIPE,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                    env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            write(folder / 'state.json', {**s, 'status': 'running', 'process_pid': proc.pid})
            deadline = time.monotonic() + c['timeout_minutes'] * 60
            reason = None
            while proc.poll() is None:
                reason = 'stopped' if (folder / 'STOP').exists() else 'timed_out' if time.monotonic() >= deadline else None
                if reason:
                    write(folder / 'state.json', {**s, 'status': 'stopping'})
                    if remote:
                        try:
                            proc.stdin.write(b'stop\n'); proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            pass
                    else:
                        os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        if remote:
                            reason = 'connection_lost'
                    break
                time.sleep(.2)
            rc = proc.wait()
            status = 'connection_lost' if remote and rc == 255 else reason or ('completed' if rc == 0 else 'failed')
            if status == 'connection_lost':
                message = 'Remote completion is unconfirmed. Inspect the host before starting another job.'
            else:
                message = {'completed': 'Training command finished successfully. Benchmark acceptance has not been evaluated.',
                           'failed': f'Training command exited with code {rc}.', 'stopped': 'Training was stopped.',
                           'timed_out': 'Training reached the configured time limit.'}.get(status, '')
    except Exception as e:
        message = str(e)
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        if proc and proc.stdin:
            proc.stdin.close()
        write(folder / 'state.json', {**s, 'status': status, 'finished_at': time.time(),
                                     'returncode': proc.returncode if proc else None, 'message': message})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', required=True)
    worker(ap.parse_args().worker)

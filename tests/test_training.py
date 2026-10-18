import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('training', Path(__file__).resolve().parents[1] / 'webui/training.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def config(root, code='print("training fixture complete", flush=True)'):
    return dict(profile='custom', target='local', host='', directory=str(root), timeout_minutes=1,
                command=shlex.join(['uv', 'run', '--no-project', 'python', '-u', '-c', code]))


def wait_for(root, statuses, seconds=12):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = training.state(root)
        if s['latest'] and s['latest']['status'] in statuses:
            return s
        time.sleep(.1)
    raise AssertionError(training.state(root))


class TrainingTests(unittest.TestCase):
    def test_validation_and_ssh_quoting(self):
        c = config('/tmp')
        c.update(target='ssh', host='user@gpu-box', directory='~/project with spaces')
        parsed = training.validate(c)
        argv = training.ssh_command(parsed, ['uv', 'run', 'train.py', 'a; echo wrong'])
        self.assertIn('"$HOME"/\'project with spaces\'', argv[-1])
        self.assertIn("'a; echo wrong'", argv[-1])
        for host in ['-oProxyCommand=bad', 'host;echo bad', 'host\ncommand']:
            with self.assertRaises(ValueError):
                training.validate({**c, 'host': host})
        with self.assertRaises(ValueError):
            training.validate({**c, 'timeout_minutes': 0})
        with self.assertRaises(ValueError):
            training.stop('/tmp', '../escape')

    def test_preflight_blocks_missing_directory(self):
        report = training.check(config('/does-not-exist/turing'))
        self.assertFalse(report['ready'])

    def test_start_complete_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config(tmp)
            self.assertTrue(training.check(c)['ready'])
            result = training.start(tmp, c)
            self.assertTrue(result['started'])
            s = wait_for(tmp, {'completed', 'failed'})
            self.assertEqual(s['latest']['status'], 'completed', s)
            self.assertIn('training fixture complete', s['log'])
            self.assertEqual(training.read(Path(tmp)/'training/config.json')['command'], c['command'])
            self.assertEqual(training.state(tmp)['latest']['id'], result['job'])

    def test_duplicate_stop_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = config(tmp, 'import time; print("started", flush=True); time.sleep(60)')
            result = training.start(tmp, c)
            wait_for(tmp, {'running'})
            with self.assertRaises(ValueError):
                training.start(tmp, c)
            training.stop(tmp, result['job'])
            s = wait_for(tmp, {'stopped', 'failed'})
            self.assertEqual(s['latest']['status'], 'stopped', s)
            training.start(tmp, config(tmp, 'raise SystemExit(7)'))
            s = wait_for(tmp, {'failed'})
            self.assertEqual(s['latest']['returncode'], 7)

    def test_remote_control_stops_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp)/'child.pid'
            child = 'import time; time.sleep(60)'
            code = ('import subprocess,time; from pathlib import Path; '
                    f'p=subprocess.Popen(["uv","run","--no-project","python","-c",{child!r}]); '
                    f'Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)')
            command = ['uv', 'run', '--no-project', 'python', '-c', code]
            proc = subprocess.Popen(['uv', 'run', '--no-project', 'python', '-u', '-c', training.REMOTE_WORKER,
                                     json.dumps(command)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if pidfile.exists():
                        break
                    time.sleep(.05)
                self.assertTrue(pidfile.exists())
                proc.stdin.write(b'stop\n'); proc.stdin.flush()
                self.assertEqual(proc.wait(timeout=10), 130)
            finally:
                if proc.poll() is None:
                    proc.kill(); proc.wait()
                proc.stdin.close(); proc.stderr.close()

    def test_recent_jobs_order_uses_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            for jid, started in [('20260101-000000-ffffffff', 100.1), ('20260101-000000-00000000', 100.2)]:
                training.write(Path(tmp)/'training/runs'/jid/'state.json',
                               {'id': jid, 'started_at': started, 'status': 'completed'})
            self.assertEqual(training.state(tmp)['latest']['started_at'], 100.2)

    def test_worker_enforces_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'training/runs/20260101-000000-00000000'
            c = config(tmp, 'import time; time.sleep(60)')
            # Bypass form validation to exercise expiration immediately.
            c['timeout_minutes'] = 0
            training.write(folder/'state.json', {'id': folder.name, 'started_at': time.time(), 'config': c})
            training.worker(folder)
            self.assertEqual(training.read(folder/'state.json')['status'], 'timed_out')

    def test_failed_preflight_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(training, 'check', return_value={'ready': False, 'checks': []}), \
                 patch.object(training.subprocess, 'Popen') as launch:
                self.assertFalse(training.start(tmp, config(tmp))['started'])
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()

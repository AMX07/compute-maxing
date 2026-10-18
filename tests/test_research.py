import csv
import importlib.util
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('research', Path(__file__).resolve().parents[1] / 'webui/research.py')
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


FIXTURE = '''import argparse,json,os,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--parameters'); p.add_argument('--output'); a=p.parse_args()
c=json.loads(Path(a.parameters).read_text()); out=Path(a.output)
print('running fixture', flush=True)
if c.get('sleep'):
    child='import time; from pathlib import Path; p=Path('+repr(str(out.parent/'heartbeat'))+'); '+ \\
          '\\nwhile True: p.write_text(str(time.time())); time.sleep(.05)'
    subprocess.Popen(['uv','run','--no-project','python','-c',child])
    (out.parent/'started').write_text(str(os.getpid()))
    time.sleep(60)
if c.get('nonzero'): raise SystemExit(7)
value = 'not numeric' if c.get('bad_metric') else c['x'] ** 2
out.write_text(json.dumps({'loss':value,'evaluated_seed':c['seed']}))
print('fixture complete', flush=True)
'''


def fixture(root, fixed=None, candidates=2):
    root = Path(root)
    source = root / 'source'
    source.mkdir()
    (source / 'fixture.py').write_text(FIXTURE)
    data = {'enabled': True, 'source_directory': str(source),
            'command': ['uv', 'run', '--no-project', 'python', '-u', 'fixture.py',
                        '--parameters', '{parameters}', '--output', '{output}'],
            'metric': 'loss', 'goal': 'min', 'max_experiments': candidates,
            'experiment_timeout_minutes': 1, 'objective': 'Reduce loss under a fixed seed.',
            'parameters': {'x': {'type': 'int', 'min': 1, 'max': 5, 'default': 3}},
            'fixed_parameters': {'seed': 42, **(fixed or {})}, 'readable': ['fixture.py']}
    research.write(root / 'project.json', {'research': data})
    return data


def engine(root, chat, fixed=None, candidates=2):
    data = fixture(root, fixed=fixed, candidates=candidates)
    folder = Path(root) / 'research/runs/20260101-000000-aabbccdd'
    research.write(folder / 'state.json', {'id': folder.name, 'started_at': time.time(), 'status': 'queued',
                                         'config': research.configuration(data)})
    research.write(folder / 'adapter.json', data)
    research.write(folder / 'project.json', {'root': str(root)})
    research.write(folder / 'worker.json', {'pid': os.getpid()})
    return research.Engine(folder, chat=chat)


def wait_until(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.05)
    raise AssertionError('Timed out waiting for fixture state.')


class ResearchTests(unittest.TestCase):
    def test_end_to_end_actions_metrics_and_no_reasoning_in_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            actions = iter([
                {'action': 'read_file', 'path': 'fixture.py'},
                {'action': 'delegate', 'tasks': ['Suggest a useful first value.', 'Check the configured objective.']},
                {'action': 'run_experiment', 'parameters': {'x': 2}, 'title': 'Try two', 'rationale': 'Test a lower value.'},
                {'action': 'run_experiment', 'parameters': {'x': 1}, 'title': 'Try one', 'rationale': 'Further reduce the parameter.'}])

            def chat(messages, role, tag, **kwargs):
                if role == 'researcher':
                    return json.dumps({'recommendation': '<think>PRIVATE_THOUGHT</think>Try x=2.', 'parameters': {'x': 2}})
                return '<think>PRIVATE_THOUGHT</think>' + json.dumps(next(actions))

            run = engine(tmp, chat)
            original = (Path(tmp) / 'source/fixture.py').read_bytes()
            run.run()
            state = research.state(tmp)
            self.assertEqual(state['latest']['status'], 'completed', state)
            self.assertEqual(state['latest']['completed_experiments'], 2)
            self.assertEqual(state['latest']['best_score'], 1)
            self.assertEqual([r['score'] for r in run.results], [9, 4, 1])
            self.assertEqual({r['parameters']['seed'] for r in run.results}, {42})
            kinds = {e['type'] for e in state['events']}
            self.assertEqual(kinds, {'lifecycle', 'model', 'tool', 'delegation', 'experiment'})
            self.assertNotIn('PRIVATE_THOUGHT', json.dumps(state))
            self.assertIn('advisor-1', {e['actor'] for e in state['events']})
            self.assertTrue(all('operation_id' in e['data'] for e in state['events']))
            delegation = [e for e in state['events'] if e['type'] == 'delegation']
            self.assertEqual(delegation[0]['data']['tasks'],
                             ['Suggest a useful first value.', 'Check the configured objective.'])
            self.assertEqual(len(delegation[1]['data']['answers']), 2)
            self.assertEqual(delegation[1]['data']['answers'][0],
                             {'advisor': 'advisor-1', 'recommendation': 'Try x=2.', 'parameters': {'x': 2}})
            self.assertEqual((Path(tmp) / 'source/fixture.py').read_bytes(), original)
            best = research.read(Path(tmp) / 'research/best.json')
            self.assertEqual(best['parameters'], {'x': 1, 'seed': 42})
            with (Path(tmp) / 'results.tsv').open() as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r['comparison_group'] for r in rows}, {run.job['id']})
            self.assertTrue(all(r['status'] == 'keep' for r in rows))
            self.assertTrue(all((Path(tmp) / 'runs' / r['run_id'] / 'metrics.json').is_file() for r in rows))
            research.record_result(tmp, run.adapter, run.results[-1], run.job['id'])
            self.assertEqual(len((Path(tmp) / 'results.tsv').read_text().splitlines()), 4)

    def test_rejected_actions_cannot_change_fixed_parameters_or_read_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = iter([
                {'action': 'read_file', 'path': '../../outside'},
                {'action': 'run_experiment', 'parameters': {'seed': 999, 'x': 1}},
                {'action': 'run_experiment', 'parameters': {'x': 1.2}},
                {'action': 'run_experiment', 'parameters': {'x': 4}, 'title': 'Worse candidate'}])
            run = engine(tmp, lambda *args, **kwargs: json.dumps(next(calls)), candidates=1)
            run.run()
            self.assertEqual(run.job['status'], 'completed')
            self.assertEqual([r['status'] for r in run.results], ['keep', 'discard'])
            self.assertEqual(run.best, {'x': 3, 'seed': 42})
            self.assertEqual(len([e for e in research.state(tmp)['events'] if e['title'] == 'Action rejected']), 3)

    def test_invalid_metric_and_nonzero_exit_never_accept_or_call_model(self):
        for fixed in ({'bad_metric': True}, {'nonzero': True}):
            with self.subTest(fixed=fixed), tempfile.TemporaryDirectory() as tmp:
                def chat(*args, **kwargs):
                    self.fail('Model must not run when baseline fails.')
                run = engine(tmp, chat, fixed=fixed)
                run.run()
                self.assertEqual(run.job['status'], 'failed')
                self.assertIsNone(run.results[0]['score'])
                self.assertEqual(run.results[0]['status'], 'crash')
                self.assertFalse((Path(tmp) / 'research/best.json').exists())

    def test_protocol_change_is_rejected_before_candidate_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            def chat(*args, **kwargs):
                source = Path(tmp) / 'source/fixture.py'
                source.write_text(source.read_text() + '\n# outside edit\n')
                return json.dumps({'action': 'run_experiment', 'parameters': {'x': 1}})
            run = engine(tmp, chat, candidates=1)
            run.run()
            self.assertEqual(run.job['status'], 'failed')
            self.assertIn('source or evaluation settings changed', run.job['message'])
            self.assertEqual(len(run.results), 1)

    def test_stop_during_model_call_returns_promptly(self):
        with tempfile.TemporaryDirectory() as tmp:
            entered, release = threading.Event(), threading.Event()
            def chat(*args, **kwargs):
                entered.set()
                release.wait(10)
                return '{}'
            run = engine(tmp, chat, candidates=1)
            thread = threading.Thread(target=run.run)
            thread.start()
            try:
                self.assertTrue(entered.wait(8))
                research.stop(tmp, run.job['id'])
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(run.job['status'], 'stopped')
                events = research.state(tmp)['events']
                self.assertTrue(any(e['type'] == 'model' and e['phase'] == 'stopped' for e in events))
                self.assertEqual(events[-1]['phase'], 'stopped')
            finally:
                release.set()
                thread.join(5)

    def test_detached_worker_stop_duplicate_and_process_group_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, fixed={'sleep': True}, candidates=1)
            with patch.object(research.llm, 'config', return_value={}):
                launched = research.start(tmp, {})
                folder = Path(tmp) / 'research/runs' / launched['run_id']
                try:
                    heartbeat = wait_until(lambda: next(folder.glob('experiments/*/heartbeat'), None))
                    with self.assertRaisesRegex(ValueError, 'already active'):
                        research.start(tmp, {})
                    research.stop(tmp, launched['run_id'])
                    wait_until(lambda: research.state(tmp)['latest']['status'] == 'stopped')
                    stamp = heartbeat.read_text()
                    time.sleep(.2)
                    self.assertEqual(heartbeat.read_text(), stamp)
                    self.assertFalse(research.state(tmp)['active'])
                    result = research.read(next(folder.glob('experiments/*/result.json')))
                    self.assertEqual(result['status'], 'stopped')
                    self.assertIsNone(result['score'])
                    events = research.state(tmp)['events']
                    experiment = [e for e in events if e['type'] == 'experiment'][-1]
                    self.assertEqual(experiment['phase'], 'stopped')
                    self.assertEqual(experiment['data']['status'], 'stopped')
                    self.assertIsNone(experiment['data']['score'])
                    self.assertEqual(events[-1]['phase'], 'stopped')
                finally:
                    research.stop(tmp, launched['run_id'])

    def test_configuration_and_start_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(research.state(tmp)['supported'])
            data = fixture(tmp)
            for request in ({'max_experiments': 21}, {'max_experiments': 1.2}, {'objective': ''},
                            {'command': ['arbitrary']}, {'experiment_timeout_minutes': True}):
                with self.subTest(request=request), self.assertRaises(ValueError):
                    research.configuration(data, request)
            for proposal in ({'x': True}, {'x': 0}, {'seed': 7}, {'unknown': 1}):
                with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                    research.validate_parameters(data, proposal)
            with patch.object(research.llm, 'config', side_effect=SystemExit('Missing API configuration.')), \
                    patch.object(research.subprocess, 'Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'Missing API'):
                    research.start(tmp, {})
                launch.assert_not_called()
            with self.assertRaises(ValueError):
                research.stop(tmp, '../unsafe')

    def test_training_conflict_and_unexpected_worker_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            folder = Path(tmp) / 'training/runs/20260101-000000-aabbccdd'
            research.write(folder / 'state.json', {'id': folder.name, 'status': 'running', 'started_at': time.time()})
            research.write(folder / 'worker.json', {'pid': os.getpid()})
            with patch.object(research.llm, 'config', return_value={}):
                with self.assertRaisesRegex(ValueError, 'standalone training'):
                    research.start(tmp, {})
            research.write(Path(tmp) / 'research/runs/20260101-000000-aabbccdd/state.json',
                           {'id': folder.name, 'status': 'running', 'started_at': time.time() - 60})
            state = research.state(tmp)
            self.assertEqual(state['latest']['status'], 'interrupted')
            self.assertFalse(state['active'])

    def test_experiment_timeout_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = engine(tmp, lambda *args, **kwargs: '{}', fixed={'sleep': True})
            run.config['experiment_timeout_minutes'] = 0
            run.run()
            self.assertEqual(run.job['status'], 'failed')
            self.assertIn('time limit', run.job['message'])

    def test_protocol_files_are_immutable_without_being_readable_by_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = engine(tmp, lambda *args, **kwargs: '{}', candidates=1)
            dataset = Path(tmp) / 'source/data.bin'
            dataset.write_bytes(b'fixed data')
            run.adapter['protocol_files'] = ['data.bin']
            run.protocol = research.protocol(run.adapter)
            with self.assertRaisesRegex(ValueError, 'allowlisted'):
                run.read_source('data.bin')
            dataset.write_bytes(b'changed data')
            run.run()
            self.assertEqual(run.job['status'], 'failed')
            self.assertIn('source or evaluation settings changed', run.job['message'])
            self.assertEqual(run.results, [])

    def test_orphaned_evaluator_blocks_restart_and_can_be_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, fixed={'sleep': True}, candidates=1)
            with patch.object(research.llm, 'config', return_value={}):
                launched = research.start(tmp, {})
                folder = Path(tmp) / 'research/runs' / launched['run_id']
                try:
                    heartbeat = wait_until(lambda: next(folder.glob('experiments/*/heartbeat'), None))
                    worker = research.read(folder / 'worker.json')['pid']
                    os.kill(worker, signal.SIGKILL)
                    wait_until(lambda: research.state(tmp)['latest']['status'] == 'orphaned')
                    self.assertTrue(research.state(tmp)['active'])
                    with self.assertRaisesRegex(ValueError, 'already active'):
                        research.start(tmp, {})
                    research.stop(tmp, launched['run_id'])
                    self.assertEqual(research.state(tmp)['latest']['status'], 'stopped')
                    stamp = heartbeat.read_text()
                    time.sleep(.2)
                    self.assertEqual(heartbeat.read_text(), stamp)
                finally:
                    research.stop(tmp, launched['run_id'])

    def test_reused_evaluator_pid_is_not_treated_as_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = engine(tmp, lambda *args, **kwargs: '{}')
            (run.folder / 'worker.json').unlink()
            run.update(started_at=time.time() - 60)
            research.write(run.folder / 'experiments/old/process.json',
                           {'pid': os.getpid(), 'identity': 'not the current process identity'})
            self.assertEqual(research.job_state(run.folder)['status'], 'interrupted')
            self.assertEqual(research.orphan_processes(run.folder), [])

    def test_reused_worker_pid_reports_interrupted_and_still_checks_for_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = engine(tmp, lambda *args, **kwargs: '{}')
            research.write(run.folder / 'worker.json', {'pid': os.getpid(), 'identity': 'original worker identity'})
            with patch.object(research, 'process_identity', return_value='new process identity'):
                self.assertEqual(research.job_state(run.folder)['status'], 'interrupted')
                self.assertFalse(research.state(tmp)['active'])
                research.write(run.folder / 'experiments/live/process.json',
                               {'pid': os.getpid(), 'identity': 'new process identity'})
                state = research.state(tmp)
                self.assertEqual(state['latest']['status'], 'orphaned')
                self.assertTrue(state['active'])

    def test_architecture_metric_and_goal_must_match_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            for metric in ({'name': 'different_loss'}, {'name': 'loss', 'goal': 'max'}):
                research.write(Path(tmp) / 'architecture.json', {'metric': metric})
                with self.subTest(metric=metric), self.assertRaisesRegex(ValueError, 'conflicts'):
                    research.metadata(tmp)
                self.assertFalse(research.state(tmp)['supported'])
            for metric in ({}, {'name': 'loss'}, {'name': 'loss', 'goal': 'min'}):
                research.write(Path(tmp) / 'architecture.json', {'metric': metric})
                self.assertTrue(research.state(tmp)['supported'])

    def test_planner_feedback_refreshes_advisor_and_action_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = []
            responses = iter([
                {'action': 'delegate', 'tasks': ['Suggest a candidate.']},
                {'action': 'delegate', 'tasks': ['Try another advisor.']},
                {'action': 'read_file', 'path': 'fixture.py'},
                'invalid action',
                {'action': 'read_file', 'path': 'fixture.py'},
                {'action': 'run_experiment', 'parameters': {'x': 2}}])

            def chat(messages, role, tag, **kwargs):
                if role == 'researcher':
                    return json.dumps({'recommendation': '<think>PRIVATE_THOUGHT</think>Try x=2.',
                                       'parameters': {'x': 2}})
                observed.append(json.loads(json.dumps(messages)))
                response = next(responses)
                return '<think>PRIVATE_THOUGHT</think>' + (json.dumps(response) if isinstance(response, dict) else response)

            run = engine(tmp, chat, candidates=1)
            run.advisors = 2
            params, _, _ = run.choose(1)
            self.assertEqual(params, {'x': 2, 'seed': 42})
            self.assertEqual(len(observed), research.MAX_TURNS)
            contexts = [json.loads(messages[-1]['content']) for messages in observed]
            self.assertEqual([c['advisor_calls_remaining'] for c in contexts], [1, 0, 0, 0, 0, 0])
            self.assertEqual([c['action_turns_remaining'] for c in contexts], [6, 5, 4, 3, 2, 1])
            self.assertEqual(contexts[1]['type'], 'tool_result')
            self.assertEqual(contexts[2]['type'], 'action_rejected')
            self.assertEqual(contexts[4]['type'], 'action_rejected')
            self.assertIn('Choose run_experiment now', contexts[-1]['instruction'])
            self.assertTrue(all('No advisor calls remain' in c['instruction'] for c in contexts[1:]))
            self.assertNotIn('PRIVATE_THOUGHT', json.dumps(observed))
            self.assertNotIn('PRIVATE_THOUGHT', json.dumps(research.state(tmp)['events']))


if __name__ == '__main__':
    unittest.main()

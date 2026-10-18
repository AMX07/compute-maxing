import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'webui'))
import server


class ResearchApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name
        self.registry = patch.object(server, 'projects', return_value={'fixture': {'path': self.root}})
        self.registry.start()
        self.http = server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.registry.stop()
        self.temp.cleanup()

    def post(self, action, body, origin=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.http.server_port)
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        connection.request('POST', f'/api/p/fixture/{action}', json.dumps(body), headers)
        response = connection.getresponse()
        result = response.status, json.loads(response.read())
        connection.close()
        return result

    def test_start_and_stop_forward_bounded_settings_and_run_identity(self):
        settings = {'objective': 'Improve validation accuracy', 'max_experiments': 2,
                    'experiment_timeout_minutes': 3}
        with patch.object(server.research, 'start', return_value={'started': True, 'run_id': 'session'}) as start:
            status, result = self.post('research-start', settings)
            self.assertEqual(status, 200)
            self.assertEqual(result['run_id'], 'session')
            start.assert_called_once_with(self.root, settings)
        with patch.object(server.research, 'stop', return_value={'ok': True}) as stop:
            self.assertEqual(self.post('research-stop', {'run_id': 'session'})[0], 200)
            stop.assert_called_once_with(self.root, 'session')

    def test_standalone_training_cannot_overlap_active_or_orphaned_research(self):
        with patch.object(server.research, 'state', return_value={'active': True}), \
                patch.object(server.training, 'start') as launch:
            status, result = self.post('training-start', {'config': {}})
            self.assertEqual(status, 400)
            self.assertIn('Stop', result['error'])
            launch.assert_not_called()

    def test_invalid_configuration_is_reported_without_internal_error(self):
        with patch.object(server.research, 'start', side_effect=ValueError('Only bounded parameters are supported')):
            status, result = self.post('research-start', {'command': 'unexpected'})
            self.assertEqual(status, 400)
            self.assertEqual(result['error'], 'Only bounded parameters are supported')

    def test_cross_origin_request_never_launches_research(self):
        with patch.object(server.research, 'start') as launch:
            self.assertEqual(self.post('research-start', {}, 'https://unrelated.example')[0], 403)
            launch.assert_not_called()


class MetricSummaryTests(unittest.TestCase):
    def test_best_score_respects_goal_and_comparison_series(self):
        for goal, expected, curve in [('min', 0, [0, 0, 0, 0]), ('max', 0.7, [0, 0.9, 0, 0.7])]:
            with self.subTest(goal=goal), tempfile.TemporaryDirectory() as root:
                path = Path(root)
                (path / 'architecture.json').write_text(json.dumps({'metric': {'name': 'accuracy', 'goal': goal}}))
                (path / 'results.tsv').write_text(
                    'commit\taccuracy\tmemory_gb\tstatus\trun_id\tdescription\tcomparison_group\n'
                    '-\t0\t\tkeep\t\tbaseline\tfirst\n'
                    '-\t0.9\t\tkeep\t\tcandidate\tfirst\n'
                    '-\t0\t\tkeep\t\tbaseline\tsecond\n'
                    '-\t0.7\t\tkeep\t\tcandidate\tsecond\n')
                state = server.build_state(root)
                card = server.project_card('fixture', {'path': root})
                self.assertEqual(state['summary']['baseline_val_loss'], 0)
                self.assertEqual(state['summary']['best_val_loss'], expected)
                self.assertIsNone(state['experiments'][2]['best_before'])
                self.assertEqual(card['best'], expected)
                self.assertEqual(card['curve'], curve)
                self.assertEqual(card['goal'], goal)

    def test_research_metric_without_architecture_map(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            (path / 'project.json').write_text(json.dumps({'research': {'enabled': True, 'metric': 'accuracy', 'goal': 'max'}}))
            (path / 'results.tsv').write_text('commit\taccuracy\tmemory_gb\tstatus\trun_id\tdescription\n-\t0.1\t\tkeep\t\tbaseline\n-\t0.5\t\tkeep\t\tcandidate\n')
            state = server.build_state(root)
            self.assertEqual(state['metric'], 'accuracy')
            self.assertEqual(state['goal'], 'max')
            self.assertEqual(state['summary']['best_val_loss'], .5)
            self.assertEqual(server.project_card('fixture', {'path': root})['goal'], 'max')

    def test_crash_zero_is_not_a_valid_score(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'results.tsv').write_text('commit\tval_loss\tmemory_gb\tstatus\trun_id\tdescription\n-\t0\t\tcrash\t\tfailed\n')
            self.assertIsNone(server.read_results(root)[0]['val_loss'])


if __name__ == '__main__':
    unittest.main()

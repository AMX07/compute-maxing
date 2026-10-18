import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).resolve().parents[1]/'webui/evaluation.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

class EvaluationTests(unittest.TestCase):
    def test_reads_constants_without_executing_project(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'prepare.py'
            path.write_text('raise RuntimeError("must never run")\nTIME_BUDGET=300\nEVAL_TOKENS=40*524288\nMAX_SHARD=6542\nVAL_SHARD=MAX_SHARD\ndef evaluate_bpb(model):\n    """Bits per byte."""\n    return total_nats / (math.log(2) * total_bytes)\n')
            result=evaluation.describe(root,'val_bpb',{})
            facts={f['label']:f['value'] for f in result['facts']}
            self.assertEqual(facts['Evaluation token budget'],20971520)
            self.assertEqual(facts['Pinned validation shard'],6542)
            self.assertEqual(result['description'],'Bits per byte.')
            self.assertIn('target bytes', result['formula'])
            self.assertEqual(len(result['source']['sha256']),64)

    def test_missing_and_unrecognized_evaluators_do_not_invent_protocols(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(evaluation.describe(root,'custom_score',{})['source'])
            (Path(root)/'prepare.py').write_text('def evaluate_bpb(x):\n    return x\n')
            self.assertEqual(evaluation.describe(root,'custom_score',{})['facts'],[])

    def test_custom_project_metadata_needs_no_adapter(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'evaluation.json').write_text(json.dumps({
                'title':'Validation accuracy', 'description':'Held-out classification accuracy',
                'facts':[{'label':'Examples','value':10000}, {'wrong':'ignored'}],
                'formula':'correct / total'}))
            result=evaluation.describe(root,'accuracy',{'metric':{'goal':'max'}})
            self.assertEqual(result['title'],'Validation accuracy')
            self.assertEqual(result['facts'],[{'label':'Examples','value':10000,'unit':''}])
            self.assertEqual(result['metadata_file'],'evaluation.json')

    def test_history_remains_attributed_and_distinguishes_retimings(self):
        result=evaluation.describe('/unused','training_seconds',{'history_only':True,'summary':'Published speed records'})
        self.assertEqual(result['description'],'Published speed records')
        self.assertIn('not been independently reproduced',result['note'])
        self.assertIn('Re-timings',result['note'])
        self.assertIsNone(result['source'])

if __name__=='__main__': unittest.main()

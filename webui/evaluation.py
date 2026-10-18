"""Read evaluation documentation and constants without importing project code."""
import ast
import hashlib
import json
from pathlib import Path

FIELDS = {
    'TIME_BUDGET': ('Training budget', 'seconds'),
    'EVAL_TOKENS': ('Evaluation token budget', 'tokens'),
    'MAX_SEQ_LEN': ('Evaluation context', 'tokens'),
    'VAL_SHARD': ('Pinned validation shard', ''),
    'EVAL_IMAGES': ('Held-out images', 'images'),
    'EVAL_TIMESTEPS': ('Timesteps per image', ''),
    'INFERENCE_TIME_LIMIT': ('Sampling time limit', 'seconds'),
}


def constant(node, values):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return node.value
    if isinstance(node, ast.Name):
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        a, b = constant(node.left, values), constant(node.right, values)
        if type(a) in (int, float) and type(b) in (int, float):
            return a * b
    raise ValueError('Not a supported constant')


def describe(root, metric, arch):
    arch = arch or {}
    result = {'description': 'No evaluator documentation is available for this project.',
              'facts': [], 'source': None, 'formula': None,
              'note': 'Scores come from results.tsv. Missing measurements are shown as unavailable.'}
    # Projects can document any evaluator without adding a UI-specific adapter.
    metadata_path = Path(root) / 'evaluation.json'
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError, UnicodeError):
        metadata = None
    if isinstance(metadata, dict):
        for key in ('title', 'description', 'formula', 'note'):
            if isinstance(metadata.get(key), str) and metadata[key].strip():
                result[key] = metadata[key]
        result['facts'] = [
            {'label': fact['label'], 'value': fact['value'], 'unit': fact.get('unit', '')}
            for fact in metadata.get('facts', []) if isinstance(fact, dict)
            and isinstance(fact.get('label'), str) and type(fact.get('value')) in (str, int, float)
            and isinstance(fact.get('unit', ''), str)
        ] if isinstance(metadata.get('facts'), list) else []
        result['metadata_file'] = 'evaluation.json'
        if arch.get('history_only'):
            result['note'] += ' Published records are not independently reproduced by Turing; re-timings start new comparison series.'
        return result
    if arch.get('history_only'):

        result.update(description=arch.get('summary') or 'Published benchmark history.',
                      note='Published results have not been independently reproduced by Turing. Re-timings start new comparison series; compare scores within the same series.')
        return result
    path = Path(root) / 'prepare.py'
    try:
        source = path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError):
        return result
    name = {'val_bpb': 'evaluate_bpb', 'val_loss': 'evaluate_loss'}.get(metric)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if function is None:
        return result
    values = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = constant(n.value, values)
            except (ValueError, KeyError, TypeError, OverflowError):
                pass
    result.update(description=ast.get_docstring(function) or 'The project provides no docstring for this evaluator.',
                  source={'file': 'prepare.py', 'function': name, 'line': function.lineno,
                          'sha256': hashlib.sha256(source.encode()).hexdigest()},
                  note='Protocol details describe prepare.py in the current checkout. Historical runs may have used different settings; scores are read from results.tsv.')
    result['facts'] = [{'label': label, 'value': values[key], 'unit': unit} for key, (label, unit) in FIELDS.items() if key in values]
    if name == 'evaluate_bpb' and 'total_nats / (math.log(2) * total_bytes)' in ast.get_source_segment(source, function):
        result['formula'] = 'bits per byte = total token loss (nats) / (ln(2) × target bytes)'
    return result

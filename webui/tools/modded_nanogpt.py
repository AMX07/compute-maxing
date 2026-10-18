"""Import modded-nanogpt's published track-1 history into a separate Turing view.

No trainer code is executed, no LLM calls are made, and upstream files are not edited.
Run with uv run --no-project python webui/tools/modded_nanogpt.py /path/to/modded-nanogpt --register.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings
from urllib.parse import quote, urlparse

UPSTREAM = "https://github.com/KellerJordan/modded-nanogpt"
OWNER = "turing-modded-nanogpt-v1"
LINK = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
CATEGORIES = [
    ("embedding", "Embeddings & output", r"embed|lm.head|logit|softcap|bigram|ngram|n-gram"),
    ("attention", "Attention", r"attn|attention|rotary|qk|rope|yarn|window|key.offset|xsa|mha"),
    ("mlp", "MLP & depth", r"mlp|relu|depth"),
    ("residual", "Residual connections", r"skip|residual|shortcut|u-net|hyperconnection|mudd|backout|smear"),
    ("optimizer", "Optimizers", r"muon|adam|optimizer|anvil|polar.express|momentum|weight.decay|cwd|newton"),
    ("schedule", "Data & schedules", r"batch|cooldown|warmup|learning.rate|\blr\b|schedule|token.prediction|seq.len|eos|document|prefix"),
    ("systems", "Precision & systems", r"fp8|bfloat|bf16|precision|torch|triton|kernel|fuse|fused|comms|communication|distributed|all.reduce|reduce.scatter|cuda|transpose|async|overhead|compile"),
]


def plain(text):
    return re.sub(r"\s+", " ", LINK.sub(lambda m: m[1], text).replace('`', '')).strip()


def duration(text):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s)\s*", text, re.I)
    if not m or float(m[1]) <= 0:
        raise ValueError(f"Unsupported record duration: {text!r}")
    return round(float(m[1]) * (60 if m[2].lower().startswith('m') else 1), 6)


def parse_records(text):
    try:
        section = text.split('## World record history', 1)[1].split('## Rules', 1)[0]
    except IndexError as e:
        raise ValueError('World record history section not found') from e
    rows, seen, era = [], set(), 0
    for line in section.splitlines():
        if not re.match(r'^\s*\|?\s*\d+\s*\|', line):
            continue
        fields = [f.strip() for f in line.strip().strip('|').split('|')]
        if len(fields) != 6:
            raise ValueError(f'Expected six history columns: {line}')
        number, timing, description, date, evidence, contributors = fields
        number = int(number)
        kind = 'retime' if number in seen else 'record'
        if kind == 'retime':
            era += 1
        seen.add(number)
        rows.append(dict(number=number, seconds=duration(timing), description=plain(description),
                         date=datetime.strptime(date, '%m/%d/%y').date().isoformat(),
                         contributors=plain(contributors), kind=kind, comparison_group=f'timing-{era}',
                         links=LINK.findall(description + ' ' + evidence),
                         log_paths=[url for _, url in LINK.findall(evidence) if url.startswith('records/')]))
    if not rows:
        raise ValueError('No published records found')
    return rows


def local_file(repo, relative):
    path = (repo / relative).resolve()
    return path if path.is_relative_to(repo.resolve()) and path.is_file() else None


def snapshot(repo, record):
    """Only accept syntactically valid Python from the exact linked log's source block.

    Never substitute the current trainer for missing historical code. Bundled multi-file
    logs or directory links remain evidence links if no unambiguous source is available.
    """
    for relative in record['log_paths']:
        path = local_file(repo, relative)
        if not path:
            continue
        text = path.read_text(errors='replace')
        chunks = re.split(r'(?m)^={20,}\s*$', text)
        for chunk in chunks[:3]:
            code = chunk.strip()
            if not code or not re.search(r'(?m)^(?:import torch|from torch import)', code):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(code)
            except SyntaxError:
                continue
            return code + '\n', relative
    return None, None


def source_url(url, revision):
    if url.startswith('records/'):
        return f'{UPSTREAM}/blob/{revision}/{quote(url, safe="/")}'
    parts = urlparse(url)
    return url if parts.scheme in ('http', 'https') and parts.netloc else None


def categories(description):
    ids = [cid for cid, _, pattern in CATEGORIES if re.search(pattern, description, re.I)]
    return ids or ['other']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def import_history(repo, output):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    # Read the pinned commit's README, not a possibly edited working-tree table.
    readme = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:README.md'], text=True)
    records = parse_records(readme)
    marker = output / 'import.json'
    if output.exists() and any(output.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()).get('importer') != OWNER:
            raise ValueError(f'Refusing to overwrite a non-import project: {output}')
    output.mkdir(parents=True, exist_ok=True)
    license_text = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:LICENSE'], text=True)
    (output / 'LICENSE').write_text(license_text)
    changes, manifest, table = {}, [], []
    previous, previous_record, previous_era = None, None, None
    snapshots = diffs = 0
    for idx, record in enumerate(records):
        rid = f"record-{record['number']:03d}-{record['date']}"
        run = output / 'runs' / rid
        run.mkdir(parents=True, exist_ok=True)
        links = [{'label': label or 'Source', 'url': source_url(url, revision)} for label, url in record['links']]
        links = [link for link in links if link['url']]
        links.insert(0, {'label': 'Published record table', 'url': f'{UPSTREAM}/blob/{revision}/README.md#world-record-history'})
        code, code_path = snapshot(repo, record)
        # A local modified log is not a snapshot of the pinned upstream commit.
        if code_path:
            tracked = subprocess.check_output(['git', '-C', str(repo), 'show', f'{revision}:{code_path}'])
            if tracked != (repo / code_path).read_bytes():
                code, code_path = None, None
        for name in ('train.py', 'change.diff'):
            (run / name).unlink(missing_ok=True)
        provenance = {k: record[k] for k in ('number', 'date', 'contributors', 'kind', 'comparison_group')}
        provenance.update(sources=links, upstream_revision=revision, source_snapshot=code_path,
                          evidence='Published leaderboard entry; not independently rerun by Turing.',
                          comparison_note='Re-timings start a new comparison series. Times across timing rules or software versions are not controlled comparisons.')
        if code:
            snapshots += 1
            (run / 'train.py').write_text(code)
            if previous and record['kind'] != 'retime' and previous_era == record['comparison_group']:
                diff = ''.join(difflib.unified_diff(previous.splitlines(True), code.splitlines(True),
                                                   fromfile=f'published-record-{previous_record}/train.py',
                                                   tofile=f'published-record-{record["number"]}/train.py'))
                if diff:
                    (run / 'change.diff').write_text(diff)
                    provenance['diff_from_record'] = previous_record
                    diffs += 1
            previous, previous_record, previous_era = code, record['number'], record['comparison_group']
        else:
            previous = None  # do not silently bridge missing historical source
        ids = ['timing'] if record['kind'] == 'retime' else categories(record['description'])
        primary = ids[0] if len(ids) == 1 else 'combined'
        note = ('Re-measurement of the same record; no new architecture claim.' if record['kind'] == 'retime' else
                'Category inferred from the published description. This entry may bundle multiple changes; it is not an isolated ablation.')
        if idx:
            changes[str(idx)] = {'component': primary, 'label': record['description'], 'after': record['description'],
                                 'related_components': ids, 'note': note}
        write_json(run / 'metrics.json', {'training_seconds': record['seconds'], 'target_val_loss': 3.28,
                                         'created_at': record['date'] + 'T00:00:00'})
        write_json(run / 'idea.json', {'title': record['description'], 'category': primary, 'provenance': provenance})
        table.append(['', record['seconds'], '', 'retime' if record['kind'] == 'retime' else 'keep', rid,
                      record['description'], record['comparison_group']])
        manifest.append({**provenance, 'run_id': rid, 'training_seconds': record['seconds']})
    components = [{'id': cid, 'name': name, 'group': 'changes',
                   'baseline': {'label': 'llm.c baseline', 'detail': 'History category, inferred from record descriptions; not a complete architecture specification.'}}
                  for cid, name in [(c[0], c[1]) for c in CATEGORIES] +
                  [('combined', 'Combined changes'), ('timing', 'Timing protocol'), ('other', 'Other changes')]]
    write_json(output / 'architecture.json', {
        'name': 'modded-nanogpt · 8×H100 speedrun', 'history_only': True,
        'summary': 'Published track-1 speed records: minimize training seconds while reaching ≤3.28 validation loss on FineWeb with 8 H100s. '
                   'This is a curated record history, not all attempted experiments. Categories are inferred from descriptions; Latest shows the most recent milestone in each category, not a reconstructed model. '
                   'Re-timings reset comparisons. Turing has not independently reproduced these results.',
        'metric': {'name': 'training_seconds', 'goal': 'min', 'unit': 's', 'scale': 'log'},
        'groups': [{'id': 'changes', 'name': 'Published research changes'}], 'components': components,
        'changes': changes, 'hypotheses': [],
        'execution_note': 'Published history only. Reproducing the speedrun requires 8 NVIDIA H100 GPUs and the upstream CUDA environment. '
                          'The current trainer spans train_gpt.py and track_1_short/; Turing’s single-file research loop is not configured for it.',
        'source': f'{UPSTREAM}/blob/{revision}/README.md#world-record-history',
    })
    with (output / 'results.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['commit', 'training_seconds', 'memory_gb', 'status', 'run_id', 'description', 'comparison_group'])
        writer.writerows(table)
    write_json(marker, {'importer': OWNER, 'upstream': UPSTREAM, 'revision': revision,
                        'readme_sha256': hashlib.sha256(readme.encode()).hexdigest(), 'records': manifest})
    return {'project': str(output), 'milestones': len(records), 'records': sum(r['kind'] == 'record' for r in records),
            'retimings': sum(r['kind'] == 'retime' for r in records), 'source_snapshots': snapshots, 'diffs': diffs,
            'revision': revision}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('repo', type=Path)
    ap.add_argument('--output', type=Path, help='default: <repo>/.turing')
    ap.add_argument('--register', action='store_true', help='add the generated history to Turing’s project browser')
    args = ap.parse_args()
    result = import_history(args.repo, args.output or args.repo / '.turing')
    if args.register:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from webui.server import register, project_id
        register(result['project'], opened=True)
        result['route'] = f"#/p/{project_id(result['project'])}/evolution"
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

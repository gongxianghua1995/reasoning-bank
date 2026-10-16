"""Audit saved predictions, memory visibility and API usage without model calls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .common import file_hash, load, write_json


def summarize(out):
    state = load(out / 'status.json')
    manifest = load(out / 'manifest.json')
    usage = defaultdict(Counter)
    requests, responses, errors = Counter(), Counter(), Counter()
    rows = []
    for folder in sorted((out / 'cases').glob('*')):
        identity = load(folder / 'case_identity.json')
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()] if (folder / 'events.jsonl').exists() else []
        retrieval = load(folder / 'retrieval.json')
        metrics = load(folder / 'metrics.json') if (folder / 'metrics.json').exists() else None
        judgment = load(folder / 'self_judgment.json') if (folder / 'self_judgment.json').exists() else {}
        selected = retrieval['selected_memory']
        solve_requests = [load(f) for f in sorted((folder / 'requests').glob('*.json'))]
        for event in events:
            phase = event.get('phase')
            if event['event'] == 'model_request':
                requests[phase] += 1
            if event['event'] == 'model_response':
                responses[phase] += 1
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    usage[phase][key] += event.get('usage', {}).get(key, 0) or 0
                usage[phase]['cached_prompt_tokens'] += (event.get('usage', {}).get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
            if event['event'] == 'model_error':
                errors[event['error']] += 1
        rows.append({**identity, 'metrics': metrics, 'self_judgment': judgment.get('status'),
                     'selected_memory_ids': retrieval['episode_ids'],
                     'solve_requests': len(solve_requests),
                     'memory_in_all_solve_requests': bool(solve_requests) and all(selected in m[0]['content'] for m in solve_requests) if selected else None,
                     'memory_items': len(load(folder / 'memory_entry.json')['memory_items']) if (folder / 'memory_entry.json').exists() else 0,
                     'action_format_repairs': sum(e['event'] == 'action_format_error' for e in events),
                     'empty_responses': sum(e['event'] == 'model_response' and not e.get('content') for e in events),
                     'completed_steps': sum(e['event'] == 'step_finished' for e in events)})
    changed = [name for name, sha in load(out / 'source_hashes.json').items()
               if file_hash(out / 'source_at_run' / name) != sha]
    summary = {'status': state['status'], 'elapsed_seconds': state['elapsed_seconds'],
               'mode': manifest['options']['mode'], 'scope': manifest['options']['scope'],
               'expected_cases': len(manifest['selected_case_ids']) * (manifest['options']['epochs'] if manifest['options']['mode'] == 'train' else 1),
               'cases': rows, 'model_requests': dict(requests), 'model_responses': dict(responses),
               'model_errors': dict(errors), 'returned_usage_by_phase': {p: dict(v) for p, v in usage.items()},
               'memory_entries': len(load(out / 'memory.json')['entries']),
               'changed_frozen_source': changed,
               'frozen_memory_unchanged': file_hash(out / 'memory.json') == manifest['memory_initial_sha256'] if manifest['options']['mode'] == 'eval' else None,
               'run_summary': load(out / 'summary.json') if (out / 'summary.json').exists() else None,
               'case_failures': load(out / 'failures.json') if (out / 'failures.json').exists() else [],
               'interpretation': 'Offline fixed-observation Mind2Web smoke; no claim of memory benefit from this sample.'}
    if (out / 'resume.json').exists():
        recovery = load(out / 'resume.json')
        summary['recovery'] = {k: v for k, v in recovery.items() if k != 'source_artifact_hashes'}
        summary['recovery']['usage_includes_inherited_calls'] = True
        summary['recovery']['elapsed_seconds_scope'] = 'Current process only; source runtime recorded separately.'
    write_json(out / 'audit.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.output.resolve()), ensure_ascii=False, indent=2))

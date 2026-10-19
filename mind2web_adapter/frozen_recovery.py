"""Reuse completed evaluation steps without changing observations or memory."""
from pathlib import Path
from .common import load, write_json, file_hash, digest, event


def restore_prefix(roots, folder, public, config, retrieval, identity):
    """Follow the earliest trajectory; never select a rerun by its score."""
    chosen, history, source = [], [], None
    evidence = []
    for root in map(Path, roots):
        candidate = root/'cases'/f"01_{digest(public['case_id'])[:16]}"
        if not (candidate/'predictions_local.json').exists():
            continue
        if load(root/'config.json') != config or load(root/'options.json')['mode'] != 'eval':
            raise ValueError('Evaluation recovery configuration differs')
        if (file_hash(root/'memory.json') != retrieval['memory_before_sha256']
                or load(candidate/'public_case.json') != public
                or load(candidate/'case_identity.json') != identity):
            raise ValueError('Evaluation recovery input or memory differs')
        previous = load(candidate/'retrieval.json')
        if any(previous[k] != retrieval[k] for k in
               ('episode_ids', 'selected_memory', 'scope', 'memory_before_sha256')):
            raise ValueError('Evaluation recovery retrieval differs')
        actions = load(candidate/'predictions_local.json')
        trace = load(candidate/'trajectory.json')[:len(actions)]
        if (len(actions) > len(public['observations']) or len(trace) != len(actions)
                or any(t.get('final_action') != a or t['observation'] != o
                       for t,a,o in zip(trace,actions,public['observations']))):
            raise ValueError('Evaluation recovery checkpoint inconsistent')
        # Earlier independent retries can have different predictions. Keep the
        # earliest valid prefix, accepting only extensions of that exact prefix.
        extends = len(actions) >= len(chosen) and actions[:len(chosen)] == chosen
        evidence.append({'folder':str(candidate), 'steps':len(actions), 'accepted':extends,
                         'predictions_sha256':file_hash(candidate/'predictions_local.json')})
        if extends:
            chosen, history, source = actions, trace, str(candidate)
    if chosen:
        write_json(folder/'predictions_local.json', chosen)
        write_json(folder/'trajectory.json', history)
    write_json(folder/'eval_recovery.json', {'source':source, 'reused_steps':len(chosen),
        'candidates':evidence, 'policy':'Earliest prefix, then matching extensions; fresh infrastructure time window; no score selection.'})
    return len(chosen)


def execute_windows(public, config, folder, memory, progress, execute, recoverable,
                    windows=3):
    for attempt in range(windows):
        path = folder/'predictions_local.json'
        actions = load(path) if path.exists() else []
        try:
            return execute(public, config, folder, memory, progress,
                           resume_steps=len(actions), previous_elapsed=0)
        except Exception as exc:
            if not recoverable(exc) or attempt + 1 == windows:
                raise
            event(folder, 'eval_window_recovery', window=attempt+1, error=type(exc).__name__)
            write_json(folder/'eval_windows'/f'{attempt+1:03d}.json', {
                'error':type(exc).__name__,
                'checkpoint':load(folder/'execution_checkpoint.json') if (folder/'execution_checkpoint.json').exists() else None,
                'predictions':load(path) if path.exists() else [],
                'trajectory':load(folder/'trajectory.json') if (folder/'trajectory.json').exists() else []})

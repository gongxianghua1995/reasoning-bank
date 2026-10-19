"""Shared scientific identity and train-only reference action boundary."""
from .common import digest
import json


def execution_identity(config):
    if not config.get('protocol_version'):
        return {}
    fields = ['protocol_version', 'model', 'input_mode', 'thinking', 'temperature',
              'max_tokens', 'learning_max_tokens', 'judge_temperature', 'extract_temperature',
              'reference_actions']
    return {'protocol_sha256': digest(json.dumps({k: config.get(k) for k in fields}, sort_keys=True)),
            'input_mode': config['input_mode']}


def reference_actions(case, task, metrics, train_ids):
    """Same fields/filter as Swarm 90b6cd5 Mind2WebBench.reference_actions."""
    if case.id not in train_ids:
        raise ValueError('Reference actions require a selected training case')
    if task['annotation_id'] != case.id:
        raise ValueError('Reference task identity mismatch')
    refs = []
    for index, (step, score) in enumerate(zip(task['steps'], metrics['steps'], strict=True)):
        if score['step_success_rate'] >= 1:
            continue
        mapping = step['id_mapping']['original_to_local']
        displayed = set(step['observation']['candidate_ids'])
        refs.append({'step_index': index,
                     'element_ids': [mapping[eid] for eid in step['gold']['element_ids']
                                     if eid in mapping and mapping[eid] in displayed],
                     'operation': step['gold']['operation'], 'value': step['gold']['value']})
    return refs

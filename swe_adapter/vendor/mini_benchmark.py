"""Public task inputs and image metadata for Verified and Pro."""
import json


def is_pro(record):
    return bool(record.get('dockerhub_tag'))


def image_name(record):
    if is_pro(record):
        return 'jefzda/sweap-images:' + record['dockerhub_tag'][:128]
    return 'swebench/sweb.eval.x86_64.' + record['instance_id'].replace('__', '_1776_') + ':latest'


def required_images(record):
    image = image_name(record)
    return [image] if is_pro(record) else [image, 'sweb.eval.x86_64.' + record['instance_id'] + ':latest']


def public_text(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return value.strip()
    return decoded.strip() if isinstance(decoded, str) else value.strip()


def task_text(record):
    text = public_text(record['problem_statement'])
    if is_pro(record):
        for key, heading in [('requirements', 'Public requirements'), ('interface', 'Public interface')]:
            value = public_text(record.get(key))
            if value:
                text += '\n\n## ' + heading + '\n' + value
    return text

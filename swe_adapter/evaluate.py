"""Independent offline Verified evaluation, run under swebench 2.1.6 Python.

Adapted from GPTSwarm experiments/swebench_mini_eval.py and its existing-image
compatibility hook. No agent framework or generation context is imported.
"""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch


def evaluate(data_path, out):
    import docker
    from docker.models.containers import ContainerCollection
    import swebench.harness.docker_build as build
    from swebench.harness.run_evaluation import (
        get_dataset_from_preds, load_swebench_dataset, run_instances, make_run_report)

    prediction = json.loads((out/'predictions.json').read_text())[0]
    original_create = ContainerCollection.create
    original_build = build.build_instance_image

    def existing_image(test_spec, client, logger, nocache):
        client.images.get(test_spec.instance_image_key)  # preflight; never download
        return None

    def offline_create(collection, *args, **kwargs):
        kwargs['network_mode'] = 'none'
        kwargs['labels'] = dict(kwargs.get('labels') or {}, **{'reasoningbank.swe.run':out.name})
        container = original_create(collection, *args, **kwargs)
        try:
            container.reload()
            network = container.attrs['HostConfig']['NetworkMode']
            if network != 'none':
                raise RuntimeError('Evaluation container is not offline')
            with (out/'eval_network.jsonl').open('a') as f:
                f.write(json.dumps(dict(container=container.name, network=network))+'\n')
            return container
        except Exception:
            container.remove(force=True)
            raise

    ids = [prediction['instance_id']]
    predictions = {ids[0]:prediction}
    run_id = out.name
    os.chdir(out)  # harness reports and detailed logs belong to this run
    dataset = get_dataset_from_preds(str(data_path), 'test', ids, predictions, run_id)
    try:
        build.build_instance_image = existing_image
        with patch.object(ContainerCollection, 'create', offline_create):
            if dataset:
                run_instances(predictions, dataset, cache_level='none', clean=False,
                              force_rebuild=False, max_workers=1, run_id=run_id, timeout=1800)
        report = make_run_report(predictions, load_swebench_dataset(str(data_path), 'test', ids),
                                 docker.from_env(), run_id)
        (out/'report.json').write_bytes(Path(report).read_bytes())
    finally:
        build.build_instance_image = original_build


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    evaluate(args.data_path.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()

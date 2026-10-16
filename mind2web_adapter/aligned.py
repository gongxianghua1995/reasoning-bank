"""ReasoningBank single-agent training with SwarmEvolution's feedback and batch schedule."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import sys

from .common import Progress, digest, event, file_hash, load, write_json
from .run import recoverable_case_error

ROOT = Path(__file__).resolve().parents[1]


def epoch_order(cases, seed, epoch):
    # SwarmEvolution load_epoch_cases: family order, then original per-family order.
    ordered = [c for site in sorted({c.family_id for c in cases}) for c in cases if c.family_id == site]
    random.Random(seed + epoch - 1).shuffle(ordered)
    return ordered


def identity(manifest, config):
    return {'split_sha256': manifest['split_sha256'],
            'data_identity': digest(json.dumps(manifest['cache_identity'], sort_keys=True)),
            'embedding_model': config['embedding_model'], 'model': config['model'],
            'retrieval': 'web-query-cosine-top1-plain-v2', 'feedback_source': 'official'}


def commit_batch(bank, path, entries, before):
    """Prepare an atomic replacement, then roll it forward after interrupted commits."""
    after = {**before, 'entries': before['entries'] + entries}
    if bank.state != before and bank.state != after:
        raise ValueError('Bank differs from both transaction states')
    write_json(path, after)
    bank.state = after


def parallel(fn, values, workers):
    # Preserve input order on disk, irrespective of completion order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, values))


def execute(out):
    from fastembed import TextEmbedding
    from swarmevo.benches.mind2web.dataset import load_inventory, read_case
    from swarmevo.benches.mind2web.metrics import score_case, aggregate
    from .agent import execute as act, public_case, original_predictions
    from .memory import Bank, learn

    options, config = load(out/'options.json'), load(out/'config.json')
    lock = (out/'.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    progress = Progress(out, mode='aligned_train', phase='preparing', completed=0, processed=0)
    try:
        if config.get('feedback_source') != 'official' or config.get('embedding_query_format') != 'plain':
            raise ValueError('Aligned training requires official feedback and plain query encoding')
        for name, sha in load(out/'source_hashes.json').items():
            if file_hash(out/'source_at_run'/name) != sha:
                raise ValueError('Source snapshot changed')
        for name, sha in load(out/'input_hashes.json').items():
            if file_hash(out/name) != sha:
                raise ValueError('Run configuration changed')
        manifest, parts = load_inventory(Path(options['inventory']), seed=options['seed'])
        if manifest['diagnostic']:
            raise ValueError('Full official inventory required')
        cases = parts['train']
        if options.get('max_cases'):
            cases = cases[:options['max_cases']]
        cache = Path(options['embedding_cache'])
        weights = {str(f.relative_to(cache)): file_hash(f) for f in cache.rglob('*') if f.is_file()}
        if not weights:
            raise ValueError('No local embedding weights')
        os.environ['HF_HUB_OFFLINE'] = '1'
        embedder = TextEmbedding(config['embedding_model'], cache_dir=str(cache),
                                  threads=config['embedding_threads'], local_files_only=True)
        bank = Bank(out/'memory.json', embedder, identity(manifest, config), 'global', query_format='plain')
        schedule = [{'epoch': e, 'batches': [[c.id for c in ordered[i:i+options['batch_size']]]
                     for i in range(0, len(ordered), options['batch_size'])]}
                    for e in range(1, options['epochs']+1)
                    for ordered in [epoch_order(cases, options['seed'], e)]]
        run_manifest = {'options': options, 'config': config, 'split_sha256': manifest['split_sha256'],
                        'inventory_sha256': file_hash(Path(options['inventory'])/'manifest.json'),
                        'embedding_files': weights, 'schedule': schedule, 'learning_protocol': 'swarm_aligned',
                        'feedback': 'official scores including per-step metrics; no reference actions',
                        'bank_identity': bank.state['identity']}
        if (out/'manifest.json').exists() and load(out/'manifest.json') != run_manifest:
            raise ValueError('Data, model weights, or training schedule changed')
        write_json(out/'manifest.json', run_manifest)
        write_json(out/'splits.json', {k: manifest[k] for k in ['parts','counts','split_sha256','source_files']})
        if not (out/'memory.json').exists():
            write_json(out/'memory.json', bank.state)
        rows, failures = [], []
        by_id = {c.id: c for c in cases}
        expected_entries = []
        for ep in schedule:
            for index, ids in enumerate(ep['batches'], 1):
                batch = out/'batches'/f"e{ep['epoch']:02d}_b{index:03d}"
                batch.mkdir(parents=True, exist_ok=True)
                before_file = batch/'memory_before.json'
                if before_file.exists():
                    before = load(before_file)
                else:
                    if bank.state['entries'] != expected_entries:
                        raise ValueError('Unexpected pre-batch memory')
                    before = json.loads(json.dumps(bank.state))
                    write_json(before_file, before)
                if before['entries'] != expected_entries:
                    raise ValueError('Batch snapshot contains current/future experience')
                if (batch/'committed.json').exists():
                    receipt = load(batch/'committed.json')
                    if file_hash(batch/'transaction.json') != receipt['transaction_sha256']:
                        raise ValueError('Committed transaction changed')
                    tx = load(batch/'transaction.json')
                else:
                    # Recovery always retrieves from the saved batch-start bank.
                    snapshot_bank = Bank(before_file, embedder, bank.state['identity'], 'global', query_format='plain',
                                         allow_same_case=True)
                    prepared = []
                    for cid in ids:
                        case = by_id[cid]
                        folder = out/'cases'/f"{ep['epoch']:02d}_{digest(cid)[:16]}"
                        folder.mkdir(parents=True, exist_ok=True)
                        public = public_case(read_case(case))
                        selected, text, vector, scores = snapshot_bank.select(cid, case.family_id, public['task'])
                        retrieval = {'episode_ids': selected, 'selected_memory': text, 'scores': scores,
                                     'memory_before_sha256': file_hash(before_file), 'scope': 'global'}
                        for name, value in [('public_case.json',public), ('retrieval.json',retrieval),
                            ('case_identity.json',{'case_id':cid,'episode_id':f"epoch-{ep['epoch']}:{cid}",
                             'website':case.family_id,'source_sha256':case.payload['sha256'],'split':'train'})]:
                            if (folder/name).exists() and load(folder/name) != value:
                                raise ValueError('Saved task input changed')
                            write_json(folder/name, value)
                        prepared.append((case, folder, public, text, vector, selected))
                    progress.update(phase='solve', epoch=ep['epoch'], batch=index, completed=len(rows), processed=len(rows)+len(failures))

                    def solve(item):
                        case, folder, public, text, vector, selected = item
                        if (folder/'solve_receipt.json').exists():
                            receipt = load(folder/'solve_receipt.json')
                            for name, sha in receipt.get('hashes',{}).items():
                                if file_hash(folder/name) != sha:
                                    raise ValueError('Completed action artifact changed')
                            return receipt
                        cp = Progress(folder, phase='solve', active=case.id)
                        try:
                            partial = load(folder/'predictions_local.json') if (folder/'predictions_local.json').exists() else []
                            previous = load(folder/'execution_checkpoint.json')['elapsed_seconds'] if (folder/'execution_checkpoint.json').exists() else 0
                            predictions = act(public,config,folder,text,cp,resume_steps=len(partial),previous_elapsed=previous)
                            original = original_predictions(predictions, read_case(case))
                            metrics = score_case(original,read_case(case))
                            write_json(folder/'predictions.json',original)
                            write_json(folder/'metrics.json',metrics)
                            receipt = {'status':'complete','hashes':{n:file_hash(folder/n) for n in
                                ['predictions_local.json','predictions.json','trajectory.json','metrics.json']}}
                        except Exception as exc:
                            if not options['continue_case_errors'] or not recoverable_case_error(exc):
                                raise
                            receipt = {'status':'action_failed','error':type(exc).__name__}
                            write_json(folder/'failure.json',receipt)
                        finally:
                            cp.close(status='stopped', phase=None)
                        write_json(folder/'solve_receipt.json',receipt)
                        return receipt
                    solved = parallel(solve, prepared, options['workers'])
                    progress.update(phase='learning')

                    def extract(pair):
                        item, receipt = pair
                        case, folder, public, text, vector, selected = item
                        episode = f"epoch-{ep['epoch']}:{case.id}"
                        if receipt['status'] != 'complete':
                            return {'case_id':case.id,'episode_id':episode,'status':'action_failed'}
                        if (folder/'learn_receipt.json').exists():
                            saved = load(folder/'learn_receipt.json')
                            if saved['status']=='complete' and file_hash(folder/'memory_entry.json') != saved['sha256']:
                                raise ValueError('Learned memory changed')
                            return saved
                        cp = Progress(folder,phase='learning',active=case.id)
                        try:
                            entry = learn(public,folder,config,vector,episode,metrics=load(folder/'metrics.json'))
                            saved = {'status':'complete','sha256':file_hash(folder/'memory_entry.json')}
                        except Exception as exc:
                            if not options['continue_case_errors'] or not recoverable_case_error(exc,learning=True):
                                raise
                            saved = {'status':'learning_failed','error':type(exc).__name__}
                            write_json(folder/'failure.json',saved)
                        finally:
                            cp.close(status='stopped', phase=None)
                        write_json(folder/'learn_receipt.json',saved)
                        return saved
                    learned = parallel(extract, list(zip(prepared,solved)),options['workers'])
                    batch_rows, batch_failures, entries = [], [], []
                    for item, solve_result, learn_result in zip(prepared,solved,learned,strict=True):
                        case, folder, public, text, vector, selected = item
                        row = {'case_id':case.id,'episode_id':f"epoch-{ep['epoch']}:{case.id}",
                               'website':case.family_id,'selected_memory_ids':selected}
                        if solve_result['status']=='complete':
                            row['metrics']=load(folder/'metrics.json')
                            row['learning_error']=learn_result.get('error')
                            batch_rows.append(row)
                            write_json(folder/'result.json',row)
                        if learn_result['status']=='complete':
                            entries.append(load(folder/'memory_entry.json'))
                        else:
                            batch_failures.append({**row,'status':learn_result['status']})
                    tx = {'rows':batch_rows,'failures':batch_failures,'entries':entries}
                    if (batch/'transaction.json').exists() and load(batch/'transaction.json') != tx:
                        raise ValueError('Pending transaction changed')
                    write_json(batch/'transaction.json',tx)
                    commit_batch(bank,out/'memory.json',tx['entries'],before)
                    write_json(batch/'committed.json',{'transaction_sha256':file_hash(batch/'transaction.json'),
                               'memory_after_sha256':file_hash(out/'memory.json')})
                expected_entries.extend(tx['entries'])
                rows.extend(tx['rows']); failures.extend(tx['failures'])
                write_json(out/'results.json',rows); write_json(out/'failures.json',failures)
                write_json(out/'case_outcomes.json',[{'episode_id':f"epoch-{e['epoch']}:{cid}"} for e in schedule
                    for j,b in enumerate(e['batches'],1) for cid in b
                    if e['epoch']<ep['epoch'] or (e['epoch']==ep['epoch'] and j<=index)])
                progress.update(completed=len(rows),processed=len(load(out/'case_outcomes.json')),failed=len(failures))
            checkpoint = out/'checkpoints'/f"epoch_{ep['epoch']}"
            state = {**bank.state,'entries':list(expected_entries)}
            if (checkpoint/'memory.json').exists() and load(checkpoint/'memory.json') != state:
                raise ValueError('Epoch checkpoint changed')
            write_json(checkpoint/'memory.json',state)
            write_json(checkpoint/'summary.json',{'epoch':ep['epoch'],'memory_entries':len(expected_entries),
                'memory_sha256':file_hash(checkpoint/'memory.json'),
                'metrics':aggregate([r['metrics'] for r in rows if r['episode_id'].startswith(f"epoch-{ep['epoch']}:")])})
        if bank.state['entries'] != expected_entries:
            raise ValueError('Final memory differs from committed transactions')
        summary={'status':'finished_with_failures' if failures else 'finished','learning_protocol':'swarm_aligned',
                 'mode':'aligned_train','completed':len(rows),'processed':len(cases)*options['epochs'],
                 'failed':len(failures),'memory_entries':len(expected_entries),
                 'metrics':aggregate([r['metrics'] for r in rows]),
                 'coverage':{'scheduled':len(cases)*options['epochs'],'scored':len(rows)},
                 'interpretation':'Training metrics aggregate epochs, not final generalization performance.'}
        write_json(out/'summary.json',summary)
        progress.close(status=summary['status'],phase=None)
        return 0
    except BaseException as exc:
        event(out,'run_error',error=type(exc).__name__)
        progress.close(status='error',error=type(exc).__name__)
        raise
    finally:
        lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path)
    p.add_argument('--config',type=Path,default=ROOT/'mind2web_adapter/config_aligned.json')
    p.add_argument('--embedding-cache',type=Path,default=ROOT/'.models')
    p.add_argument('--env-file',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-cases',type=int)
    p.add_argument('--continue-case-errors',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--frozen',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if a.frozen:
        raise SystemExit(execute(out))
    from dotenv import load_dotenv
    if a.env_file:
        load_dotenv(a.env_file,override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        p.error('OPENAI_API_KEY missing')
    if not a.resume:
        if not a.inventory or min(a.epochs,a.batch_size,a.workers)<1 or (a.max_cases is not None and a.max_cases<1):
            p.error('Inventory and positive dimensions required')
        out.mkdir(parents=True,exist_ok=False)
        frozen=out/'source_at_run'
        shutil.copytree(ROOT/'mind2web_adapter',frozen/'mind2web_adapter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write_json(out/'source_hashes.json',{str(f.relative_to(frozen)):file_hash(f) for f in frozen.rglob('*') if f.is_file()})
        write_json(out/'config.json',load(a.config))
        options={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()
                 if k not in {'config','env_file','resume','frozen'}}
        write_json(out/'options.json',options)
        write_json(out/'input_hashes.json',{n:file_hash(out/n) for n in ['config.json','options.json']})
    frozen=out/'source_at_run'
    os.environ['PYTHONPATH']=str(frozen);os.chdir(frozen)
    os.execv(sys.executable,[sys.executable,'-u','-m','mind2web_adapter.aligned','--frozen','--output',str(out)])


if __name__=='__main__':
    main()

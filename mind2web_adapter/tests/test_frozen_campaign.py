import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mind2web_adapter.common import write_json, file_hash, load
from mind2web_adapter.frozen_campaign import prepare


class FrozenCampaignTest(unittest.TestCase):
    def test_epoch_one_only_full_paired_coverage_and_no_training_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            counts = {'train':303, 'validation':101, 'test_task':177,
                      'test_website':142, 'test_domain':694}
            ids = {k:[f'{k}-{i}' for i in range(n)] for k,n in counts.items()}
            manifest = {'diagnostic':False, 'parts':ids, 'counts':counts,
                        'source_files':{}, 'split_sha256':'fixed'}
            write_json(root/'inventory/manifest.json', manifest)
            source = root/'checkpoint/memory.json'
            bank = {'identity':{'split_sha256':'fixed'}, 'entries':[
                {'case_id':c, 'episode_id':'epoch-1:'+c} for c in ids['train']]}
            write_json(source, bank)
            write_json(source.with_name('summary.json'), {'epoch':1, 'memory_sha256':file_hash(source)})
            args = argparse.Namespace(inventory=root/'inventory', memory_from=source,
                output=root/'out', embedding_cache=root/'models', workers=4)
            with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest,ids)):
                plan = prepare(args)
            self.assertEqual(plan['expected_episodes'],2228)
            self.assertEqual({j['group'] for j in plan['jobs']}, {'aligned','none'})
            for split in ('validation','test_task','test_website','test_domain'):
                for group in ('aligned','none'):
                    observed = [c for j in plan['jobs'] if j['group']==group and j['split']==split for c in j['case_ids']]
                    self.assertCountEqual(observed, ids[split])
                    self.assertEqual(len(set(observed)), len(observed))
            for job in plan['jobs']:
                self.assertNotIn('--epochs', job['command'])
                self.assertEqual('--memory-from' in job['command'], job['group']=='aligned')
                if job['group']=='aligned' and job['split']!='validation':
                    self.assertEqual(len(job['depends']),8)
            self.assertEqual(file_hash(args.output/'selected_memory.json'), file_hash(source))
            self.assertIn('selected_memory.json', load(args.output/'input_hashes.json'))
            bank['entries'][0]['episode_id']='epoch-2:'+ids['train'][0]
            write_json(source,bank)
            write_json(source.with_name('summary.json'), {'epoch':1, 'memory_sha256':file_hash(source)})
            args.output=root/'bad'
            with patch('swarmevo.benches.mind2web.dataset.load_inventory', return_value=(manifest,ids)):
                with self.assertRaisesRegex(ValueError,'first-epoch'):
                    prepare(args)
            self.assertFalse(args.output.exists())

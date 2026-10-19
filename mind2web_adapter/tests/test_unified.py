import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mind2web_adapter.agent import execute, public_case
from mind2web_adapter.memory import learn
from mind2web_adapter.model import Model, IncompleteResponse
from mind2web_adapter.protocol import execution_identity, reference_actions
from mind2web_adapter.tests.test_contract import CONFIG, ITEM, FakeModel, task_fixture
from swarmevo.benches.mind2web.metrics import score_case
from swarmevo.runtime.images import image_usage


def test_reference_boundary_and_unrecalled_targets(tmp_path):
    task = task_fixture()
    case = SimpleNamespace(id=task['annotation_id'])
    metrics = score_case([None, None], task)
    refs = reference_actions(case, task, metrics, {case.id})
    assert len(refs) == 2
    for ref, step in zip(refs, task['steps']):
        assert set(ref['element_ids']) <= set(step['observation']['candidate_ids'])
        assert ref['value'] == step['gold']['value']
    metrics['steps'][0]['step_success_rate'] = 1
    assert len(reference_actions(case, task, metrics, {case.id})) == 1
    task['steps'][1]['gold']['element_ids'] = ['not_displayed']
    assert reference_actions(case, task, metrics, {case.id})[0]['element_ids'] == []
    with pytest.raises(ValueError):
        reference_actions(case, task, metrics, set())
    public = public_case(task)
    (tmp_path/'trajectory.json').write_text('[]')
    FakeModel.responses = [ITEM]; FakeModel.requests = []
    learn(public,tmp_path,{**CONFIG,'feedback_source':'official','reference_actions':True},
          [1.,0.],'epoch-1:x',model_factory=FakeModel,metrics=metrics,reference_actions=refs)
    assert 'reference_actions' in json.dumps(FakeModel.requests)
    assert 'reference_actions' not in json.dumps(public)
    assert not (tmp_path/'self_judgment.json').exists()
    with pytest.raises(ValueError):
        learn(public,tmp_path,CONFIG,[1.,0.],'x',model_factory=FakeModel,reference_actions=refs)


def test_text_vision_share_observations_and_only_current_image(tmp_path):
    task = task_fixture()
    for i, step in enumerate(task['steps']):
        path = tmp_path/f'{i}.png'; Image.new('RGB',(32,24),(i*100,0,0)).save(path)
        step['action_uid'] = f'u{i}'; step['target_action_index'] = i
        step['image'] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size,'annotation_id':task['annotation_id'],
            'action_uid':f'u{i}','target_action_index':i}
    text = public_case(task); vision = public_case(task,'multimodal')
    assert text['observations'] == [{k:v for k,v in o.items() if k!='image'} for o in vision['observations']]
    requests = []
    class Actor:
        def __init__(self,*a,**kw):pass
        def complete(self,messages,temperature):
            i=len(requests);requests.append(copy.deepcopy(messages))
            return json.dumps({'element_id':vision['observations'][i]['candidate_ids'][0],
                              'operation':'CLICK','value':'','reason':'fixture'})
    execute(vision,CONFIG,tmp_path,model_factory=Actor)
    assert [image_usage(m)['image_count'] for m in requests] == [1,1]
    assert image_usage(requests[0])['images'][0]['sha256'] != image_usage(requests[1])['images'][0]['sha256']
    task['steps'][0]['image']['action_uid'] = 'future'
    with pytest.raises(ValueError):public_case(task,'multimodal')


def test_thinking_budget_and_truncation_are_not_silent(tmp_path):
    recorded=[]
    def create(**kwargs):
        recorded.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}',reasoning_content='reason'),finish_reason='length')],usage=None,id='test',model='fake')
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    import time
    c={**CONFIG,'model':'fake','max_tokens':64000,'learning_max_tokens':50000,
       'thinking':{'type':'enabled'},'request_attempts':1,'api_timeout':180}
    with pytest.raises(IncompleteResponse):
        Model(c,tmp_path,'solve',time.monotonic()+180,client=client).complete([{'role':'user','content':'test'}],0)
    assert recorded[0]['max_tokens']==64000
    assert recorded[0]['temperature']==0
    assert recorded[0]['extra_body']=={'thinking':{'type':'enabled'}}
    with pytest.raises(IncompleteResponse):
        Model(c,tmp_path,'extract',time.monotonic()+180,client=client).complete([{'role':'user','content':'test'}],1)
    assert recorded[1]['max_tokens']==50000


def test_memory_identity_changes_with_modality_and_feedback():
    c={'protocol_version':'new','input_mode':'text','reference_actions':True}
    assert execution_identity(c)!=execution_identity({**c,'input_mode':'multimodal'})
    assert execution_identity(c)!=execution_identity({**c,'reference_actions':False})

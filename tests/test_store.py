import pytest
from vlogflow.store import Store, Conflict

@pytest.fixture
def store(tmp_path): return Store(tmp_path)

def approved(store):
    p=store.create('A night walk','en')
    p=store.edit(p['id'], {'media':[{'id':'m','duration':5}], 'scenes':[{'id':'s','media_id':'m','start':0,'end':5,'narration':'Hello','description':'street','ambient':0}]})
    return store.approve(p['id'],p['epoch'])

def test_approved_snapshot_is_immutable_and_edits_invalidate_outputs(store):
    p=approved(store); rev=p['revision']
    job=store.create_job(p['id'],'render')
    store.publish(job['id'], {'artifacts':{'video':'out.mp4'}, 'voice_result':{'key':'abc'}},'complete')
    q=store.get(p['id']); scenes=q['scenes']; scenes[0]['narration']='Bye'
    q=store.edit(p['id'],{'scenes':scenes},q['epoch'])
    assert q['revision'] is None and q['artifacts']=={} and q['voice_result'] is None
    assert store.revision(rev)['scenes'][0]['narration']=='Hello'

def test_stale_worker_cannot_publish_over_newer_edit(store):
    p=approved(store); job=store.create_job(p['id'],'render')
    store.edit(p['id'],{'brief':'new brief'},p['epoch'])
    assert not store.publish(job['id'],{'artifacts':{'video':'old.mp4'}},'complete')
    assert store.get(p['id'])['artifacts']=={}
    assert store.job(job['id'])['status']=='superseded'

def test_lost_update_and_subtitle_change_keep_cached_voice(store):
    p=approved(store); job=store.create_job(p['id'],'render')
    store.publish(job['id'],{'voice_result':{'key':'abc'},'artifacts':{'video':'out'}},'complete')
    q=store.edit(p['id'],{'subtitle_mode':'zh'},p['epoch'])
    assert q['revision']==p['revision'] and q['voice_result']=={'key':'abc'} and q['artifacts']=={}
    with pytest.raises(Conflict): store.edit(p['id'],{'brief':'lost'},p['epoch'])

def test_restart_marks_running_jobs_interrupted_and_preserves_project(store):
    p=approved(store); j=store.create_job(p['id'],'render'); store.update_job(j['id'],status='running')
    store.recover()
    assert store.job(j['id'])['status']=='interrupted'
    assert store.get(p['id'])['revision']==p['revision']

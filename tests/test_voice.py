import json
import struct
from pathlib import Path
import pytest
from vlogflow.models import subset_weights
from vlogflow.voice import cache_key, group_scenes
from vlogflow.alignment import align_scenes


def test_cache_key_changes_with_reference_text_voice_language_and_context():
    args=dict(text='Hello',reference_hash='abc',reference_text='My voice',model='pinned',language='en',context=['Hello'])
    first=cache_key(**args)
    for key,value in [('reference_hash','xyz'),('reference_text','Other'),('language','zh'),('context',['Before','Hello'])]:
        assert cache_key(**(args|{key:value}))!=first


def test_subset_preserves_exact_tensor_bytes_and_excludes_text_table(tmp_path):
    source=tmp_path/'model.safetensors'; out=tmp_path/'subset.safetensors'
    header={'a':{'dtype':'F32','shape':[2],'data_offsets':[0,8]},'text':{'dtype':'F32','shape':[1],'data_offsets':[8,12]}}
    encoded=json.dumps(header).encode(); source.write_bytes(struct.pack('<Q',len(encoded))+encoded+struct.pack('<fff',1,2,3))
    record=subset_weights(source,'',out,excluded=('text',))
    with out.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; result=json.loads(f.read(n)); raw=f.read()
    assert list(result)==['a'] and raw==struct.pack('<ff',1,2) and record['byte_identical']


def test_alignment_uses_pause_between_scenes_and_rejects_unrelated_speech():
    scenes=[{'id':'a','narration':'Hello there.'},{'id':'b','narration':'Good evening.'}]
    words=[{'word':'Hello','start':.1,'end':.4},{'word':'there','start':.4,'end':.7},{'word':'Good','start':1.1,'end':1.4},{'word':'evening','start':1.4,'end':1.8}]
    result=align_scenes(scenes,words,2,'en')
    assert result[0]['end']==pytest.approx(.9) and result[1]['start']==pytest.approx(.9)
    with pytest.raises(ValueError,match='transcript|recognition'): align_scenes(scenes,[{'word':'unrelated','start':0,'end':1}],2,'en')


def test_chinese_alignment_and_short_sentence_grouping():
    scenes=[{'id':'a','narration':'你好。'},{'id':'b','narration':'晚上好。'}]
    words=[{'word':'你好','start':.1,'end':.7},{'word':'晚上好','start':1,'end':1.9}]
    result=align_scenes(scenes,words,2,'zh')
    assert result[0]['end']==pytest.approx(.85) and result[-1]['end']==2
    assert len(group_scenes(scenes,'zh'))==1

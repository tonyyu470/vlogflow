from pathlib import Path
import subprocess
import numpy as np
import pytest
import soundfile as sf
from vlogflow.media import probe, import_video, prepare_reference, render
from vlogflow.captions import layout_caption, write_srt
from vlogflow.processes import Runner

@pytest.fixture
def clip(tmp_path):
    p=tmp_path/"视频 ' sample.mp4"
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=30','-t','2','-c:v','libx264','-threads','2','-pix_fmt','yuv420p',str(p)],check=True)
    return p

def test_import_has_stable_id_and_timestamped_samples_for_unicode_filename(tmp_path,clip):
    a=import_video(clip,tmp_path/'media',Runner(tmp_path/'logs'))
    b=import_video(clip,tmp_path/'media',Runner(tmp_path/'logs'))
    assert a['id']==b['id'] and a['duration']==2 and not a['has_audio']
    assert len(a['frames'])==3 and all(0<=f['time']<2 for f in a['frames'])
    assert Path(a['path']).exists()

def test_reference_rejects_silence_and_records_clipping(tmp_path):
    p=tmp_path/'silent.wav'; sf.write(p,np.zeros(24000*4),24000)
    with pytest.raises(ValueError,match='speech|quiet|silence'): prepare_reference(p,tmp_path/'ref',Runner(tmp_path/'logs'))
    t=np.arange(24000*4)/24000; sf.write(p,np.clip(2*np.sin(2*np.pi*220*t),-1,1),24000)
    with pytest.raises(ValueError,match='clip'): prepare_reference(p,tmp_path/'ref2',Runner(tmp_path/'logs'))

def test_render_no_source_audio_and_rejects_voice_overflow(tmp_path,clip):
    runner=Runner(tmp_path/'logs'); m=import_video(clip,tmp_path/'media',runner)
    voice=tmp_path/'voice.wav'; sf.write(voice,np.sin(2*np.pi*220*np.arange(24000)/24000)*.1,24000)
    scene=dict(id='s',media_id=m['id'],start=0,end=2,narration='Hello',ambient=.1)
    cues=[dict(start=0,end=1,zh='广州夜宵',en='A late night snack')]
    out=render([m],[scene],[str(voice)],cues,'bilingual',tmp_path/'render',runner,'portrait')
    info=probe(Path(out['video']),runner)
    assert info['has_audio'] and abs(info['duration']-1)<.1 and info['width']==720
    scene['end']=.5
    with pytest.raises(ValueError,match='s.*range|s.*配音'): render([m],[scene],[str(voice)],cues,'en',tmp_path/'bad',runner)

def test_caption_layout_stays_inside_safe_frame_and_srt_bounds(tmp_path):
    img,box=layout_caption('很长的一句中文说明，需要正确自动换行。'*3,'A very long English sentence with spaces. '*4,720,1280)
    assert box[0]>=20 and box[2]<=700 and box[3]<=1240
    assert img.size==(720,1280)
    with pytest.raises(ValueError): write_srt([dict(start=0,end=4,zh='你好',en='Hi')],'en',tmp_path/'bad.srt',2)

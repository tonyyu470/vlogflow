"""Local CPU-only speech recognition, unloaded when this process exits."""
import argparse
import json
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--audio',required=True)
p.add_argument('--output',required=True); p.add_argument('--language',default=''); p.add_argument('--prompt',default='')
a=p.parse_args()
from faster_whisper import WhisperModel
model=WhisperModel(a.model,device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,local_files_only=True)
segments,info=model.transcribe(a.audio,language=a.language or None,beam_size=5,word_timestamps=True,vad_filter=True,
                               condition_on_previous_text=False,initial_prompt=a.prompt or None)
items=list(segments)
result=dict(text=' '.join(s.text.strip() for s in items),language=info.language,words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability) for s in items for w in s.words or []])
Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2))

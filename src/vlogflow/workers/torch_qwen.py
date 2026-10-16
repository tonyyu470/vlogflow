"""Experimental official Qwen backend; not hardware-verified on Windows/Linux."""
import argparse
import json
import os
from pathlib import Path

os.environ.update(OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); args=p.parse_args()
m=json.loads(Path(args.manifest).read_text())
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

torch.set_num_threads(2)
device='cuda:0' if torch.cuda.is_available() else 'cpu'
model=Qwen3TTSModel.from_pretrained(m['model_path'],device_map=device,dtype=torch.bfloat16 if device!='cpu' else torch.float32,attn_implementation='sdpa')
torch.manual_seed(42)
wavs,sr=model.generate_voice_clone(text=m['group']['text'],language=m['language'],ref_audio=m['reference_audio'],ref_text=m['reference_text'],max_new_tokens=m['max_tokens'])
sf.write(Path(args.manifest).parent/'speech.wav',wavs[0],sr,subtype='PCM_24')

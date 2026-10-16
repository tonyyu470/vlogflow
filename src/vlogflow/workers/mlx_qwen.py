"""Run unchanged Qwen operations in separate processes to bound peak memory."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import struct
import time
from types import SimpleNamespace

for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', USE_TORCH='0', USE_TF='0')
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil
import soundfile as sf
from mlx_audio.tts.models.qwen3_tts.config import ModelConfig, Qwen3TTSTokenizerConfig, filter_dict_for_dataclass
from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model
from mlx_audio.tts.models.qwen3_tts.speech_tokenizer import Qwen3TTSSpeechTokenizer, Qwen3TTSSpeechTokenizerEncoder, Qwen3TTSSpeechTokenizerDecoder
from mlx_audio.tts.models.qwen3_tts.speaker_encoder import Qwen3TTSSpeakerEncoder

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['reference', 'codes', 'decode'])
parser.add_argument('--manifest', required=True)
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text())
OUT = Path(args.manifest).parent
MODEL = Path(manifest['model_path'])
group = manifest['group']
work = OUT
WEIGHTS = Path(manifest['weights'])
FEATURES = Path(manifest['features'])
work.mkdir(parents=True, exist_ok=True)
mx.set_memory_limit(3*1024**3)
mx.set_cache_limit(32*1024**2)
mx.set_wired_limit(int(2.5*1024**3))
mx.random.seed(42)
started = time.monotonic()
state = {'stage': args.stage, 'group': group['id'], 'pid': os.getpid()}
def record(phase, **extra):
    state.update(phase=phase, elapsed_seconds=round(time.monotonic()-started, 3), mlx_peak_bytes=mx.get_peak_memory(), mlx_active_bytes=mx.get_active_memory(), rss_bytes=psutil.Process().memory_info().rss, **extra)
    (work/f'{args.stage}-status.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)

verification = manifest['model_record']
config = ModelConfig.from_dict(json.loads((MODEL/'config.json').read_text()))
token_config = Qwen3TTSTokenizerConfig(**filter_dict_for_dataclass(Qwen3TTSTokenizerConfig, json.loads((MODEL/'speech_tokenizer/config.json').read_text())))
reference, sr = sf.read(manifest['reference_audio'], dtype='float32')
assert sr == 24000
ref_hash = hashlib.sha256(Path(manifest['reference_audio']).read_bytes()).hexdigest()
assert ref_hash == manifest['reference_sha256']
ref = mx.array(reference)
record('starting')

if args.stage == 'reference':
    encoder = Qwen3TTSSpeechTokenizerEncoder(token_config.encoder_config)
    weights = Qwen3TTSSpeechTokenizer.sanitize(mx.load(str(WEIGHTS/'encoder.safetensors')))
    weights = {k.removeprefix('encoder_model.'): v for k, v in weights.items()}
    from mlx.utils import tree_flatten
    defaults = dict(tree_flatten(encoder.parameters()))
    missing = defaults.keys()-weights.keys()
    assert all(k.endswith('.codebook.initialized') for k in missing), missing
    weights.update({k: defaults[k] for k in missing})
    encoder.load_weights(list(weights.items()), strict=True)
    mx.eval(encoder.parameters()); encoder.eval()
    del weights, defaults; gc.collect(); mx.clear_cache()
    for stack in (encoder.quantizer.rvq_first, encoder.quantizer.rvq_rest):
        for layer in stack.vq.layers:
            layer.codebook.update_in_place()
    codes = encoder.encode(ref[None, None, :])
    mx.eval(codes)
    del encoder; gc.collect(); mx.clear_cache()
    record('reference_codes_ready')
    speaker = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    weights = Model.sanitize(mx.load(str(WEIGHTS/'speaker.safetensors')))
    speaker.load_weights([(k.removeprefix('speaker_encoder.'), v) for k, v in weights.items()], strict=True)
    mx.eval(speaker.parameters()); speaker.eval()
    del weights; gc.collect(); mx.clear_cache()
    embedding = Model.extract_speaker_embedding(SimpleNamespace(speaker_encoder=speaker), ref)
    mx.eval(embedding)
    temporary_features = FEATURES.with_suffix('.partial.safetensors')
    mx.save_safetensors(str(temporary_features), {'codes': codes, 'speaker': embedding}, metadata={'reference_sha256': ref_hash})
    temporary_features.replace(FEATURES)
    record('complete', reference_codes_shape=list(codes.shape), speaker_shape=list(embedding.shape))

elif args.stage == 'codes':
    assert group is not None
    features, metadata = mx.load(str(FEATURES), return_metadata=True)
    assert metadata['reference_sha256'] == ref_hash
    class CodesCaptured(Exception):
        pass
    class CachedTokenizer:
        has_encoder = True
        def encode(self, audio):
            return features['codes']
        def decode(self, codes):
            mx.eval(codes)
            mx.save_safetensors(str(work/'generated-codes.safetensors'), {'full_codes': codes}, metadata={'reference_sha256': ref_hash, 'text': group['text']})
            state['full_code_frames'] = int(codes.shape[1])
            state['generated_code_frames'] = int(codes.shape[1]-features['codes'].shape[2])
            raise CodesCaptured()
    class DiskTextEmbedding(nn.Module):
        """Exact BF16 row lookup without materializing the unused vocabulary."""
        def __init__(self):
            super().__init__()
            path=MODEL/'model.safetensors'
            with path.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(n))
            entry=header['talker.model.text_embedding.weight']
            assert entry['dtype']=='BF16'
            self._rows=np.memmap(path,mode='r',dtype='<u2',offset=8+n+entry['data_offsets'][0],shape=tuple(entry['shape']))
        def __call__(self, ids):
            indexes=np.array(ids)
            assert indexes.min()>=0 and indexes.max()<self._rows.shape[0]
            bits=np.array(self._rows[indexes],copy=True)
            values=mx.array(bits).view(mx.bfloat16)
            mx.eval(values)
            assert np.array_equal(np.array(values.view(mx.uint16)),bits)
            return values
    from mlx_audio.utils import apply_quantization
    from transformers import AutoTokenizer
    model=Model(config)
    model.talker.model.text_embedding=DiskTextEmbedding()
    weights=Model.sanitize(mx.load(str(WEIGHTS/'talker-without-text-table.safetensors')))
    apply_quantization(model,json.loads((MODEL/'config.json').read_text()),weights,model.model_quant_predicate)
    model.load_weights(list(weights.items()),strict=True)
    mx.eval(model.parameters()); model.eval()
    del weights
    model.tokenizer=AutoTokenizer.from_pretrained(str(MODEL))
    model.speech_tokenizer=CachedTokenizer()
    model.load_generate_config(json.loads((MODEL/'generation_config.json').read_text()))
    model.extract_speaker_embedding = lambda *a, **kw: features['speaker']
    gc.collect(); mx.clear_cache()
    record('loaded_talker')
    # Reset just before sampling, independently of initialization in other processes.
    mx.random.seed(42)
    captured = False
    try:
        for _ in model.generate(text=group['text'], ref_audio=ref, ref_text=manifest['reference_text'], lang_code=manifest['language'], temperature=.8, top_k=50, top_p=1., repetition_penalty=1.5, max_tokens=manifest['max_tokens'], stream=False, verbose=False):
            raise AssertionError('Expected code capture before waveform decoding')
    except CodesCaptured:
        captured = True
    assert captured and 1 < state['generated_code_frames'] < manifest['max_tokens'], 'No output or generation token limit reached'
    record('complete')

else:
    assert group is not None
    stored, metadata = mx.load(str(work/'generated-codes.safetensors'), return_metadata=True)
    assert metadata['reference_sha256'] == ref_hash and metadata['text'] == group['text']
    features = mx.load(str(FEATURES))
    decoder = Qwen3TTSSpeechTokenizerDecoder(token_config.decoder_config)
    weights = Qwen3TTSSpeechTokenizer.sanitize(mx.load(str(WEIGHTS/'decoder.safetensors')))
    decoder.load_weights([(k.removeprefix('decoder.'), v) for k, v in weights.items()], strict=True)
    mx.eval(decoder.parameters()); decoder.eval()
    del weights; gc.collect(); mx.clear_cache()
    codes = stored['full_codes']
    transposed = mx.transpose(codes, (0, 2, 1))
    wavs = []
    for start in range(0, transposed.shape[-1], 20):
        end = min(start+20, transposed.shape[-1]); context = min(start, 25)
        wave = decoder(transposed[..., start-context:end])
        wave = wave[..., context*decoder.total_upsample:]
        mx.eval(wave); wavs.append(np.array(wave, dtype=np.float32)); mx.clear_cache()
    audio = np.concatenate(wavs, axis=-1).reshape(-1)
    valid_len = int((codes[..., 0] > 0).sum())*token_config.decode_upsample_rate
    if 0 < valid_len < len(audio):
        audio = audio[:valid_len]
    cut = int(features['codes'].shape[2]/max(codes.shape[1], 1)*len(audio))
    assert 0 < cut < len(audio)
    audio = audio[cut:]
    assert np.isfinite(audio).all() and .2 < len(audio)/sr < 30
    assert np.sqrt(np.mean(audio**2)) > .0001
    output = work/'speech.wav'
    sf.write(output, audio, sr, subtype='PCM_24')
    code_state = json.loads((work/'codes-status.json').read_text())
    record('complete', output=str(output), text=group['text'], audio_seconds=len(audio)/sr, sample_rate=sr, model=verification['repo'], revision=verification['revision'], reference=manifest['reference_audio'], generation_seconds=code_state['elapsed_seconds']+time.monotonic()-started, peak_amplitude=float(np.max(np.abs(audio))), rms_dbfs=float(20*np.log10(np.sqrt(np.mean(audio**2)))), mode='Reference audio plus transcript; staged native ICL, exact on-demand text embedding rows, reference-conditioned 20/25-frame decode; no pitch/speed change')
    (work/'worker-result.json').write_text(json.dumps(state, indent=2))

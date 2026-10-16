# VlogFlow

**Turn everyday footage into a story, narrated in your own voice.**

[简体中文](README.md) · [Usage](docs/usage.md) · [Verification](docs/verification.md)

VlogFlow is a local browser application with a review-first workflow:

1. Import multiple clips and describe your idea. A vision model examines timestamped sample frames and drafts an editable shot-by-shot script. Review narration, source ranges, ordering and ambient audio.
2. Select Chinese, English or bilingual captions. Narration language follows the approved script; caption language is independent. Review and edit translations.
3. Upload a reference recording or enter its local path, correct its transcript, audition the voice and render an MP4. Download the MP4, SRT and final script JSON.

Approved narration is never silently rewritten, truncated or sped up. If narration exceeds its approved source range, the app asks you to shorten the text or extend the range. No lip-sync generation is included.

## Apple Silicon quick start

Use Python 3.11–3.12, FFmpeg and approximately 12 GB of disk space for models and caches. 16 GB RAM or more is recommended; staged MLX inference requires at least 4.1 GiB of available memory. The application stops a task when memory/swap limits are exceeded and keeps completed intermediates for retry.

```bash
brew install ffmpeg ollama uv
git clone https://github.com/tonyyu470/vlogflow.git
cd vlogflow
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[mlx,asr]'
```

Start Ollama in a second terminal:

```bash
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve
```

Download the models once, then start the app:

```bash
ollama pull qwen3-vl:2b
vlogflow models download qwen-mlx
vlogflow models download asr
vlogflow doctor
vlogflow serve
```

Open **http://127.0.0.1:7860**. No Node.js or frontend build is required. `--port` and `--no-browser` are supported. Existing model directories can be configured in Settings; they must match the pinned model manifests.

## Models and cost

Local Ollama vision, Qwen3-TTS voice cloning, faster-whisper recognition and FFmpeg rendering have no API invocation fee. Local hardware, storage and electricity are still required. Optional OpenAI-compatible vision endpoints may charge according to their provider.

For an OpenAI-compatible endpoint, configure a base URL (including `/v1` when required), a multimodal model and an API key. The provider must support image input and JSON output. Sample frames, the brief and translation text are sent to the selected provider. Reference recordings and voice generation remain local.

## Data and security

The server binds only to loopback and is intended for one local user. Do not expose it through a public tunnel or port forwarding. macOS data lives under `~/Library/Application Support/VlogFlow`; other platforms use their standard application data directory. Override with `VLOGFLOW_DATA_DIR`.

SQLite retains revisions and job state. Uploaded files, imported media copies, reference recordings, model files and outputs stay outside the source repository. Credentials are kept in local settings and hidden in API reads. Changes invalidate dependent outputs; stale jobs cannot overwrite newer work. Heavy stages are serialized. Cancellation targets only owned worker processes.

## Current limitations

This is the initial **v0.1.0** release. See the [verification record](docs/verification.md) for exact tested behavior.

- Apple Silicon is the primary target. The official Qwen PyTorch adapter is experimental and has not been hardware-tested on Windows, Linux or CUDA. See the Chinese usage guide for setup.
- Vision uses three representative frames per source, not exhaustive video understanding. Review all facts and narration.
- When ASR is configured, the first 30 seconds of each source are checked for existing speech. Preview the clips and mute original audio where appropriate; automatic speech separation is not included.
- ASR checks transcription/alignment, not subjective speaker similarity. Clean reference speech matters. Cross-language cloning and very short utterances may need iteration.
- Chinese and English are supported. Prefer one or two short sentences per scene.
- Up to 50 source clips per project, 10 minutes per clip, 4 GiB per file. Reference recordings: 3–30 seconds, up to 64 MiB uploaded.
- H.264/AAC output at 30 fps, 720-pixel short edge; aspect mismatches are letterboxed.

## Development

```bash
uv pip install -e '.[dev]'
pytest
python -m build
```

CI uses synthetic video and explicitly substituted model outputs. Real-model verification is reported separately. The tests include revision isolation, stale jobs, invalid ranges, caption modes, cancellation, memory protection, HTTP integration and real FFmpeg encode/decode.

The application code is MIT licensed; model and dependency licenses are separate. Use your own voice or an authorized reference. See [third-party notices](THIRD_PARTY_NOTICES.md).

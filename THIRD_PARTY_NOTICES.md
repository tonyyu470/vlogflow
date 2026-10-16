# Third-party notices

VlogFlow application code is MIT licensed. It is an independent implementation; it does not bundle the NarratoAI application or FFmpeg binaries. Earlier workflow experiments were informed by [NarratoAI](https://github.com/linyqh/NarratoAI) (MIT), but production modules here do not vendor its code.

| Component | License / source | Distribution |
|---|---|---|
| Qwen3-TTS models and official library | Apache-2.0, [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Separately installed/downloaded |
| MLX-Audio | MIT, [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) | Optional dependency; native Qwen implementation is imported |
| MLX | MIT, [ml-explore/mlx](https://github.com/ml-explore/mlx) | Optional dependency |
| faster-whisper | MIT, [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Optional dependency |
| Whisper models | MIT, [openai/whisper](https://github.com/openai/whisper) | Separately downloaded |
| Qwen3-VL | Apache-2.0, [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | Suggested Ollama model, separately downloaded |
| Ollama | MIT, [ollama/ollama](https://github.com/ollama/ollama) | Separately installed service |
| FFmpeg | LGPL/GPL depending on build, [FFmpeg legal](https://ffmpeg.org/legal.html) | User-supplied executable; not bundled |
| Noto Sans SC | SIL Open Font License 1.1, [Google Fonts](https://github.com/google/fonts/tree/main/ofl/notosanssc) | Bundled unmodified font; complete license in `src/vlogflow/assets/OFL.txt` |

The bundled font retains its upstream copyright: Copyright 2014–2021 Adobe, with Reserved Font Name “Source”. It is distributed under the OFL, separately from the application MIT license.

Other runtime dependencies (FastAPI, Starlette, Uvicorn, HTTPX, Pydantic, Pillow, NumPy, soundfile, psutil, platformdirs, filelock and huggingface-hub) retain their own licenses. Installation tools fetch their distributions and license metadata. No claim is made that all dependencies or model weights are covered by VlogFlow's MIT license.

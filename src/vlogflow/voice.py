from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .alignment import align_scenes, tokens
from .config import Settings
from .domain import digest
from .media import file_hash
from .models import manifest, prepare_components, verify_model
from .processes import ASR_BUDGET, Budget, Runner

WORKERS = Path(__file__).parent / "workers"
PARAMS = dict(
    seed=42,
    temperature=0.8,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.5,
    max_tokens=300,
    decoder_chunk=20,
    decoder_context=25,
    implementation=1,
)


def cache_key(*, text, reference_hash, reference_text, model, language, context):
    return digest(
        dict(
            text=text,
            reference_hash=reference_hash,
            reference_text=reference_text,
            model=model,
            language=language,
            context=context,
            params=PARAMS,
        )
    )


def group_scenes(scenes, language):
    groups = []
    current = []
    units = 0
    for s in scenes:
        count = len(tokens(s["narration"]))
        limit = 42 if language == "zh" else 29
        if current and (units + count > limit or len(current) >= 3):
            groups.append(current)
            current = []
            units = 0
        current.append(s)
        units += count
    if current:
        groups.append(current)
    return groups


def transcribe(audio: Path, settings: Settings, root: Path, runner: Runner, language="", prompt=""):
    model = Path(settings.asr_model).expanduser() if settings.asr_model else root / "models/asr"
    verify_model(model, "asr")
    key = digest(
        dict(
            audio=file_hash(audio),
            model=manifest("asr")["revision"],
            language=language,
            prompt=prompt,
            version=1,
        )
    )
    folder = root / "cache/asr"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"{key}.json"
    if not output.exists():
        temp = output.with_suffix(".partial")
        runner.run(
            [
                sys.executable,
                str(WORKERS / "asr.py"),
                "--model",
                str(model),
                "--audio",
                str(audio),
                "--output",
                str(temp),
                "--language",
                language,
                "--prompt",
                prompt,
            ],
            budget=ASR_BUDGET,
        )
        data = json.loads(temp.read_text())
        if not isinstance(data.get("words"), list):
            raise ValueError("Speech recognition did not return timestamps")
        temp.replace(output)
    return json.loads(output.read_text())


def _cached(folder):
    result = folder / "result.json"
    if result.exists():
        data = json.loads(result.read_text())
        if all(
            Path(s["path"]).is_file() and file_hash(Path(s["path"])) == s["sha256"] for s in data["segments"]
        ):
            return data
    return None


def generate_voice(scenes, language, reference, settings: Settings, root: Path, runner: Runner):
    if not reference or not reference.get("transcript", "").strip():
        raise ValueError("Review and save the reference recording transcript before generating voice")
    ref = Path(reference["path"])
    if file_hash(ref) != reference["sha256"]:
        raise ValueError("Reference recording changed; import it again")
    kind = "qwen-mlx" if settings.tts_backend == "mlx" else "qwen-torch"
    model = Path(settings.tts_model).expanduser() if settings.tts_model else root / "models" / kind
    runner.phase("verifying model files")
    record = verify_model(model, kind)
    if kind == "qwen-mlx":
        runner.phase("preparing model components")
        weights = prepare_components(model, root / "cache/components" / record["revision"], runner)
    else:
        weights = None
    features_dir = (
        root / "cache/references" / digest(dict(reference=reference["sha256"], model=record["revision"]))
    )
    features_dir.mkdir(parents=True, exist_ok=True)
    features = features_dir / "features.safetensors"
    all_segments = []
    keys = []
    grouped = group_scenes(scenes, language)
    for index, group in enumerate(grouped):
        text = " ".join(s["narration"] for s in group)
        context = [s["narration"] for s in scenes]
        key = cache_key(
            text=text,
            reference_hash=reference["sha256"],
            reference_text=reference["transcript"],
            model=kind + record["revision"],
            language=language,
            context=context,
        )
        keys.append(key)
        folder = root / "cache/voice" / key
        folder.mkdir(parents=True, exist_ok=True)
        cached = _cached(folder)
        if cached:
            # Cache IDs belong to the current script, even across projects with identical narration.
            for s, segment in zip(group, cached["segments"]):
                all_segments.append(dict(segment, scene_id=s["id"]))
            continue
        runner.phase(f"voice {index + 1}/{len(grouped)}")
        inputs = dict(
            model_path=str(model),
            model_record=record,
            weights=str(weights),
            features=str(features),
            reference_audio=str(ref),
            reference_sha256=reference["sha256"],
            reference_text=reference["transcript"],
            language="Chinese" if language == "zh" else "English",
            group=dict(id=f"g{index}", text=text),
            max_tokens=PARAMS["max_tokens"],
        )
        inputs_path = folder / "inputs.json"
        inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=2))
        raw = folder / "speech.wav"
        # Only a completed worker and later validated waveform may be reused.
        if kind == "qwen-mlx":
            budget = Budget(max(4.1, settings.min_available_gib), 1.6, 3.5, 128, 720)
            if not features.exists():
                runner.run(
                    [
                        settings.worker_python,
                        str(WORKERS / "mlx_qwen.py"),
                        "reference",
                        "--manifest",
                        str(inputs_path),
                    ],
                    budget=budget,
                )
            runner.run(
                [
                    settings.worker_python,
                    str(WORKERS / "mlx_qwen.py"),
                    "codes",
                    "--manifest",
                    str(inputs_path),
                ],
                budget=budget,
            )
            runner.run(
                [
                    settings.worker_python,
                    str(WORKERS / "mlx_qwen.py"),
                    "decode",
                    "--manifest",
                    str(inputs_path),
                ],
                budget=budget,
            )
        else:
            runner.run(
                [settings.worker_python, str(WORKERS / "torch_qwen.py"), "--manifest", str(inputs_path)],
                budget=Budget(10, 2, 12, 128, 1200),
            )
        runner.phase("aligning narration")
        audio, sr = sf.read(raw, dtype="float32")
        if sr != 24000 or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("TTS returned an invalid 24 kHz waveform")
        asr = transcribe(raw, settings, root, runner, language, prompt=text)
        aligned = align_scenes(group, asr["words"], len(audio) / sr, language)
        segments = []
        for n, (scene, span) in enumerate(zip(group, aligned)):
            data = audio[round(span["start"] * sr) : round(span["end"] * sr)].copy()
            fade = min(96, len(data) // 4)
            data[:fade] *= np.linspace(0, 1, fade)
            data[-fade:] *= np.linspace(1, 0, fade)
            output = folder / f"scene-{n}.wav"
            sf.write(output, data, sr, subtype="PCM_24")
            segments.append(
                dict(
                    scene_id=scene["id"],
                    path=str(output),
                    sha256=file_hash(output),
                    duration=len(data) / sr,
                    coverage=span["coverage"],
                    words=[[max(0, a - span["start"]), max(0, b - span["start"])] for a, b in span["words"]],
                )
            )
        result = dict(segments=segments, key=key, recognized_text=asr["text"])
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        all_segments += segments
    return dict(segments=all_segments, key=digest(keys), model=record["repo"], revision=record["revision"])

from __future__ import annotations

import json
from pathlib import Path

from .alignment import caption_cues
from .domain import digest
from .media import VIDEO_EXTENSIONS, file_hash, import_video, prepare_reference, render
from .providers import Provider, analyze_clip, compose_script, translate_script
from .voice import generate_voice, transcribe


def analyze_source_speech(media, settings, root, runner):
    """Inspect up to 30 seconds of original audio; keep the scope explicit."""
    if not media["has_audio"]:
        return dict(speech_detected=False, source_transcript="", speech_checked_seconds=0)
    folder = root / "cache/source-audio"
    folder.mkdir(parents=True, exist_ok=True)
    audio = folder / (media["sha256"] + ".wav")
    seconds = min(30, media["duration"])
    if not audio.exists():
        runner.run(
            [
                settings.ffmpeg,
                "-v",
                "error",
                "-y",
                "-i",
                media["path"],
                "-t",
                str(seconds),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s24le",
                str(audio),
            ]
        )
    result = transcribe(audio, settings, root, runner)
    return dict(
        speech_detected=bool(result["words"]),
        source_transcript=result["text"],
        speech_checked_seconds=seconds,
    )


def run_job(job, settings, root, runner):
    p = job["snapshot"]
    kind = job["kind"]
    payload = job.get("payload", {})
    project_dir = root / "projects" / p["id"]
    project_dir.mkdir(parents=True, exist_ok=True)
    if kind == "import":
        paths = []
        for raw in payload.get("paths", []):
            path = Path(raw).expanduser().resolve()
            paths += (
                sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS and f.is_file())
                if path.is_dir()
                else [path]
            )
        if not paths or len(paths) + len(p["media"]) > 50:
            raise ValueError("Import 1–50 video clips per project")
        media = list(p["media"])
        seen = {m["id"] for m in media}
        for i, path in enumerate(paths):
            runner.phase(f"importing {i + 1}/{len(paths)}")
            m = import_video(path, project_dir / "media", runner, settings.ffmpeg, settings.ffprobe)
            if m["id"] not in seen:
                media.append(m)
                seen.add(m["id"])
        return dict(
            media=media, revision=None, voice_result=None, translations={}, artifacts={}
        ), "review" if p["scenes"] else "draft"
    if kind == "analyze":
        if not p["media"]:
            raise ValueError("Import videos before analysis")
        provider = Provider(settings, runner)
        media = []
        cache = root / "cache/analysis"
        cache.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(p["media"]):
            runner.phase(f"analyzing {i + 1}/{len(p['media'])}")
            key = digest(
                dict(
                    source=m["sha256"],
                    provider=settings.provider,
                    endpoint=settings.endpoint,
                    model=settings.model,
                    version=1,
                )
            )
            path = cache / f"{key}.json"
            if path.exists():
                analysis = json.loads(path.read_text())
            else:
                analysis = analyze_clip(provider, m)
                path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
            speech = {}
            asr_path = Path(settings.asr_model).expanduser() if settings.asr_model else root / "models/asr"
            if asr_path.is_dir():
                runner.phase(f"checking original audio {i + 1}/{len(p['media'])}")
                speech = analyze_source_speech(m, settings, root, runner)
            media.append(dict(m, analysis=analysis, **speech))
        runner.phase("writing script")
        scenes = compose_script(provider, dict(p, media=media))
        return dict(
            media=media, scenes=scenes, revision=None, voice_result=None, translations={}, artifacts={}
        ), "review"
    if kind == "reference":
        source = Path(payload.get("path", "")).expanduser().resolve()
        ref = prepare_reference(source, project_dir / "references" / job["id"], runner, settings.ffmpeg)
        runner.phase("transcribing reference")
        text = payload.get("transcript", "").strip()
        if text:
            ref["transcript"] = text
        else:
            result = transcribe(Path(ref["path"]), settings, root, runner)
            if not result["text"].strip() or not result["words"]:
                raise ValueError(
                    "No speech recognized. Provide a clear recording or enter its transcript manually."
                )
            ref["transcript"] = result["text"]
            ref["language"] = result["language"]
        return dict(voice=ref, voice_result=None, artifacts={}), "voice_ready"
    if not p["revision"]:
        raise ValueError("Please approve the script before this step")
    if kind == "subtitles":
        runner.phase("translating subtitles")
        translations = translate_script(Provider(settings, runner), p, p["subtitle_mode"])
        return dict(translations=translations, artifacts={}), "subtitles_ready"
    if kind not in ("render", "audition"):
        raise ValueError("Unknown job type")
    if not p.get("voice"):
        raise ValueError("Import a reference recording and review its transcript")
    scenes = p["scenes"][:1] if kind == "audition" else p["scenes"]
    request_key = digest(
        dict(
            revision=p["revision"],
            voice=p["voice"],
            backend=settings.tts_backend,
            model=settings.tts_model,
            kind=kind,
        )
    )
    voice = p.get("voice_result")
    if (
        not voice
        or voice.get("request_key") != request_key
        or not all(
            Path(s["path"]).exists() and file_hash(Path(s["path"])) == s["sha256"] for s in voice["segments"]
        )
    ):
        voice = generate_voice(scenes, p["language"], p["voice"], settings, root, runner)
        voice["request_key"] = request_key
    if kind == "audition":
        return dict(artifacts=dict(p["artifacts"], audition=voice["segments"][0]["path"])), "voice_ready"
    # Translation has its own review step. Never call an LLM while rendering approved text.
    cues = caption_cues(p["scenes"], voice, p["translations"], p["language"], p["subtitle_mode"])
    folder = project_dir / "outputs" / job["id"]
    artifacts = render(
        p["media"],
        p["scenes"],
        [s["path"] for s in voice["segments"]],
        cues,
        p["subtitle_mode"],
        folder,
        runner,
        p["aspect"],
        settings.ffmpeg,
        settings.ffprobe,
    )
    script = folder / "script.json"
    script.write_text(
        json.dumps(
            dict(
                revision=p["revision"],
                brief=p["brief"],
                language=p["language"],
                subtitle_mode=p["subtitle_mode"],
                scenes=p["scenes"],
                translations=p["translations"],
                voice_model=voice.get("model"),
                voice_revision=voice.get("revision"),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    artifacts["script"] = str(script)
    return dict(artifacts=artifacts, voice_result=voice), "complete"

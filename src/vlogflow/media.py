from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from .captions import layout_caption, validate_cues, write_srt
from .processes import Runner

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024**2):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path, runner: Runner, ffprobe="ffprobe") -> dict:
    data = json.loads(
        runner.run([ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    )
    video = next(
        (
            s
            for s in data["streams"]
            if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    duration = float(data.get("format", {}).get("duration") or (video or {}).get("duration") or 0)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Media has no finite positive duration")
    result = dict(
        duration=duration,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
        width=0,
        height=0,
    )
    if video:
        w, h = int(video["width"]), int(video["height"])
        rotation = next((s.get("rotation", 0) for s in video.get("side_data_list", []) if "rotation" in s), 0)
        if abs(rotation) % 180 == 90:
            w, h = h, w
        result.update(width=w, height=h, codec=video.get("codec_name"))
    return result


def import_video(source: Path, directory: Path, runner: Runner, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Choose a supported local video file")
    if source.stat().st_size > 4 * 1024**3:
        raise ValueError("Each source video must be smaller than 4 GiB")
    sha = file_hash(source)
    folder = directory / sha[:24]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ("source" + source.suffix.lower())
    if not target.exists():
        temp = target.with_suffix(".partial")
        shutil.copyfile(source, temp)
        temp.replace(target)
    info = probe(target, runner, ffprobe)
    if not info["width"] or info["duration"] > 600:
        raise ValueError("Provide video clips up to 10 minutes each")
    frames = []
    for index, fraction in enumerate((0.12, 0.5, 0.86)):
        stamp = min(info["duration"] - 0.035, max(0, info["duration"] * fraction))
        frame = folder / f"frame-{index}.jpg"
        if not frame.exists():
            runner.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    f"{stamp:.4f}",
                    "-i",
                    str(target),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=512:512:force_original_aspect_ratio=decrease",
                    "-threads",
                    "2",
                    str(frame),
                ]
            )
        frames.append(dict(time=round(stamp, 3), path=str(frame)))
    return dict(id=sha[:24], sha256=sha, name=source.name, path=str(target), frames=frames, **info)


def prepare_reference(source: Path, directory: Path, runner: Runner, ffmpeg="ffmpeg"):
    if not source.is_file() or source.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ValueError("Choose a WAV, MP3, M4A or FLAC recording")
    directory.mkdir(parents=True, exist_ok=True)
    decoded = directory / "decoded.wav"
    runner.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-t",
            "31",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            "pcm_s24le",
            str(decoded),
        ]
    )
    audio, sr = sf.read(decoded, dtype="float32")
    if len(audio) / sr > 30.1:
        raise ValueError("Reference audio is longer than 30 seconds; provide a clean 4–30 second excerpt")
    if not np.isfinite(audio).all() or len(audio) / sr < 3:
        raise ValueError("Provide at least 3 seconds of clear speech")
    clipping = float(np.mean(np.abs(audio) >= 0.999))
    if clipping > 0.01:
        raise ValueError("Reference has significant clipping; record at a lower microphone level")
    frames = len(audio) // 480
    energy = np.sqrt(np.mean(audio[: frames * 480].reshape(frames, 480) ** 2, axis=1))
    active = np.where(energy > max(0.0015, float(np.max(energy)) * 0.04))[0]
    if len(active) * 0.02 < 2:
        raise ValueError("Reference contains silence or is too quiet; provide clear speech")
    start = max(0, int(active[0] * 480) - 2400)
    end = min(len(audio), int((active[-1] + 1) * 480) + 2400)
    audio = audio[start:end]
    peak = float(np.max(np.abs(audio)))
    gain = min(10 ** (10 / 20), 0.85 / max(peak, 1e-6))
    audio *= gain
    output = directory / "reference.wav"
    sf.write(output, audio, sr, subtype="PCM_24")
    return dict(
        path=str(output),
        sha256=file_hash(output),
        duration=len(audio) / sr,
        transcript="",
        clipping_fraction=clipping,
        warnings=["Reference is quiet; gain was increased. A closer microphone may improve similarity."]
        if gain > 2
        else [],
    )


def _geometry(media, aspect):
    if aspect == "portrait":
        return 720, 1280
    if aspect == "landscape":
        return 1280, 720
    w, h = media["width"], media["height"]
    scale = 720 / min(w, h)
    return max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)


def render(
    media,
    scenes,
    voices,
    cues,
    mode,
    directory: Path,
    runner: Runner,
    aspect="source",
    ffmpeg="ffmpeg",
    ffprobe="ffprobe",
):
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {m["id"]: m for m in media}
    fps = 30
    sr = 24000
    if len(voices) != len(scenes):
        raise ValueError("One voice segment is required for each scene")
    durations = []
    for scene, voice in zip(scenes, voices):
        info = sf.info(voice)
        duration = math.ceil(info.duration * fps - 1e-7) / fps
        if duration > scene["end"] - scene["start"] + 0.001:
            raise ValueError(
                f"Scene {scene['id']}: voice {duration:.2f}s exceeds selected range {scene['end'] - scene['start']:.2f}s. Shorten narration or extend the approved range."
            )
        durations.append(duration)
    total = sum(durations)
    validate_cues(cues, total)
    w, h = _geometry(by_id[scenes[0]["media_id"]], aspect)
    if max(w, h) > 2560:
        raise ValueError("Unusually wide source aspect; choose portrait or landscape")
    srt = directory / "subtitles.srt"
    write_srt(cues, mode, srt, total)
    parts = []
    offset = 0
    mixed = directory / "mix.wav"
    with sf.SoundFile(mixed, "w", samplerate=sr, channels=1, subtype="PCM_24") as audio_out:
        for i, (scene, voice, duration) in enumerate(zip(scenes, voices, durations)):
            runner.check()
            runner.phase(f"cutting {i + 1}/{len(scenes)}")
            source = by_id[scene["media_id"]]
            part = directory / f"part-{i:03}.mp4"
            parts.append(part)
            cmd = [ffmpeg, "-v", "error", "-y", "-ss", str(scene["start"]), "-i", source["path"]]
            active = [c for c in cues if c["start"] < offset + duration - 0.001 and c["end"] > offset + 0.001]
            for n, c in enumerate(active):
                png = directory / f"caption-{i:03}-{n:03}.png"
                img, _ = layout_caption(
                    c.get("zh", "") if mode != "en" else "", c.get("en", "") if mode != "zh" else "", w, h
                )
                img.save(png)
                cmd += ["-loop", "1", "-i", str(png)]
            filters = [
                f"[0:v]setpts=PTS-STARTPTS,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={fps}[v0]"
            ]
            for n, c in enumerate(active):
                a = max(0, c["start"] - offset)
                b = min(duration, c["end"] - offset)
                filters.append(
                    f"[v{n}][{n + 1}:v]overlay=0:0:enable='gte(t,{a:.6f})*lt(t,{b:.6f})'[v{n + 1}]"
                )
            cmd += [
                "-filter_complex",
                ";".join(filters),
                "-filter_complex_threads",
                "1",
                "-map",
                f"[v{len(active)}]",
                "-an",
                "-frames:v",
                str(round(duration * fps)),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                str(part),
            ]
            runner.run(cmd)
            voice_data, voice_sr = sf.read(voice, dtype="float32", always_2d=True)
            if voice_sr != sr:
                raise ValueError("Voice intermediate must use 24 kHz PCM")
            count = round(duration * sr)
            data = np.zeros(count, dtype=np.float32)
            mono = voice_data.mean(axis=1)
            data[: min(count, len(mono))] = mono[:count]
            rms = float(np.sqrt(np.mean(data**2)))
            gain = min(10 ** (-20 / 20) / max(rms, 1e-6), 0.75 / max(float(np.max(np.abs(data))), 1e-6))
            data *= gain
            if source["has_audio"] and scene["ambient"] > 0:
                ambient = directory / f"ambient-{i}.wav"
                runner.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        str(scene["start"]),
                        "-i",
                        source["path"],
                        "-t",
                        str(duration),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        str(sr),
                        "-c:a",
                        "pcm_s24le",
                        str(ambient),
                    ]
                )
                env, _ = sf.read(ambient, dtype="float32")
                n = min(len(env), count)
                data[:n] += env[:n] * scene["ambient"]
            peak = float(np.max(np.abs(data)))
            if peak > 0.75:
                data *= 0.75 / peak
            audio_out.write(data)
            offset += duration
    runner.phase("rendering")
    concat = directory / "parts.txt"
    concat.write_text("\n".join(f"file '{p.name}'" for p in parts) + "\n")
    output = directory / "video.mp4"
    runner.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(concat),
            "-i",
            str(mixed),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-t",
            f"{total:.6f}",
            str(output),
        ]
    )
    runner.phase("verifying")
    info = probe(output, runner, ffprobe)
    if not info["has_audio"] or abs(info["duration"] - total) > 0.12:
        raise ValueError("Final audio/video duration check failed")
    runner.run([ffmpeg, "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"])
    metadata = dict(
        duration=total,
        width=w,
        height=h,
        captions=len(cues),
        full_decode=True,
        sha256=file_hash(output),
        process_stats=runner.stats,
    )
    (directory / "validation.json").write_text(json.dumps(metadata, indent=2))
    return dict(video=str(output), srt=str(srt), validation=str(directory / "validation.json"))

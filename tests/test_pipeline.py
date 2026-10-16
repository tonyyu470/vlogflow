import time
from pathlib import Path

import numpy as np
import soundfile as sf
from test_media import clip as clip

from vlogflow.config import Settings
from vlogflow.jobs import Jobs
from vlogflow.media import file_hash
from vlogflow.store import Store


def wait(jobs, jid):
    for _ in range(400):
        j = jobs.store.job(jid)
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("Job did not finish")


def test_synthetic_pipeline_renders_all_subtitle_modes_with_reused_voice(tmp_path, clip, monkeypatch):
    # Explicit ASR/TTS fixture. Real import, store, jobs, subtitles and FFmpeg are exercised.
    import vlogflow.pipeline as pipeline

    calls = []

    def fake_voice(scenes, language, reference, settings, root, runner):
        calls.append(language)
        path = root / "synthetic.wav"
        sf.write(path, 0.1 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000), 24000)
        return {
            "key": "test-tone",
            "segments": [
                dict(scene_id="s", path=str(path), sha256=file_hash(path), duration=1, words=[[0, 1]])
            ],
        }

    monkeypatch.setattr(pipeline, "generate_voice", fake_voice)
    store = Store(tmp_path)
    jobs = Jobs(store)
    p = store.create("Synthetic test", "en")
    try:
        j = jobs.submit(p["id"], "import", {"paths": [str(clip)]})
        assert wait(jobs, j["id"])["status"] == "complete"
        p = store.get(p["id"])
        mid = p["media"][0]["id"]
        p = store.edit(
            p["id"],
            {
                "scenes": [
                    dict(
                        id="s",
                        media_id=mid,
                        start=0,
                        end=2,
                        narration="Hello",
                        description="Test pattern",
                        ambient=0,
                    )
                ]
            },
        )
        p = store.approve(p["id"], p["epoch"])
        p = store.edit(
            p["id"],
            {
                "voice": {"path": "unused", "sha256": "test", "transcript": "Test"},
                "translations": {"s": {"en": "Hello", "zh": "你好"}},
            },
        )
        for mode in ["en", "zh", "bilingual"]:
            p = store.edit(p["id"], {"subtitle_mode": mode})
            j = jobs.submit(p["id"], "render")
            result = wait(jobs, j["id"])
            assert result["status"] == "complete", result.get("error")
            p = store.get(p["id"])
            assert Path(p["artifacts"]["video"]).exists()
            srt = Path(p["artifacts"]["srt"]).read_text()
            assert ("你好" in srt) == (mode != "en") and ("Hello" in srt) == (mode != "zh")
        assert calls == ["en"]
    finally:
        jobs.close()


def test_job_retry_uses_current_snapshot_and_cancellation_preserves_revision(tmp_path):
    store = Store(tmp_path)
    jobs = Jobs(store)
    p = store.create("Walk", "en")
    try:
        j = jobs.submit(p["id"], "render")
        result = wait(jobs, j["id"])
        assert result["status"] == "failed" and "approve" in result["error"].lower()
        store.edit(p["id"], {"brief": "Updated"})
        retry = jobs.retry(j["id"])
        assert retry["snapshot"]["brief"] == "Updated"
        jobs.cancel(retry["id"])
        result = wait(jobs, retry["id"])
        assert result["status"] in ("cancelled", "failed") and store.get(p["id"])["revision"] is None
    finally:
        jobs.close()


def test_analysis_marks_existing_speech_for_script_ambient_choice(tmp_path, clip, monkeypatch):
    import subprocess

    import vlogflow.pipeline as pipeline
    from vlogflow.processes import Runner

    spoken = tmp_path / "spoken.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(clip),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=2",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(spoken),
        ],
        check=True,
    )
    monkeypatch.setattr(
        pipeline,
        "transcribe",
        lambda *a, **k: {"text": "Hello there", "words": [{"word": "Hello", "start": 0, "end": 1}]},
    )
    m = {"path": str(spoken), "has_audio": True, "sha256": file_hash(spoken), "duration": 2}
    result = pipeline.analyze_source_speech(m, Settings(), tmp_path, Runner(tmp_path / "logs"))
    assert result["speech_detected"] and result["source_transcript"] == "Hello there"
    assert result["speech_checked_seconds"] == 2

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vlogflow.config import Settings
from vlogflow.domain import digest
from vlogflow.processes import Runner


def test_cancelled_source_extraction_is_repeated_before_asr(tmp_path, monkeypatch):
    import vlogflow.pipeline as pipeline

    class InterruptedRunner:
        calls = 0

        def run(self, args):
            self.calls += 1
            if self.calls == 1:
                Path(args[-1]).write_bytes(b"partial waveform")
                raise RuntimeError("cancelled")
            sf.write(args[-1], np.ones(24000) * 0.05, 24000)

    def recognize(audio, *args):
        data, sr = sf.read(audio)
        assert len(data) == sr == 24000
        return {"text": "Hello", "words": [dict(word="Hello", start=0, end=1)]}

    monkeypatch.setattr(pipeline, "transcribe", recognize)
    runner = InterruptedRunner()
    media = dict(has_audio=True, sha256="source", path="unused.mp4", duration=1)
    with pytest.raises(RuntimeError, match="cancelled"):
        pipeline.analyze_source_speech(media, Settings(), tmp_path, runner)
    assert pipeline.analyze_source_speech(media, Settings(), tmp_path, runner)["speech_detected"]
    assert runner.calls == 2
    pipeline.analyze_source_speech(media, Settings(), tmp_path, runner)
    assert runner.calls == 2  # A completed, validated extraction is reused.


@pytest.mark.parametrize("broken", ['{"segments":', "{}", '{"segments": []}', '{"segments": [{}]}'])
def test_incomplete_voice_manifest_is_a_cache_miss(tmp_path, broken):
    from vlogflow.voice import _cached

    (tmp_path / "result.json").write_text(broken)
    assert _cached(tmp_path) is None


@pytest.mark.parametrize("broken", ['{"speaker.safetensors":', "{}"])
def test_incomplete_component_manifest_is_rebuilt(tmp_path, monkeypatch, broken):
    import vlogflow.models as models

    (tmp_path / "components.json").write_text(broken)
    calls = []

    def subset(source, prefix, target, excluded):
        target.write_bytes(b"synthetic tensor")
        calls.append(target.name)
        return {"sha256": models.file_hash(target)}

    monkeypatch.setattr(models, "subset_weights", subset)
    models.prepare_components(tmp_path / "model", tmp_path, Runner(tmp_path / "logs"))
    assert len(calls) == 4
    models.prepare_components(tmp_path / "model", tmp_path, Runner(tmp_path / "logs"))
    assert len(calls) == 4


def test_truncated_analysis_cache_is_regenerated(tmp_path, monkeypatch):
    import vlogflow.pipeline as pipeline

    settings = Settings()
    media = dict(id="m", has_audio=False, sha256="source", duration=2)
    key = digest(
        dict(
            source="source",
            provider=settings.provider,
            endpoint=settings.endpoint,
            model=settings.model,
            version=1,
        )
    )
    folder = tmp_path / "cache/analysis"
    folder.mkdir(parents=True)
    (folder / f"{key}.json").write_text('{"description":')
    analysis = dict(description="A test scene", evidence=[], uncertainties=[])
    monkeypatch.setattr(pipeline, "analyze_clip", lambda *args: analysis)
    monkeypatch.setattr(pipeline, "compose_script", lambda *args: [])
    job = dict(kind="analyze", snapshot=dict(id="p", media=[media]))
    update, _ = pipeline.run_job(job, settings, tmp_path, Runner(tmp_path / "logs"))
    assert update["media"][0]["analysis"] == analysis


def test_truncated_asr_cache_is_regenerated(tmp_path, monkeypatch):
    import vlogflow.voice as voice

    monkeypatch.setattr(voice, "verify_model", lambda *args: {})
    audio = tmp_path / "audio.wav"
    sf.write(audio, np.ones(24000) * 0.05, 24000)
    key = digest(
        dict(
            audio=voice.file_hash(audio),
            model=voice.manifest("asr")["revision"],
            language="",
            prompt="",
            version=1,
        )
    )
    output = tmp_path / "cache/asr" / f"{key}.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"words":')
    result = dict(text="Hello", language="en", words=[dict(word="Hello", start=0, end=1)])

    class Worker:
        def run(self, args, **kwargs):
            Path(args[args.index("--output") + 1]).write_text(json.dumps(result))

    assert voice.transcribe(audio, Settings(), tmp_path, Worker()) == result

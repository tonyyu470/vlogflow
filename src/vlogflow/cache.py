"""Completion markers are atomic; interrupted or malformed caches are misses."""

from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


def read_json(path: Path, valid=lambda data: True):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and valid(data):
            return data
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def valid_wave(path: Path, duration=None):
    try:
        with sf.SoundFile(path) as audio:
            if audio.samplerate != 24000 or audio.channels != 1 or not audio.frames:
                return False
            if duration is not None and (
                not math.isfinite(duration) or abs(audio.frames / audio.samplerate - duration) > 0.01
            ):
                return False
            return all(np.isfinite(block).all() for block in audio.blocks(blocksize=24000))
    except (OSError, RuntimeError, ValueError):
        return False

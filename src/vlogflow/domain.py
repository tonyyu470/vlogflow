from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Scene(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    media_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,64}$')
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    description: str = Field(default='', max_length=4000)
    narration: str = Field(min_length=1, max_length=2000)
    ambient: float = Field(default=0.12, ge=0, le=1)

    @field_validator('narration')
    @classmethod
    def nonblank(cls, text):
        if not text.strip():
            raise ValueError('Narration cannot be empty')
        return text.strip()


def validate_scenes(media: list[dict], scenes: list[dict]) -> list[dict]:
    if not 1 <= len(scenes) <= 100:
        raise ValueError('Please provide 1–100 scenes')
    sources = {m['id']: m for m in media}
    result = [Scene.model_validate(s).model_dump() for s in scenes]
    if len({s['id'] for s in result}) != len(result):
        raise ValueError('Scene IDs must be unique')
    for s in result:
        m = sources.get(s['media_id'])
        if not m:
            raise ValueError(f"Scene {s['id']}: source video does not exist")
        if s['end'] <= s['start'] or s['end'] > m['duration'] + .001:
            raise ValueError(f"Scene {s['id']}: selected range exceeds source duration")
    return result


class ProjectEdit(BaseModel):
    model_config = ConfigDict(extra='forbid')
    epoch: int
    brief: str | None = Field(default=None, min_length=1, max_length=10000)
    language: Literal['zh', 'en'] | None = None
    scenes: list[Scene] | None = None
    subtitle_mode: Literal['zh', 'en', 'bilingual'] | None = None
    translations: dict[str, dict[str, str]] | None = None
    aspect: Literal['source', 'portrait', 'landscape'] | None = None

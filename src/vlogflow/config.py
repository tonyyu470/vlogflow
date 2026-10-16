from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from platformdirs import user_data_path
from pydantic import BaseModel, ConfigDict, Field, field_validator


def data_root() -> Path:
    root = (
        Path(os.environ.get("VLOGFLOW_DATA_DIR", user_data_path("VlogFlow", appauthor=False)))
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama", "openai"] = "ollama"
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen3-vl:2b"
    api_key: str = ""
    tts_backend: Literal["mlx", "torch"] = "mlx"
    tts_model: str = ""
    asr_model: str = ""
    worker_python: str = sys.executable
    last_language: Literal["zh", "en"] = "zh"
    min_available_gib: float = Field(default=4.1, ge=4.1, le=128)
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"

    @field_validator("endpoint")
    @classmethod
    def valid_endpoint(cls, v):
        p = urlparse(v)
        if (
            p.scheme not in ("http", "https")
            or not p.hostname
            or p.username
            or p.password
            or p.query
            or p.fragment
        ):
            raise ValueError("Use an http(s) base URL without embedded credentials or query parameters")
        return v.rstrip("/")


def load_settings(root: Path) -> Settings:
    path = root / "settings.json"
    return Settings.model_validate_json(path.read_text()) if path.exists() else Settings()


def save_settings(root: Path, settings: Settings):
    temp = root / "settings.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(settings.model_dump_json(indent=2))
    temp.replace(root / "settings.json")


def public_settings(settings: Settings) -> dict:
    values = settings.model_dump()
    values["api_key"] = ""
    values["has_api_key"] = bool(settings.api_key)
    return values

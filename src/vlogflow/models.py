from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from .media import file_hash

ASSETS = Path(__file__).parent / "assets"


def manifest(kind):
    return json.loads((ASSETS / f"{kind}-manifest.json").read_text())


def verify_model(directory: Path, kind: str):
    record = manifest(kind)
    for item in record["files"]:
        path = directory / item["file"]
        if not path.is_file():
            raise ValueError(f"Model file missing: {item['file']}. Run vlogflow models download {kind}.")
        if item.get("bytes") and path.stat().st_size != item["bytes"]:
            raise ValueError(f"Model file size mismatch: {item['file']}")
        if item.get("sha256") and file_hash(path) != item["sha256"]:
            raise ValueError(f"Model checksum mismatch: {item['file']}. Re-download this model.")
    return record


def download_model(root: Path, kind: str):
    from huggingface_hub import hf_hub_download

    record = manifest(kind)
    target = root / "models" / kind
    target.mkdir(parents=True, exist_ok=True)
    for item in record["files"]:
        hf_hub_download(record["repo"], item["file"], revision=record["revision"], local_dir=target)
    verify_model(target, kind)
    return target


def subset_weights(source: Path, prefix: str, target: Path, excluded=()):
    """Stream raw tensor bytes without loading model tensors into memory."""
    temp = target.with_suffix(".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 * 1024**2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(f.read(n))
        selected = [
            (k, v)
            for k, v in header.items()
            if k != "__metadata__" and k.startswith(prefix) and k not in excluded
        ]
        if not selected:
            raise ValueError("No matching model weights")
        output = {}
        offset = 0
        for key, value in selected:
            begin, end = value["data_offsets"]
            size = end - begin
            if begin < 0 or size < 0 or 8 + n + end > source.stat().st_size:
                raise ValueError("Invalid safetensors offsets")
            output[key] = dict(value, data_offsets=[offset, offset + size])
            offset += size
        encoded = json.dumps(output, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        source_hash = hashlib.sha256()
        with temp.open("wb") as out:
            out.write(struct.pack("<Q", len(encoded)))
            out.write(encoded)
            for _, value in selected:
                begin, end = value["data_offsets"]
                f.seek(8 + n + begin)
                remaining = end - begin
                while remaining:
                    chunk = f.read(min(remaining, 1024**2))
                    if not chunk:
                        raise ValueError("Truncated model weights")
                    out.write(chunk)
                    source_hash.update(chunk)
                    remaining -= len(chunk)
    target_hash = hashlib.sha256()
    with temp.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n)
        while chunk := f.read(1024**2):
            target_hash.update(chunk)
    if source_hash.digest() != target_hash.digest():
        raise ValueError("Export changed tensor bytes")
    temp.replace(target)
    return dict(
        tensor_bytes=offset,
        raw_tensor_sha256=target_hash.hexdigest(),
        sha256=file_hash(target),
        byte_identical=True,
    )


def prepare_components(model: Path, target: Path, runner):
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "components.json"
    if marker.exists():
        records = json.loads(marker.read_text())
        if all((target / k).is_file() and file_hash(target / k) == v["sha256"] for k, v in records.items()):
            return target
    runner.check()
    specs = [
        ("speaker.safetensors", model / "model.safetensors", "speaker_encoder.", ()),
        ("encoder.safetensors", model / "speech_tokenizer/model.safetensors", "encoder.", ()),
        ("decoder.safetensors", model / "speech_tokenizer/model.safetensors", "decoder.", ()),
        (
            "talker-without-text-table.safetensors",
            model / "model.safetensors",
            "",
            ("talker.model.text_embedding.weight",),
        ),
    ]
    records = {}
    for name, source, prefix, excluded in specs:
        runner.check()
        records[name] = subset_weights(source, prefix, target / name, excluded)
    marker.write_text(json.dumps(records, indent=2))
    return target

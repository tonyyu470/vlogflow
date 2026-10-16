"""Bound multipart uploads while receiving, before any whole-file spooling."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException, Request
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header

from .cache import read_json, write_json
from .media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS

AUDIO_LIMIT = 64 * 1024**2
VIDEO_LIMIT = 4 * 1024**3
REQUEST_LIMIT = 8 * 1024**3 + 1024**2
FILE_LIMIT = 50


def display_name(raw: str):
    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    return Path(name).stem[:200] + Path(name).suffix.lower()


def uploaded_name(path: Path, folder: Path):
    if path.parent != folder.resolve():
        return None
    metadata = read_json(
        path.with_suffix(path.suffix + ".json"), lambda data: isinstance(data.get("name"), str)
    )
    return display_name(metadata["name"]) if metadata else None


async def receive_uploads(request: Request, folder: Path):
    content_type, options = parse_options_header(request.headers.get("content-type", ""))
    boundary = options.get(b"boundary", b"")
    if content_type != b"multipart/form-data" or not 1 <= len(boundary) <= 200:
        raise HTTPException(400, "Expected multipart files with a valid boundary")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            size = int(length)
        except ValueError:
            raise HTTPException(400, "Invalid Content-Length") from None
        if size < 0:
            raise HTTPException(400, "Invalid Content-Length")
        if size > REQUEST_LIMIT:
            raise HTTPException(413, "Upload batch exceeds 8 GiB; use smaller batches or local paths")
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    part = {}
    completed = False
    successful = False

    def begin():
        part.clear()
        part.update(headers={}, field=bytearray(), value=bytearray(), header_bytes=0, count=0)

    def header(which, data, start, end):
        part["header_bytes"] += end - start
        if part["header_bytes"] > 8192:
            raise HTTPException(413, "Upload part headers are too large")
        part[which].extend(data[start:end])

    def header_end():
        part["headers"][bytes(part["field"]).lower()] = bytes(part["value"])
        part["field"].clear()
        part["value"].clear()

    def headers_finished():
        _, params = parse_options_header(part["headers"].get(b"content-disposition", b""))
        if params.get(b"name") != b"files" or b"filename" not in params:
            raise HTTPException(400, "Only file parts named 'files' are accepted")
        if len(entries) >= FILE_LIMIT:
            raise HTTPException(413, "Upload 1–50 files at a time")
        name = display_name(params[b"filename"].decode("utf-8", errors="replace"))
        suffix = Path(name).suffix.lower()
        if suffix not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
            raise HTTPException(400, "Unsupported audio or video extension")
        path = folder / (uuid.uuid4().hex + suffix)
        entry = dict(path=path, partial=path.with_suffix(".partial"), name=name)
        entries.append(entry)
        part["stream"] = entry["partial"].open("xb")
        part["maximum"] = AUDIO_LIMIT if suffix in AUDIO_EXTENSIONS else VIDEO_LIMIT

    def data_received(data, start, end):
        part["count"] += end - start
        if part["count"] > part["maximum"]:
            raise HTTPException(413, "Uploaded file is too large (audio 64 MiB; video 4 GiB)")
        part["stream"].write(data[start:end])

    def end_part():
        if "stream" not in part or not part["count"]:
            raise HTTPException(400, "Uploaded file is empty")
        part["stream"].close()

    def end():
        nonlocal completed
        completed = True

    parser = MultipartParser(
        boundary,
        dict(
            on_part_begin=begin,
            on_header_field=lambda *args: header("field", *args),
            on_header_value=lambda *args: header("value", *args),
            on_header_end=header_end,
            on_headers_finished=headers_finished,
            on_part_data=data_received,
            on_part_end=end_part,
            on_end=end,
        ),
    )
    try:
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > REQUEST_LIMIT:
                raise HTTPException(413, "Upload batch exceeds 8 GiB; use smaller batches or local paths")
            parser.write(chunk)
        parser.finalize()
        if not completed or not entries:
            raise HTTPException(400, "Incomplete multipart upload")
        for entry in entries:
            entry["partial"].replace(entry["path"])
            write_json(entry["path"].with_suffix(entry["path"].suffix + ".json"), dict(name=entry["name"]))
        successful = True
        return [str(entry["path"]) for entry in entries]
    except MultipartParseError:
        raise HTTPException(400, "Malformed multipart upload") from None
    finally:
        if "stream" in part:
            part["stream"].close()
        if not successful:
            for entry in entries:
                entry["partial"].unlink(missing_ok=True)
                entry["path"].unlink(missing_ok=True)
                entry["path"].with_suffix(entry["path"].suffix + ".json").unlink(missing_ok=True)

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from test_media import clip as clip
from test_pipeline import wait

from vlogflow.api import create_app


def multipart(parts):
    chunks = []
    for name, data in parts:
        chunks.extend(
            [
                b'--test\r\nContent-Disposition: form-data; name="files"; filename="'
                + name.encode()
                + b'"\r\nContent-Type: application/octet-stream\r\n\r\n',
                *data,
                b"\r\n",
            ]
        )
    return chunks + [b"--test--\r\n"]


def streamed_request(chunks, length=None):
    received = []

    async def receive():
        index = len(received)
        received.append(index)
        return dict(type="http.request", body=chunks[index], more_body=index < len(chunks) - 1)

    headers = [(b"content-type", b"multipart/form-data; boundary=test")]
    if length is not None:
        headers.append((b"content-length", str(length).encode()))
    return Request(dict(type="http", method="POST", headers=headers), receive), received


def test_audio_limit_stops_receiving_chunked_body_and_cleans_partial_files(tmp_path, monkeypatch):
    from vlogflow import uploads

    monkeypatch.setattr(uploads, "AUDIO_LIMIT", 4)
    chunks = multipart([("voice.wav", [b"123", b"456", b"unread" * 100])])
    request, received = streamed_request(chunks)
    with pytest.raises(HTTPException) as error:
        asyncio.run(uploads.receive_uploads(request, tmp_path))
    assert error.value.status_code == 413
    assert len(received) < len(chunks)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("declared", [False, True])
def test_aggregate_limit_is_enforced_with_or_without_content_length(tmp_path, monkeypatch, declared):
    from vlogflow import uploads

    monkeypatch.setattr(uploads, "REQUEST_LIMIT", 200)
    chunks = multipart([("one.mp4", [b"x" * 100]), ("two.mp4", [b"y" * 100])])
    request, received = streamed_request(chunks, sum(map(len, chunks)) if declared else None)
    with pytest.raises(HTTPException) as error:
        asyncio.run(uploads.receive_uploads(request, tmp_path))
    assert error.value.status_code == 413
    assert len(received) < len(chunks)
    assert not list(tmp_path.iterdir())


def test_file_count_and_truncated_request_remove_the_entire_batch(tmp_path, monkeypatch):
    from vlogflow import uploads

    monkeypatch.setattr(uploads, "FILE_LIMIT", 1)
    chunks = multipart([("one.mp4", [b"one"]), ("two.mp4", [b"two"])])
    for body in [chunks, chunks[:2]]:
        request, _ = streamed_request(body)
        with pytest.raises(HTTPException):
            asyncio.run(uploads.receive_uploads(request, tmp_path))
        assert not list(tmp_path.iterdir())


def test_uploaded_filenames_remain_recognizable_and_duplicate_names_are_distinct(tmp_path, clip):
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as client:
        headers = {"X-VlogFlow-Token": client.get("/api/bootstrap").json()["token"]}
        pid = client.post("/api/projects", headers=headers, json={"brief": "Upload test"}).json()["id"]
        data = clip.read_bytes()
        response = client.post(
            f"/api/projects/{pid}/uploads",
            headers=headers,
            files=[("files", ("entrance.mp4", data)), ("files", ("entrance.mp4", data + b"\n"))],
        )
        assert response.status_code == 200, response.text
        paths = response.json()["paths"]
        assert len(set(paths)) == 2 and all(Path(p).name != "entrance.mp4" for p in paths)
        job = client.post(
            f"/api/projects/{pid}/jobs", headers=headers, json={"kind": "import", "paths": paths}
        ).json()
        result = wait(client.app.state.jobs, job["id"])
        assert result["status"] == "complete", result.get("error")
        media = client.get(f"/api/projects/{pid}").json()["media"]
        assert [m["name"] for m in media] == ["entrance.mp4", "entrance.mp4"]
        assert media[0]["id"] != media[1]["id"]

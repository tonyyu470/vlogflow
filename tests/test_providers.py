import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vlogflow.config import Settings
from vlogflow.providers import Provider, compose_script, translate_script


@contextmanager
def server(responses):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            response = responses[min(len(calls) - 1, len(responses) - 1)]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"message": {"content": response}, "choices": [{"message": {"content": response}}]}
                ).encode()
            )

        def log_message(self, *args):
            pass

    s = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{s.server_port}", calls
    finally:
        s.shutdown()
        t.join()
        s.server_close()


def test_ollama_sends_images_and_unloads_model(tmp_path):
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"image")
    with server(['{"ok":true}']) as (endpoint, calls):
        assert Provider(Settings(endpoint=endpoint)).complete("inspect", [img]) == {"ok": True}
        body = calls[0][1]
        assert calls[0][0] == "/api/chat" and body["keep_alive"] == 0
        assert body["messages"][-1]["images"] == ["aW1hZ2U="]


def test_openai_serializes_multimodal_content_and_bounds_json_retries():
    with server(["not json", "still not json", '{"ok":true}']) as (endpoint, calls):
        with pytest.raises(ValueError, match="JSON"):
            Provider(Settings(provider="openai", endpoint=endpoint, api_key="secret")).complete("inspect")
        assert len(calls) == 2 and calls[0][0] == "/chat/completions"


def test_provider_cannot_invent_source_or_out_of_range_selection():
    invalid = json.dumps(
        {
            "scenes": [
                dict(id="s", media_id="missing", start=0, end=10, description="x", narration="Hi", ambient=0)
            ]
        }
    )
    with server([invalid]) as (endpoint, calls):
        with pytest.raises(ValueError):
            compose_script(
                Provider(Settings(endpoint=endpoint)),
                {
                    "brief": "walk",
                    "language": "en",
                    "media": [{"id": "m", "duration": 2, "analysis": {"description": "street"}}],
                },
            )
        assert len(calls) == 2


def test_translation_cannot_overwrite_approved_source_language():
    with server(['{"translations":{"s":{"zh":"你好","en":"Changed!"}}}']) as (endpoint, calls):
        p = {"language": "en", "scenes": [{"id": "s", "narration": "Hello."}]}
        result = translate_script(Provider(Settings(endpoint=endpoint)), p, "bilingual")
        assert result == {"s": {"en": "Hello.", "zh": "你好"}}

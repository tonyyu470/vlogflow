import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vlogflow.config import Settings
from vlogflow.providers import Provider, analyze_clip, compose_script, translate_script


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
            body = (
                response
                if not isinstance(response, str)
                else {"message": {"content": response}, "choices": [{"message": {"content": response}}]}
            )
            self.wfile.write(json.dumps(body).encode())

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


@pytest.mark.parametrize(
    ("provider", "malformed"),
    [("ollama", []), ("ollama", {"message": None}), ("openai", {"choices": [None]})],
)
def test_malformed_response_envelope_retries_and_can_recover(provider, malformed):
    with server([malformed, '{"ok":true}']) as (endpoint, calls):
        assert Provider(Settings(provider=provider, endpoint=endpoint)).complete("inspect") == {"ok": True}
        assert len(calls) == 2


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


def test_script_schema_bounds_offsets_by_the_selected_source():
    scenes = [dict(id="s1", media_id="short", start=0, end=2, narration="A short walk.")]
    project = {
        "brief": "walk",
        "language": "en",
        "media": [{"id": "short", "duration": 2}, {"id": "long", "duration": 5}],
    }
    with server([json.dumps({"scenes": scenes})]) as (endpoint, calls):
        result = compose_script(Provider(Settings(endpoint=endpoint)), project)
        assert result[0]["start"] == 0 and result[0]["end"] == 2
        variants = calls[0][1]["format"]["properties"]["scenes"]["items"]["anyOf"]
        limits = {
            v["properties"]["media_id"]["const"]: (
                v["properties"]["start"]["maximum"],
                v["properties"]["end"]["maximum"],
            )
            for v in variants
        }
        assert limits == {"short": (2, 2), "long": (5, 5)}


def test_translation_cannot_overwrite_approved_source_language():
    with server(['{"translations":{"s":{"zh":"你好","en":"Changed!"}}}']) as (endpoint, calls):
        p = {"language": "en", "scenes": [{"id": "s", "narration": "Hello."}]}
        result = translate_script(Provider(Settings(endpoint=endpoint)), p, "bilingual")
        assert result == {"s": {"en": "Hello.", "zh": "你好"}}


def test_ollama_finished_json_in_thinking_is_validated_as_analysis():
    analysis = dict(
        description="A quiet park", evidence=[dict(time=1, observation="A tree")], uncertainties=[]
    )
    response = dict(done=True, done_reason="stop", message=dict(content="", thinking=json.dumps(analysis)))
    with server([response]) as (endpoint, calls):
        result = analyze_clip(Provider(Settings(endpoint=endpoint)), dict(duration=3, frames=[]))
        assert result == analysis
        assert len(calls) == 1


def test_analysis_requests_bounded_output_schema():
    analysis = dict(description="A quiet park", evidence=[], uncertainties=[])
    with server([json.dumps(analysis)]) as (endpoint, calls):
        assert analyze_clip(Provider(Settings(endpoint=endpoint)), dict(duration=3, frames=[])) == analysis
        schema = calls[0][1]["format"]
        assert isinstance(schema, dict) and schema["type"] == "object"
        assert schema["properties"]["description"]["maxLength"] <= 400
        assert schema["properties"]["evidence"]["maxItems"] <= 3


def test_analysis_rejects_repeated_unbounded_descriptions():
    analysis = dict(description="Repeated scene. " * 100, evidence=[], uncertainties=[])
    with server([json.dumps(analysis)]) as (endpoint, calls):
        with pytest.raises(ValueError, match="description"):
            analyze_clip(Provider(Settings(endpoint=endpoint)), dict(duration=3, frames=[]))
        assert len(calls) == 2


def test_token_limit_error_is_reported_without_accepting_incomplete_output():
    response = dict(
        done=True, done_reason="length", message=dict(content="", thinking='{"description":"Repeated')
    )
    with server([response]) as (endpoint, calls):
        with pytest.raises(ValueError, match="truncated|token limit"):
            Provider(Settings(endpoint=endpoint)).complete("inspect")
        assert len(calls) <= 2


@pytest.mark.parametrize(
    "thinking",
    [
        'I will inspect the image. {"ok":true}',
        '```json\n{"ok":true}\n```',
        '{"wrong_key":"not an analysis"}',
    ],
)
def test_thinking_fallback_never_extracts_prose_or_bypasses_analysis_validation(thinking):
    response = dict(done=True, done_reason="stop", message=dict(content="", thinking=thinking))
    with server([response]) as (endpoint, calls):
        with pytest.raises(ValueError):
            analyze_clip(Provider(Settings(endpoint=endpoint)), dict(duration=3, frames=[]))
        assert len(calls) == 2


def test_final_content_takes_precedence_over_thinking_and_openai_never_falls_back():
    response = dict(
        done=True, done_reason="stop", message=dict(content='{"ok":true}', thinking='{"ok":false}')
    )
    with server([response]) as (endpoint, calls):
        assert Provider(Settings(endpoint=endpoint)).complete("inspect") == {"ok": True}
    response = dict(choices=[dict(message=dict(content="", reasoning_content='{"ok":true}'))])
    with server([response]) as (endpoint, calls):
        with pytest.raises(ValueError):
            Provider(Settings(provider="openai", endpoint=endpoint)).complete("inspect")

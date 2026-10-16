from pathlib import Path

from fastapi.testclient import TestClient

from vlogflow.api import create_app


def test_api_requires_same_origin_token_and_rejects_unregistered_paths(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as c:
        assert c.post("/api/projects", json={"brief": "Walk", "language": "en"}).status_code == 403
        token = c.get("/api/bootstrap").json()["token"]
        h = {"X-VlogFlow-Token": token}
        assert (
            c.post(
                "/api/projects",
                headers=h | {"Origin": "https://evil.example"},
                json={"brief": "Walk", "language": "en"},
            ).status_code
            == 403
        )
        r = c.post("/api/projects", headers=h, json={"brief": "Walk", "language": "en"})
        assert r.status_code == 200
        pid = r.json()["id"]
        assert c.get(f"/api/projects/{pid}/files/video").status_code == 404
        assert c.get("/api/files?path=/etc/passwd").status_code == 404
        assert c.get("/api/bootstrap", headers={"Host": "evil.example"}).status_code == 400


def test_settings_hide_key_and_empty_field_preserves_existing_key(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as c:
        h = {"X-VlogFlow-Token": c.get("/api/bootstrap").json()["token"]}
        assert c.put("/api/settings", headers=h, json={"api_key": "top-secret"}).status_code == 200
        assert "top-secret" not in c.get("/api/settings").text
        c.put("/api/settings", headers=h, json={"api_key": "", "model": "test"})
        assert c.get("/api/settings").json()["has_api_key"]


def test_reference_upload_is_staged_as_registered_input_and_validated(tmp_path):
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as c:
        h = {"X-VlogFlow-Token": c.get("/api/bootstrap").json()["token"]}
        p = c.post("/api/projects", headers=h, json={"brief": "Walk", "language": "en"}).json()
        r = c.post(
            f"/api/projects/{p['id']}/uploads",
            headers=h,
            files=[("files", ("../../escape.m4a", b"not audio", "audio/mp4"))],
        )
        assert r.status_code == 200 and not (tmp_path.parent / "escape.m4a").exists()
        assert Path(r.json()["paths"][0]).parent == tmp_path / "projects" / p["id"] / "uploads"

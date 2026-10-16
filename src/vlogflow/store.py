from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .domain import digest, validate_scenes


class Conflict(ValueError):
    pass


def uid() -> str:
    return uuid.uuid4().hex


class Store:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "vlogflow.sqlite3"
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS revisions (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, data TEXT NOT NULL, created REAL NOT NULL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _get(self, db, table, key):
        row = db.execute(f"SELECT data FROM {table} WHERE id=?", (key,)).fetchone()
        if not row:
            raise KeyError(key)
        return json.loads(row[0])

    def _put_project(self, db, p):
        p["updated"] = time.time()
        db.execute(
            "UPDATE projects SET data=?,updated=? WHERE id=?",
            (json.dumps(p, ensure_ascii=False), p["updated"], p["id"]),
        )

    def create(self, brief: str, language: str):
        if language not in ("zh", "en") or not brief.strip():
            raise ValueError("Provide a brief and a supported narration language")
        p = dict(
            id=uid(),
            brief=brief,
            language=language,
            media=[],
            scenes=[],
            revision=None,
            epoch=0,
            subtitle_mode="bilingual",
            translations={},
            voice=None,
            voice_result=None,
            artifacts={},
            aspect="source",
            status="draft",
            updated=time.time(),
        )
        with self.db() as db:
            db.execute("INSERT INTO projects VALUES(?,?,?)", (p["id"], json.dumps(p), p["updated"]))
        return p

    def get(self, key):
        with self.db() as db:
            return self._get(db, "projects", key)

    def list(self):
        with self.db() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM projects ORDER BY updated DESC")]

    def revision(self, key):
        with self.db() as db:
            return self._get(db, "revisions", key)

    def edit(self, key, changes, epoch=None):
        allowed = {"brief", "language", "media", "scenes", "subtitle_mode", "translations", "voice", "aspect"}
        if set(changes) - allowed:
            raise ValueError("Unknown project fields")
        with self.db() as db:
            p = self._get(db, "projects", key)
            if epoch is not None and p["epoch"] != epoch:
                raise Conflict("Project changed; reload before saving")
            p.update(changes)
            if p["scenes"]:
                p["scenes"] = validate_scenes(p["media"], p["scenes"])
            p["epoch"] += 1
            p["artifacts"] = {}
            if set(changes) & {"brief", "language", "media", "scenes"}:
                p.update(
                    revision=None,
                    translations={},
                    voice_result=None,
                    status="review" if p["scenes"] else "draft",
                )
            elif "voice" in changes:
                p.update(voice_result=None, status="voice_ready" if p["voice"] else "approved")
            else:
                p["status"] = "subtitles_ready" if p["revision"] else "draft"
            self._put_project(db, p)
            return p

    def approve(self, key, epoch):
        with self.db() as db:
            p = self._get(db, "projects", key)
            if p["epoch"] != epoch:
                raise Conflict("Project changed; reload before approval")
            p["scenes"] = validate_scenes(p["media"], p["scenes"])
            snapshot = {k: p[k] for k in ("id", "language", "scenes")}
            rev = digest(snapshot)
            db.execute(
                "INSERT OR IGNORE INTO revisions VALUES(?,?)", (rev, json.dumps(snapshot, ensure_ascii=False))
            )
            p.update(revision=rev, status="approved", epoch=p["epoch"] + 1, artifacts={})
            self._put_project(db, p)
            return p

    def create_job(self, project_id, kind):
        with self.db() as db:
            p = self._get(db, "projects", project_id)
            j = dict(
                id=uid(),
                project_id=project_id,
                kind=kind,
                epoch=p["epoch"],
                snapshot=p,
                status="queued",
                phase="queued",
                error=None,
                created=time.time(),
                cancel_requested=False,
            )
            db.execute("INSERT INTO jobs VALUES(?,?,?,?)", (j["id"], project_id, json.dumps(j), j["created"]))
            return j

    def job(self, key):
        with self.db() as db:
            return self._get(db, "jobs", key)

    def jobs(self, project_id):
        with self.db() as db:
            return [
                json.loads(r[0])
                for r in db.execute(
                    "SELECT data FROM jobs WHERE project_id=? ORDER BY created DESC", (project_id,)
                )
            ]

    def update_job(self, key, **changes):
        with self.db() as db:
            j = self._get(db, "jobs", key)
            j.update(changes)
            db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(j), key))
            return j

    def publish(self, job_id, changes, status):
        with self.db() as db:
            j = self._get(db, "jobs", job_id)
            p = self._get(db, "projects", j["project_id"])
            accepted = p["epoch"] == j["epoch"] and not j["cancel_requested"]
            if accepted:
                p.update(changes)
                p["status"] = status
                if j["kind"] in ("import", "analyze", "reference", "subtitles"):
                    p["epoch"] += 1
                self._put_project(db, p)
            j["status"] = "complete" if accepted else ("cancelled" if j["cancel_requested"] else "superseded")
            j["phase"] = j["status"]
            db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(j), job_id))
            return accepted

    def recover(self):
        with self.db() as db:
            for key, raw in db.execute("SELECT id,data FROM jobs").fetchall():
                j = json.loads(raw)
                if j["status"] in ("queued", "running", "resource_wait"):
                    j.update(
                        status="interrupted",
                        phase="interrupted",
                        error="Application restarted. Retry to reuse completed stages.",
                    )
                    db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(j), key))

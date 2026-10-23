"""Tests for the Liara platform tools against an in-process fake of the Liara API.

The fake implements the endpoints the tools use (projects, releases, sources, rollback, disks, /v1/me) and
also serves the app's own health URLs, so the full deploy -> smoke -> rollback loop runs without network.
Run: python -m pytest -q infra/tests   (or python -m unittest discover infra/tests)
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

INFRA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFRA))
os.environ["LIARA_POLL_SECONDS"] = "0.01"

import liara_apply  # noqa: E402
import liara_deploy  # noqa: E402
from liara_api import redact  # noqa: E402

# Built at runtime so secret scanners do not flag a test fixture as a real token.
FAKE_JWT = ".".join(["eyJ" + "hbGciOiJIUzI1NiJ9", "eyJ" + "zdWIiOiJ0ZXN0LWZpeHR1cmUifQ", "c2lnbmF0dXJlLWZha2U"])
CAPS = {"basic": {"price": 100, "zeroDowntime": False, "availableBackupInterval": [], "maxDeploymentCountInDay": 20,
                  "buildTimeout": 5, "maxSourceSize": 128},
        "standard": {"price": 200, "zeroDowntime": True, "availableBackupInterval": ["daily", "manual"],
                     "maxDeploymentCountInDay": 50, "buildTimeout": 15, "maxSourceSize": 256}}


class Fake:
    """Mutable Liara account state + behaviour switches for one test."""

    def __init__(self):
        self.calls: list[tuple[str, str, object]] = []
        self.projects = {"shop": {"_id": "p1", "project_id": "shop", "planID": "small-g2", "bundlePlanID": "basic",
                                  "scale": 1, "zeroDowntime": False, "readOnlyRootFilesystem": False,
                                  "network": {"_id": "n1", "name": "net"},
                                  "envs": [{"key": "API_KEY", "value": "s3cr3t-value", "encrypted": False}]}}
        self.disks = {"p1": [{"name": "data", "size": 1}]}
        self.releases = [{"_id": "r1", "type": "DEPLOY", "state": "READY", "createdAt": "2020-01-01T00:00:00Z", "tag": "v1"}]
        self.current = "r1"
        self.next_state = "READY"        # state a new release ends in
        self.new_release_takes_traffic = True
        self.healthy_after_deploy = True  # what /healthz answers once the new release is current
        self.rollback_status = 200

    def healthy(self):
        return self.current == "r1" or self.healthy_after_deploy


def make_server(fake: Fake):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                return json.loads(raw) if raw and self.headers.get("Content-Type", "").startswith("application/json") else raw
            except ValueError:
                return raw

        def handle_any(self, method):
            u = urlparse(self.path)
            p, q = u.path, parse_qs(u.query)
            body = self._body() if method == "POST" else None
            fake.calls.append((method, p, body if not isinstance(body, bytes) else f"<{len(body)} bytes>"))
            if p == "/healthz":  # the app itself
                return self._send(200 if fake.healthy() else 503, {"ok": fake.healthy()})
            if p.startswith("/v") and self.headers.get("Authorization") != "Bearer test-token":
                return self._send(401, {"message": "bad token"})
            if p.startswith("/v") and q.get("teamID") != ["team1"]:
                return self._send(404, {"message": "not found"})
            if p == "/v1/me":
                return self._send(200, {"team": {"balance": 100000}, "plans": {
                    "projects": {"small-g2": {"price": 500}, "medium-g2": {"price": 900}},
                    "projectBundlePlans": {"small-g2": CAPS, "medium-g2": CAPS}}})
            if p == "/v1/projects":
                return self._send(200, {"projects": list(fake.projects.values())})
            if p == "/v1/networks":
                return self._send(200, {"networks": [{"_id": "n1", "name": "net"}]})
            parts = p.strip("/").split("/")
            if parts[:2] == ["v1", "projects"] and len(parts) == 3 and method == "GET":
                pr = fake.projects.get(parts[2])
                return self._send(200, {"project": pr}) if pr else self._send(404, {"message": "no app"})
            if parts[:2] == ["v1", "projects"] and parts[3:] == ["disks"]:
                return self._send(200, {"disks": fake.disks.get(parts[2], []), "mounts": []})
            if parts[:2] == ["v1", "projects"] and parts[3:] == ["releases"]:
                return self._send(200, {"total": len(fake.releases), "currentRelease": fake.current,
                                        "releases": list(reversed(fake.releases))})
            if parts[:2] == ["v2", "projects"] and parts[3:] == ["sources"]:
                return self._send(201, {"sourceID": "src1"})
            if parts[:2] == ["v2", "projects"] and parts[3:] == ["releases"]:
                rid = f"r{len(fake.releases) + 1}"
                fake.releases.append({"_id": rid, "type": "DEPLOY", "state": fake.next_state, "createdAt": "2020-01-02T00:00:00Z"})
                if fake.next_state == "READY" or (fake.next_state == "UNHEALTHY" and fake.new_release_takes_traffic):
                    fake.current = rid
                return self._send(201, {"releaseID": rid})
            if parts[:2] == ["v1", "releases"] and len(parts) == 3:
                r = next((x for x in fake.releases if x["_id"] == parts[2]), None)
                return self._send(200, {"release": r}) if r else self._send(404, {})
            if parts[:2] == ["v2", "releases"] and parts[3:] == ["build-logs"]:
                return self._send(200, {"buildOutput": [{"line": "Step 1/9 : FROM python", "stream": "STDOUT"},
                                                        {"line": "token=" + FAKE_JWT, "stream": "STDERR"}]})
            if parts[:2] == ["v1", "releases"] and parts[3:] == ["rollback"]:
                if fake.rollback_status != 200:
                    return self._send(fake.rollback_status, {"data": {"code": "rollback_not_supported"}})
                rid = f"r{len(fake.releases) + 1}"
                fake.releases.append({"_id": rid, "type": "DEPLOY", "state": "READY", "createdAt": "2020-01-03T00:00:00Z"})
                fake.current = "r1"  # serving the old image again
                return self._send(200, {})
            if method == "POST":
                return self._send(200, {})
            if method == "DELETE":
                return self._send(204, {})
            return self._send(404, {"message": "unhandled " + p})

        def do_GET(self):
            self.handle_any("GET")

        def do_POST(self):
            self.handle_any("POST")

        def do_DELETE(self):
            self.handle_any("DELETE")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Base(unittest.TestCase):
    def setUp(self):
        self.fake = Fake()
        self.srv = make_server(self.fake)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.env = {"LIARA_API_TOKEN": "test-token", "LIARA_TEAM_ID": "team1", "LIARA_API_BASE": self.url}
        self._old = {k: os.environ.get(k) for k in [*self.env, "GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ACTIONS"]}
        os.environ.update(self.env)
        for k in ("GITHUB_STEP_SUMMARY", "GITHUB_OUTPUT", "GITHUB_ACTIONS"):
            os.environ.pop(k, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self._make_repo()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _make_repo(self):
        r = self.repo
        (r / "app").mkdir()
        (r / "app" / "main.py").write_text("print('hi')\n")
        (r / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (r / "data").mkdir()
        (r / "data" / "x.db").write_text("db")
        (r / ".liaraignore").write_text("data\n**/*.db\ndocs\n")
        (r / "docs").mkdir()
        (r / "docs" / "a.md").write_text("doc")
        manifest = {"apps": [{"name": "shop", "url": self.url, "platform": "docker", "port": 8080, "plan": "small-g2",
                              "bundle": "basic", "network": "net",
                              "disks": [{"name": "data", "sizeGB": 1, "mountTo": "/data"}],
                              "envs": {"required": ["API_KEY"]},
                              "healthCheck": {"command": ["CMD", "true"], "interval": 30, "timeout": 10, "retries": 3},
                              "smoke": {"attempts": 2, "checks": [{"path": "/healthz", "expectJson": {"ok": True}}]}}]}
        (r / "infra").mkdir()
        (r / "infra" / "m.json").write_text(json.dumps(manifest))
        g = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
        subprocess.run(["git", "add", "-A", "-f"], cwd=r, check=True)
        subprocess.run([*g, "commit", "-q", "-m", "init"], cwd=r, check=True)
        self.manifest = str(r / "infra" / "m.json")

    def deploy(self, *extra):
        out = io.StringIO()
        with redirect_stdout(out):
            code = liara_deploy.main(["deploy", "--manifest", self.manifest, "--root", str(self.repo), "--message", "v9", *extra])
        return code, out.getvalue()

    def posts(self, prefix):
        return [c for c in self.fake.calls if c[0] == "POST" and c[1].startswith(prefix)]


class TestPackaging(Base):
    def test_ignore_rules(self):
        pats = ["data", "**/*.db", "docs", "/build", "frontend/dist", "*.log"]
        ig = liara_deploy.is_ignored
        self.assertTrue(ig("data/x.json", pats))
        self.assertTrue(ig("a/b/c.db", pats))
        self.assertTrue(ig("build/x", pats))
        self.assertFalse(ig("src/build/x", pats))  # anchored
        self.assertTrue(ig("frontend/dist/index.html", pats))
        self.assertTrue(ig("deep/err.log", pats))
        self.assertFalse(ig("app/main.py", pats))

    def test_package_is_deterministic_and_filtered(self):
        a1, files = liara_deploy.package(self.repo)
        a2, _ = liara_deploy.package(self.repo)
        self.assertEqual(a1, a2)
        self.assertIn("app/main.py", files)
        self.assertNotIn("data/x.db", files)
        self.assertNotIn("docs/a.md", files)

    def test_package_subdirectory_as_context(self):
        archive, files = liara_deploy.package(self.repo, subdir="app")
        self.assertEqual(files, ["main.py"])
        with self.assertRaises(SystemExit):
            liara_deploy.package(self.repo, subdir="nope")

    def test_dirty_tree_refused(self):
        (self.repo / "app" / "main.py").write_text("changed\n")
        with self.assertRaises(SystemExit):
            liara_deploy.package(self.repo)

    def test_release_body_never_contains_envs(self):
        app = json.loads(Path(self.manifest).read_text())["apps"][0]
        body = liara_deploy.release_body(app, "s", "v1", {"commit": "abc"})
        self.assertNotIn("envs", body)
        self.assertEqual(body["disks"], [{"name": "data", "mountTo": "/data"}])
        self.assertEqual(body["healthCheck"]["interval"], 30)
        self.assertEqual(body["gitInfo"], {"commit": "abc"})


class TestDeploy(Base):
    def test_happy_path(self):
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_OK, out)
        rel = self.posts("/v2/projects/shop/releases")[0][2]
        self.assertEqual(rel["sourceID"], "src1")
        self.assertEqual(rel["message"], "v9")
        self.assertEqual(len(rel["gitInfo"]["commit"]), 40)
        self.assertNotIn("envs", rel)
        self.assertFalse(self.posts("/v1/releases/"))  # no rollback
        self.assertIn("deployed", out)

    def test_dry_run_makes_no_writes(self):
        code, out = self.deploy("--dry-run")
        self.assertEqual(code, 0, out)
        self.assertEqual([c for c in self.fake.calls if c[0] != "GET"], [])
        self.assertIn("dry-run", out)

    def test_smoke_failure_rolls_back(self):
        self.fake.healthy_after_deploy = False
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_ROLLED_BACK, out)
        self.assertEqual(self.posts("/v1/releases/r1/rollback")[0][1], "/v1/releases/r1/rollback")
        self.assertIn("rolled back", out)

    def test_unhealthy_release_that_took_traffic_rolls_back(self):
        self.fake.next_state = "UNHEALTHY"
        self.fake.healthy_after_deploy = False
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_ROLLED_BACK, out)
        self.assertTrue(self.posts("/v1/releases/r1/rollback"))

    def test_build_failure_keeps_old_release_and_redacts_logs(self):
        self.fake.next_state = "FAILED"
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_FAILED, out)
        self.assertFalse(self.posts("/v1/releases/"))
        self.assertNotIn(FAKE_JWT[:20], out)
        self.assertIn("old release still serving", out)

    def test_rollback_refused_is_exit_3(self):
        self.fake.healthy_after_deploy = False
        self.fake.rollback_status = 428
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_ROLLBACK_FAILED, out)

    def test_missing_env_refuses_before_upload(self):
        self.fake.projects["shop"]["envs"] = []
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_REFUSED, out)
        self.assertFalse(self.posts("/v2/"))
        self.assertIn("API_KEY", out)

    def test_stopped_app_refused(self):
        self.fake.projects["shop"]["scale"] = 0
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_REFUSED, out)

    def test_daily_quota_refused(self):
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date().isoformat()
        self.fake.releases += [{"_id": f"q{i}", "type": "DEPLOY", "state": "READY", "createdAt": today + "T01:00:00Z"} for i in range(20)]
        code, out = self.deploy()
        self.assertEqual(code, liara_deploy.EXIT_REFUSED, out)
        self.assertIn("quota", out)

    def test_summary_written(self):
        summary = self.repo / "summary.md"
        code, _ = self.deploy("--summary", str(summary))
        self.assertEqual(code, 0)
        text = summary.read_text()
        self.assertIn("| smoke tests | pass |", text)
        self.assertIn("release_id: `r2`", text)


class TestApply(Base):
    def manifest_with(self, **over):
        m = json.loads(Path(self.manifest).read_text())
        m["apps"][0].update(over)
        p = self.repo / "infra" / "m2.json"
        p.write_text(json.dumps(m))
        return str(p)

    def run_cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = liara_apply.main(list(args))
        return code, out.getvalue()

    def test_no_changes(self):
        code, out = self.run_cli("plan", "-f", self.manifest, "--no-color")
        self.assertEqual(code, 0, out)
        self.assertIn("No changes", out)
        self.assertEqual([c for c in self.fake.calls if c[0] != "GET"], [])

    def test_plan_shows_changes_and_is_read_only(self):
        f = self.manifest_with(plan="medium-g2", disks=[{"name": "data", "sizeGB": 3, "mountTo": "/data"},
                                                        {"name": "logs", "sizeGB": 1, "mountTo": "/logs"}])
        code, out = self.run_cli("plan", "-f", f, "--no-color")
        self.assertEqual(code, 2, out)
        self.assertIn('~ plan                     = "small-g2" -> "medium-g2"', out)
        self.assertIn("+ disk.logs", out)
        self.assertIn('~ disk.data.size', out)
        self.assertIn("Plan: 1 to add, 2 to change, 0 to destroy.", out)
        self.assertEqual([c for c in self.fake.calls if c[0] != "GET"], [])
        self.assertNotIn("s3cr3t-value", out)

    def test_invalid_capability_is_error(self):
        f = self.manifest_with(zeroDowntime=True)
        code, out = self.run_cli("plan", "-f", f, "--no-color")
        self.assertEqual(code, 1, out)
        self.assertIn("zero-downtime", out)

    def test_apply_executes_and_converges(self):
        f = self.manifest_with(plan="medium-g2")
        code, out = self.run_cli("apply", "-f", f, "--auto-approve", "--no-color")
        self.assertEqual(code, 0, out)
        self.assertEqual(self.posts("/v1/projects/shop/resize")[0][2], {"planID": "medium-g2"})

    def test_apply_never_deletes_without_flag(self):
        self.fake.disks["p1"].append({"name": "old", "size": 1})
        code, out = self.run_cli("apply", "-f", self.manifest, "--auto-approve", "--no-color")
        self.assertEqual(code, 0, out)
        self.assertIn("not declared (kept", out)
        self.assertFalse([c for c in self.fake.calls if c[0] == "DELETE"])
        code, out = self.run_cli("apply", "-f", self.manifest, "--auto-approve", "--allow-delete", "--no-color")
        self.assertEqual([c[1] for c in self.fake.calls if c[0] == "DELETE"], ["/v1/projects/p1/disks/old"])

    def test_missing_env_value_comes_from_secret_store_only(self):
        self.fake.projects["shop"]["envs"] = []
        code, out = self.run_cli("apply", "-f", self.manifest, "--auto-approve", "--no-color")
        self.assertEqual(code, 1, out)
        self.assertIn("LIARA_ENV__API_KEY", out)
        os.environ["LIARA_ENV__API_KEY"] = "from-vault"
        try:
            code, out = self.run_cli("apply", "-f", self.manifest, "--auto-approve", "--no-color")
        finally:
            del os.environ["LIARA_ENV__API_KEY"]
        self.assertEqual(code, 0, out)
        self.assertEqual(self.posts("/v1/projects/update-envs")[-1][2]["variables"], [{"key": "API_KEY", "value": "from-vault"}])
        self.assertNotIn("from-vault", out)


class TestRedact(unittest.TestCase):
    def test_redacts_token_and_jwt(self):
        os.environ["LIARA_API_TOKEN"] = "supersecrettokenvalue"  # noqa: S105 - test fixture
        try:
            s = redact("Authorization: Bearer supersecrettokenvalue and " + FAKE_JWT)
        finally:
            del os.environ["LIARA_API_TOKEN"]
        self.assertNotIn("supersecrettokenvalue", s)
        self.assertNotIn(FAKE_JWT[:20], s)


if __name__ == "__main__":
    unittest.main()

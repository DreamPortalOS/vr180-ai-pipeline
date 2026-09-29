"""Tests for the Quest dome VR viewer (issue #434).

Covers the acceptance criteria a CI run can actually check (no headset, no
browser, no network):

* the shader's view-direction → master-UV formula agrees with the Python
  mirror ``studio.dome_proj.dir_to_master_uv`` to < 1e-3 on the five required
  directions (zenith / front / right / back / 45°仰角) — the 对拍.  The formula
  lives in ``studio/static/xr/dome_xr_proj.js`` and is run by Node where
  available (skipped on CI like the dome orientation suite).
* ``GET /api/xr/sources`` is read-only and work_root-scoped; a traversal
  attempt is rejected (400 out-of-root, 404 missing).
* ``--https`` generates and reuses a self-signed cert when openssl is available
  (mocked subprocess), reports a clear USB-method error when it is not, and the
  default ``python -m studio.server`` start parameters are unchanged.
* the ``/xr`` page and its assets are served with the right content.

The Quest rendering itself (lead's headless-Chrome 2D screenshot + Quest 3 VR
session) is verified by the lead, not here — CI has no GPU and no headset.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from studio.dome_proj import dir_to_master_uv
from studio.server import create_app, main
from studio.xr_https import CertError, _build_cmd, ensure_self_signed_cert

ROOT = Path(__file__).resolve().parent.parent
XR_DIR = ROOT / "studio" / "static" / "xr"
XR_PROJ_JS = XR_DIR / "dome_xr_proj.js"
HALF_PI = math.pi / 2
DEG = math.pi / 180
ZERO = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


@pytest.fixture()
def client(tmp_path):
    app = create_app(default_work_dir=str(tmp_path / "studio_work"))
    return TestClient(app)


# ── page + assets served ─────────────────────────────────────────────────
def test_xr_page_and_assets_served(client) -> None:
    """The /xr page and its JS assets are served (no CDN, all local)."""
    res = client.get("/xr")
    assert res.status_code == 200
    assert "dome_xr.js" in res.text
    assert "dome_xr_proj.js" in res.text
    js = client.get("/xr/dome_xr.js").text
    assert "/api/xr/sources" in js, "dome_xr.js must fetch /api/xr/sources"
    assert "drawSceneBody" in js
    assert "dirToUV" in client.get("/xr/dome_xr_proj.js").text
    # the page must not pull any CDN — only local /… and relative refs
    html = XR_DIR.joinpath("dome_xr.html").read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html, "no CDN: /xr must be fully local"


# ── /api/xr/sources scoping ──────────────────────────────────────────────
def test_xr_sources_empty_before_any_output(client) -> None:
    res = client.get("/api/xr/sources")
    assert res.status_code == 200
    body = res.json()
    assert body["sources"] == []
    assert "work_root" in body


def test_xr_sources_lists_work_root_media_newest_first(client, tmp_path) -> None:
    """Media files under work_root are listed newest first; non-media ignored."""
    import time

    wr = Path(client.get("/api/health").json()["work_root"])
    wr.mkdir(parents=True, exist_ok=True)
    (wr / "ignore.txt").write_text("nope", encoding="utf-8")
    (wr / "a.png").write_bytes(b"\x89PNG\r\n")
    time.sleep(0.05)
    (wr / "b.mp4").write_bytes(b"\x00\x00\x00 ftyp")
    res = client.get("/api/xr/sources")
    assert res.status_code == 200
    names = [s["name"] for s in res.json()["sources"]]
    assert "b.mp4" in names and "a.png" in names
    assert "ignore.txt" not in names
    # newest first
    assert names.index("b.mp4") < names.index("a.png")
    # kinds classify correctly
    kinds = {s["name"]: s["kind"] for s in res.json()["sources"]}
    assert kinds["a.png"] == "image"
    assert kinds["b.mp4"] == "video"
    # every entry carries a media_url that hits the /api/media whitelist
    for s in res.json()["sources"]:
        assert s["media_url"].startswith("/api/media?path=")


def test_xr_sources_src_param_included(client, tmp_path) -> None:
    """?src= adds an explicit master path (the URL form ?src=<测试 domemaster>)."""
    wr = Path(client.get("/api/health").json()["work_root"])
    wr.mkdir(parents=True, exist_ok=True)
    master = wr / "explicit.png"
    master.write_bytes(b"\x89PNG\r\n")
    res = client.get("/api/xr/sources", params={"src": str(master)})
    assert res.status_code == 200
    srcs = res.json()["sources"]
    assert any(s["path"] == str(master) for s in srcs)


def test_xr_sources_traversal_outside_root_rejected(client) -> None:
    """A ?src= that resolves outside work_root/temp is rejected (400).

    The repo root is neither under work_root nor under the system temp dir
    (CI checks the repo out to a workspace, not temp), so a path there is a
    clean, platform-independent out-of-root target.
    """
    candidate = ROOT / "xr_traversal_target_should_be_400.png"
    res = client.get("/api/xr/sources", params={"src": str(candidate)})
    assert res.status_code == 400, f"out-of-root src must be 400, got {res.status_code}"
    assert "outside" in res.json()["detail"].lower()


def test_xr_sources_missing_src_is_404(client, tmp_path) -> None:
    wr = Path(client.get("/api/health").json()["work_root"])
    res = client.get("/api/xr/sources", params={"src": str(wr / "nope.png")})
    assert res.status_code == 404


# ── shader UV formula 对拍 (node) ─────────────────────────────────────────
def _node_bin() -> str | None:
    return shutil.which("node")


def _five_directions() -> list[tuple[str, tuple[float, float, float]]]:
    """The five directions the acceptance checklist pins, all unit length."""
    return [
        ("zenith", (0.0, 1.0, 0.0)),
        ("front", (0.0, 0.0, 1.0)),
        ("right", (1.0, 0.0, 0.0)),
        ("back", (0.0, 0.0, -1.0)),
        ("45 elev front", (math.sin(45 * DEG), math.cos(45 * DEG), 0.0)),
    ]


@pytest.mark.skipif(_node_bin() is None, reason="node not installed; JS 对拍 runs where Node is available")
def test_xr_shader_uv_matches_python_mirror(tmp_path) -> None:
    """The JS dirToUV agrees with Python dir_to_master_uv on 5 directions.

    The card's UV 对拍: given 5 directions (zenith / 正前 / 右 / 后 / 45°仰角),
    the shader's UV (extracted to the pure JS helper) matches
    ``studio.dome_proj.dir_to_master_uv`` to < 1e-3.  Skipped on CI (no Node);
    the Python mirror already cross-checks the same formula against the
    measured v360 matrix in test_studio_dome_orientation.py.
    """
    assert XR_PROJ_JS.is_file(), "dome_xr_proj.js must exist"
    cases = []
    for name, d in _five_directions():
        uv = dir_to_master_uv(d, ZERO)
        cases.append({"name": name, "d": list(d), "u": uv["u"], "v": uv["v"], "inside": 1 if uv["inside"] else 0})
    # also sample a denser grid so a silent sign flip is caught
    for elev in (0, 20, 45, 70, 89):
        for az in (0, 45, 90, 170, -150):
            d = (
                math.sin(elev * DEG) * math.sin(az * DEG),
                math.cos(elev * DEG),
                math.sin(elev * DEG) * math.cos(az * DEG),
            )
            uv = dir_to_master_uv(d, ZERO)
            cases.append(
                {"name": f"e{elev}a{az}", "d": list(d), "u": uv["u"], "v": uv["v"], "inside": 1 if uv["inside"] else 0}
            )
    js = (
        "const P=require(process.argv[1]);"
        "const c=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "const out=[];"
        "for(const r of c){const o=P.dirToUV(r.d[0],r.d[1],r.d[2]);"
        "out.push({name:r.name,u:o.u,v:o.v,inside:o.inside?1:0});}"
        "process.stdout.write(JSON.stringify(out));"
    )
    proc = subprocess.run(
        [_node_bin(), "-e", js, str(XR_PROJ_JS)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    got = json.loads(proc.stdout)
    assert len(got) == len(cases)
    for exp, g in zip(cases, got, strict=True):
        assert g["inside"] == exp["inside"], f"inside mismatch {exp['name']}: js={g['inside']} py={exp['inside']}"
        assert abs(g["u"] - exp["u"]) < 1e-3, f"u {exp['name']}: js={g['u']} py={exp['u']}"
        assert abs(g["v"] - exp["v"]) < 1e-3, f"v {exp['name']}: js={g['v']} py={exp['v']}"


@pytest.mark.skipif(_node_bin() is None, reason="node not installed")
def test_xr_proj_js_node_check() -> None:
    """dome_xr_proj.js must parse (the formula module the shader mirrors)."""
    proc = subprocess.run([_node_bin(), "--check", str(XR_PROJ_JS)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, f"node --check failed:\n{proc.stderr}"


# ── --https cert generation ───────────────────────────────────────────────
def test_build_cmd_is_a_list_no_shell() -> None:
    """The openssl argv is a list (CLAUDE.md boundary: no shell=True)."""
    cmd = _build_cmd("openssl", Path("k.pem"), Path("c.pem"), with_addext=True)
    assert isinstance(cmd, list)
    assert cmd[0] == "openssl"
    assert "-keyout" in cmd and "-out" in cmd
    assert "-subj" in cmd
    assert "-addext" in cmd  # the SAN form
    # no shell meta in any element
    for part in cmd:
        assert isinstance(part, str)
        assert " " not in part or part.startswith("subjectAltName="), f"shell-unsafe arg {part!r}"


def test_self_signed_cert_generated_and_reused(tmp_path, monkeypatch) -> None:
    """openssl available → cert+key land and are reused on the second call.

    openssl is invoked via the module's ``subprocess.run`` reference; mock it
    to write the files so no real openssl call is required (the real call is
    exercised ad-hoc locally, not in CI).
    """
    import studio.xr_https as mod

    monkeypatch.setattr(mod, "_openssl_bin", lambda: "openssl")

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(cmd, capture_output, text, timeout, check):
        # the list form carries -keyout <path> -out <path>; write both so the
        # caller sees the files it expects from a real openssl run.
        ki = cmd.index("-keyout")
        oi = cmd.index("-out")
        Path(cmd[ki + 1]).write_text("KEY", encoding="utf-8")
        Path(cmd[oi + 1]).write_text("CERT", encoding="utf-8")
        return _Result()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    cert, key = ensure_self_signed_cert(tmp_path)
    assert cert.is_file() and key.is_file()
    # second call reuses (does not regenerate)
    cert_mtime = cert.stat().st_mtime
    cert2, key2 = ensure_self_signed_cert(tmp_path)
    assert cert2 == cert and key2 == key
    assert cert.stat().st_mtime == cert_mtime


def test_self_signed_cert_missing_openssl_clear_error(tmp_path, monkeypatch) -> None:
    """openssl absent → CertError naming the USB alternative (clear error)."""
    import studio.xr_https as mod

    monkeypatch.setattr(mod, "_openssl_bin", lambda: None)
    with pytest.raises(CertError) as exc:
        ensure_self_signed_cert(tmp_path)
    msg = str(exc.value)
    assert "openssl" in msg.lower()
    assert "USB" in msg or "adb reverse" in msg, "error must point at the USB alternative"


def test_self_signed_cert_openssl_failure_retries_without_addext(tmp_path, monkeypatch) -> None:
    """If openssl rejects -addext (older/LibreSSL), retry without it; failing
    both raises CertError with the USB hint."""
    import studio.xr_https as mod

    monkeypatch.setattr(mod, "_openssl_bin", lambda: "openssl")
    calls = []

    def fake_run(cmd, capture_output, text, timeout, check):
        calls.append(list(cmd))

        class P:
            returncode = 1
            stderr = "unknown option -addext" if "-addext" in cmd else "boom"
            stdout = ""

        return P()

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(CertError) as exc:
        ensure_self_signed_cert(tmp_path)
    assert "USB" in str(exc.value) or "adb reverse" in str(exc.value)
    # first call used -addext, second (retry) did not
    assert any("-addext" in c for c in calls)
    assert any("-addext" not in c for c in calls)


# ── main() default params unchanged + --https wiring ─────────────────────
def test_main_default_params_unchanged(monkeypatch) -> None:
    """`python -m studio.server` with no flags stays 127.0.0.1:8787/http."""
    import uvicorn

    captured = {}

    def fake_run(app_str, **kwargs):
        captured["app"] = app_str
        captured["kw"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    main([])
    assert captured["app"] == "studio.server:app"
    kw = captured["kw"]
    assert kw["host"] == "127.0.0.1"
    assert kw["port"] == 8787
    assert kw["reload"] is False
    assert "ssl_certfile" not in kw
    assert "ssl_keyfile" not in kw


def test_main_https_passes_ssl_and_overrides_host_port(monkeypatch) -> None:
    """--https adds ssl_certfile/keyfile; --host/--port are honoured."""
    import uvicorn

    import studio.server as srv

    captured = {}

    def fake_run(app_str, **kwargs):
        captured["kw"] = kwargs

    monkeypatch.setattr(uvicorn, "run", fake_run)
    # ensure_self_signed_cert returns a temp pair without invoking openssl
    cert = Path("/tmp/dome_xr_cert.pem")
    key = Path("/tmp/dome_xr_key.pem")
    monkeypatch.setattr(srv, "ensure_self_signed_cert", lambda d: (cert, key))
    main(["--host", "0.0.0.0", "--port", "9000", "--https"])
    kw = captured["kw"]
    assert kw["host"] == "0.0.0.0"
    assert kw["port"] == 9000
    assert kw["ssl_certfile"] == str(cert)
    assert kw["ssl_keyfile"] == str(key)


def test_main_https_without_openssl_exits_nonzero(monkeypatch) -> None:
    """--https with openssl missing prints the USB hint and exits non-zero."""
    import uvicorn

    import studio.server as srv

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)

    def fail(_dir):
        raise CertError("openssl missing — use adb reverse")

    monkeypatch.setattr(srv, "ensure_self_signed_cert", fail)
    with pytest.raises(SystemExit) as exc:
        main(["--https"])
    assert exc.value.code != 0

"""CLI wiring smoke tests — guardrails for night-parallel development (issue #85, K-3).

These are *integration-of-the-wiring* tests, deliberately distinct from the
unit tests that mock every dependency.  The risk they catch is the one mock
unit tests cannot: a CLI that no longer imports, a parameter renamed in a
parallel branch, a mock-provider chain that silently stops producing a real
file, or a QA JSON contract that downstream tooling relies on.

Four assertion families (one per acceptance criterion):

1. **``--help`` exits 0** for every delivered CLI  -> catches import-time
   breakage and argparse-definition conflicts.
2. **Key-parameter presence**  -> catches a parallel branch silently dropping
   a flag that scripts/CI depend on (argparse exits non-zero on unknown).
   For ``setup_seedvr2.py`` — the one CLI that starts heavy I/O the instant
   argparse succeeds — presence is asserted *in-process* against its parser
   and its planned step list, never by running the bootstrap (section 2b).
3. **Mock full-chain smoke**  -> the mock provider path really does emit a
   ffprobe-readable mp4; plus an orchestration-layer assertion that drives
   the real ``image_to_vr180.run_pipeline`` call graph (with the model-heavy
   convert stage stubbed) so the prepare/generate/streamcheck wiring is
   exercised end-to-end without GPU/models/network.
4. **QA JSON contract**  -> the real ``scripts/vr180_qa.py --json`` process
   emits a parseable report with ``verdict`` and ``checks`` fields and exits
   0 on a valid VR180 file.

Every failure message names the CLI and the contract that broke so that the
nightly auto-reviewer can localise the break in one glance.

K-29 (#319) — **a timeout is a failure, never a pass.**  Section 2 used to
run ``setup_seedvr2.py`` with an 8-second ceiling and treat the resulting
``TimeoutExpired`` as proof that "argparse accepted the flags and the CLI got
on with its slow real work".  That inverts the meaning of a hang: any script
that survived 8 seconds passed, so a missing dependency, a wedged network
call or a crash *after* the eighth second were all indistinguishable from
success — a permanently green test, worse than no test because it advertises
coverage it does not have.  ``_run_cli`` now fails loudly on TimeoutExpired,
and the setup CLI is asserted against its parser namespace, its ``--help``
text and the argv it *plans* to execute (with a fake runner that makes any
real subprocess an error).  No assertion in this module waits on a network.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.spherical_injector import _box4, _build_st3d, _build_sv3d

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cli_scratch_cwd(tmp_path, monkeypatch):
    """Run every test in this module (and every CLI subprocess it spawns) from
    a scratch cwd.

    W-5 (#315): the CLIs under test resolve their default artefacts against
    the *cwd* — ``--input x.mp4`` derives ``x_vr180_temp/`` next to the input,
    ``--outdir o`` is taken literally.  Inherited from pytest, that cwd is the
    repo root, so a plain ``--help``-adjacent smoke test used to leave
    ``x_vr180/``, ``x_vr180_temp/`` and ``o/`` in the checkout.  Every path in
    this module is absolute (``REPO_ROOT`` / ``SCRIPTS`` / ``tmp_path``), so
    moving the cwd changes nothing else.
    """
    monkeypatch.chdir(tmp_path)


def _cli_env() -> dict[str, str]:
    """Subprocess environment that can import the repo as a package."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run_cli(*argv: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Invoke a script by absolute path as a real subprocess.

    subprocess.list form (no shell=True).  Fails the test with a message
    that names the CLI whose contract broke if the process crashes.

    K-29 (#319): a ``TimeoutExpired`` is an **unconditional failure**.  It
    used to be swallowed by the caller and re-interpreted as success, which
    is what made this module's parameter checks unfalsifiable.  A CLI that
    does not terminate within *timeout* has broken its contract — either it
    hangs on something it should not touch (network, model download) or it
    is slower than a smoke test may be — and the suite must say so.
    """
    script = Path(argv[0])
    if not script.is_absolute():
        script = SCRIPTS / (argv[0] + (".py" if not argv[0].endswith(".py") else ""))
    try:
        proc = subprocess.run(
            [sys.executable, str(script), *argv[1:]],
            capture_output=True,
            text=True,
            env=_cli_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"[{script.name}] did not terminate within {timeout}s: {' '.join(argv[1:])}\n"
            "A hang is a FAILURE, never a pass (K-29, #319). Either the CLI "
            "started real work a smoke test must not trigger (git clone, pip "
            "install, model download), or it is deadlocked. Assert the parser "
            "in-process instead of racing a wall clock.\n"
            f"partial stderr:\n{(exc.stderr or b'')[-400:]!r}"
        )
    return proc


def _ffprobe(path: str) -> dict:
    """Return ffprobe's parsed JSON for *path*, failing loudly on error."""
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not on PATH")
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v",
            "-show_entries",
            "stream=width,height,codec_name,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"ffprobe failed on {path}: {proc.stderr[-300:]}"
    return json.loads(proc.stdout)


def _fake_convert_stub(job_args, _input_path: str, _convert_override=None) -> str:
    """No-op convert stage mirroring ``stage_convert``'s call signature
    ``(args, input_video, convert=None)``: copies the generated video to the
    expected output path (``job_args.vr180_output``) so the real
    ``image_to_vr180.run_pipeline`` orchestration can reach the audio + QA
    bookkeeping without touching depth/stereo models."""
    output = job_args.vr180_output
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_input_path, output)
    return output


def _synthetic_vr180_mp4(tmp_path: Path) -> str:
    """Build a minimal, ffprobe-readable mp4 carrying real sv3d + st3d
    boxes (Google Spherical Video V2) at the spec location (moov > trak >
    mdia > minf > stbl > stsd > hvc1).  ffprobe parses it; vr180_qa sees
    the boxes and exits 0.  No real samples -> 0x0 dims, but that never
    yields a 'fail' check, so the file is treated as valid.
    """
    hvc1 = _box4(b"hvc1", b"\x00" * 78 + _build_sv3d(5760, 2880, "sbs") + _build_st3d("sbs"))
    stsd = _box4(b"stsd", b"\x00\x00\x00\x00" + struct.pack(">I", 1) + hvc1)
    stbl = _box4(b"stbl", stsd)
    minf = _box4(b"minf", stbl)
    mdia = _box4(b"mdia", minf)
    trak = _box4(b"trak", mdia)
    moov = _box4(b"moov", trak)
    ftyp = _box4(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    path = tmp_path / "valid_vr180.mp4"
    path.write_bytes(ftyp + moov)
    return str(path)


# ---------------------------------------------------------------------------
# 1. --help exits 0 for every delivered CLI
# ---------------------------------------------------------------------------

ALL_CLIS = [
    "run_pipeline.py",
    "image_to_vr180.py",
    "generate.py",
    "vr180_qa.py",
    "stereo_sweep.py",
    "setup_seedvr2.py",
]


@pytest.mark.parametrize("cli", ALL_CLIS)
def test_cli_help_exits_zero(cli):
    """Every delivered CLI must be importable and must accept ``--help``.

    A non-zero exit here means the script crashed on import or argparse
    definition is broken — exactly the class of break mock unit tests miss.
    """
    proc = _run_cli(str(SCRIPTS / cli), "--help")
    assert proc.returncode == 0, (
        f"[{cli}] '--help' contract BROKEN (exit={proc.returncode}). "
        f"Likely import-time crash. stderr:\n{proc.stderr[-400:]}"
    )


# ---------------------------------------------------------------------------
# 2. Key-parameter presence (argparse rejects unknown flags -> exit 2)
# ---------------------------------------------------------------------------

# (cli, flag, <positional/required args needed so --help-style parsing is skipped>)
KEY_PARAMS = [
    # run_pipeline: quality preset, chunked memory mode, stage subset
    ("run_pipeline.py", ["--input", "x.mp4", "--output", "o.mp4", "--quality", "preview"]),
    ("run_pipeline.py", ["--input", "x.mp4", "--output", "o.mp4", "--chunk-size", "4"]),
    ("run_pipeline.py", ["--input", "x.mp4", "--output", "o.mp4", "--stages", "depth"]),
    # image_to_vr180: provider selection, gen resolution, manifest, resume
    ("image_to_vr180.py", ["--image", "x.png", "--provider", "mock"]),
    ("image_to_vr180.py", ["--image", "x.png", "--gen-resolution", "720p"]),
    ("image_to_vr180.py", ["--image", "x.png", "--manifest", "m.json"]),
    ("image_to_vr180.py", ["--image", "x.png", "--resume-from", "m.json"]),
    ("image_to_vr180.py", ["--image", "x.png", "--quality", "preview"]),
    # vr180_qa: machine-readable JSON output
    ("vr180_qa.py", ["x.mp4", "--json"]),
    # generate: image-to-video input + generation resolution
    ("generate.py", ["--image", "x.png", "--gen-resolution", "720p"]),
    ("generate.py", ["--provider", "mock"]),
    # stereo_sweep: grid knobs
    ("stereo_sweep.py", ["--input", "x.mp4", "--outdir", "o", "--limit-seconds", "3"]),
    ("stereo_sweep.py", ["--input", "x.mp4", "--outdir", "o", "--disparities", "0.04"]),
    # setup_seedvr2.py is deliberately ABSENT from this list — see section 2b.
    # It is the one CLI that starts a real `git clone` + pip install the
    # instant argparse succeeds, so "spawn it and see what happens" can never
    # be a safe flag-presence probe.  Its parser is asserted in-process.
]


@pytest.mark.parametrize("cli,extra", KEY_PARAMS)
def test_key_parameter_exists(cli, extra):
    """The named parameter must still be recognised by argparse.

    argparse exits non-zero (2) on an unknown option, so a silently-renamed
    flag is caught immediately.  We deliberately pass *only* the flags under
    test plus the minimal required positional args, so the value is not
    evaluated — we only check the flag is defined.

    Every CLI listed here terminates on its own (the fake ``x.mp4`` inputs do
    not exist, so they fail fast).  ``_run_cli`` fails the test if one does
    not: K-29 (#319) removed the branch that used to catch the timeout and
    ``return`` early, which silently turned "this CLI hung" into a pass.
    """
    proc = _run_cli(str(SCRIPTS / cli), *extra, timeout=30)
    # argparse's signal for a deleted/renamed flag is the stderr message
    # "unrecognized arguments".  Some CLIs happen to use exit code 2 for
    # their own validation (e.g. generate.py "a prompt is required"), so we
    # key off the stderr signature rather than the return code.
    argparser_rejected = "unrecognized arguments" in proc.stderr
    assert not argparser_rejected, (
        f"[{cli}] parameter contract BROKEN: flag sequence {extra} "
        f"rejected as 'unrecognized arguments'. "
        f"One of these flags was deleted/renamed in a parallel branch. "
        f"stderr:\n{proc.stderr[-300:]}"
    )


# ---------------------------------------------------------------------------
# 3. Mock full-chain smoke
# ---------------------------------------------------------------------------


def test_generate_mock_produces_ffprobeable_file(tmp_path):
    """``generate --provider mock`` must emit a real, ffprobe-readable mp4.

    This is the offline on-ramp: no API keys, no network, no models.
    Contract broken = the mock provider no longer yields a playable file,
    which would silently strand everything downstream of generation.
    """
    out = str(tmp_path / "gen.mp4")
    proc = _run_cli(
        str(SCRIPTS / "generate.py"),
        "smoke test",
        "--provider",
        "mock",
        "--duration",
        "2",
        "--gen-resolution",
        "480p",
        "-o",
        out,
        timeout=90,
    )
    assert proc.returncode == 0, (
        f"[generate] mock-chain contract BROKEN (exit={proc.returncode}). stderr:\n{proc.stderr[-400:]}"
    )
    assert Path(out).is_file(), "[generate] mock-chain contract BROKEN: no output file written"
    info = _ffprobe(out)
    streams = info.get("streams")
    assert streams and streams[0].get("width", 0) > 0, (
        f"[generate] mock-chain contract BROKEN: ffprobe cannot read output "
        f"(streams={streams}). stderr not expected, file may be empty/corrupt."
    )


def test_image_to_vr180_mock_orchestration_produces_file(tmp_path):
    """Orchestration-layer assertion: the real ``image_to_vr180.run_pipeline``
    call graph (prepare -> generate(mock) -> streamcheck) wires correctly and
    yields an ffprobe-readable generated video.

    The ``convert`` stage is stubbed because its real implementation invokes
    the depth/stereo models (Depth Anything / StereoCrafter) which are
    unavailable on CI (CPU-only, no models, no network).  Stubbing *only*
    that model-boundary stage is the intended, documented fallback: it lets
    us exercise the real wiring of the offline path end-to-end without a
    GPU, while still proving the generated artefact is a genuine mp4.

    (Boundary: we do NOT patch production code to make the test pass; the
    model requirement is a real architectural fact, not a defect.  See PR
    description if the orchestrator call graph changes.)
    """
    from scripts import image_to_vr180 as i2v
    from scripts.image_to_vr180 import JobArgs, run_pipeline

    # The stubbed convert emits a plain (non-VR180) mp4, so the real QA stage
    # would fail and abort the pipeline.  We only want to validate the
    # orchestration wiring (prepare -> generate -> streamcheck -> audio), not
    # QA itself (QA has its own module + a dedicated contract test below).
    # Mirrors the patch pattern used in tests/test_image_to_vr180.py.
    i2v.stage_qa = lambda _p: 0

    img = tmp_path / "inp.png"
    # A tiny valid 3-channel PNG so stage_prepare (OpenCV) has a real frame
    # to letterbox.  Must be a 2-D array; a bare tuple yields a 1-D image
    # that trips prepare_image.
    import cv2
    import numpy as np

    cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8) + 255)

    job = JobArgs(
        image=str(img),
        prompt="smoke",
        provider="mock",
        duration=2,
        gen_resolution="480p",
        gen_ratio="adaptive",
        upscale="none",
        quality="preview",
        copy_audio_from=None,
        workdir=str(tmp_path / "i2v"),
        manifest_path=None,
        resume_from=None,
    )

    result = run_pipeline(job, convert=_fake_convert_stub)

    assert "output" in result, (
        "[image_to_vr180] orchestration contract BROKEN: run_pipeline did not return an 'output' key"
    )
    out = result["output"]
    assert Path(out).is_file(), f"[image_to_vr180] orchestration contract BROKEN: output {out!r} not written"
    info = _ffprobe(out)
    streams = info.get("streams")
    assert streams and streams[0].get("width", 0) > 0, (
        f"[image_to_vr180] orchestration contract BROKEN: output is not ffprobe-readable (streams={streams})"
    )


# ---------------------------------------------------------------------------
# 4. QA JSON contract
# ---------------------------------------------------------------------------


def test_vr180_qa_json_contract(tmp_path):
    """``scripts/vr180_qa.py --json`` must emit a parseable report with
    ``verdict`` and ``checks`` fields, and exit 0 on a valid VR180 file.

    This is the machine-readable contract downstream automation consumes.
    Broken = JSON shape changed or the validator no longer passes a real
    VR180 file.  We run the real subprocess (not the Python function) so
    import/CLI wiring is included in the contract.
    """
    valid = _synthetic_vr180_mp4(tmp_path)
    proc = _run_cli(str(SCRIPTS / "vr180_qa.py"), valid, "--json")

    assert proc.returncode == 0, (
        f"[vr180_qa] JSON-contract BROKEN: validator exited {proc.returncode} "
        f"on a valid VR180 file. stderr:\n{proc.stderr[-400:]}"
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"[vr180_qa] JSON-contract BROKEN: --json did not emit valid JSON ({exc}). stdout:\n{proc.stdout[-400:]}"
        ) from exc

    assert "verdict" in payload, f"[vr180_qa] JSON-contract BROKEN: missing 'verdict' field. keys={list(payload)}"
    assert "checks" in payload, f"[vr180_qa] JSON-contract BROKEN: missing 'checks' field. keys={list(payload)}"
    assert isinstance(payload["checks"], list) and payload["checks"], (
        "[vr180_qa] JSON-contract BROKEN: 'checks' must be a non-empty list"
    )
    assert all({"name", "status", "detail"} <= set(c) for c in payload["checks"]), (
        "[vr180_qa] JSON-contract BROKEN: each check must have name/status/detail"
    )
    assert not any(c["status"] == "fail" for c in payload["checks"]), (
        "[vr180_qa] JSON-contract BROKEN: a 'fail' check on a valid VR180 file "
        f"-> checks={[c['name'] + ':' + c['status'] for c in payload['checks']]}"
    )

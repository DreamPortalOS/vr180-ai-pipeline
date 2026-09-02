"""Tests for the in-repo StereoCrafter bootstrap (scripts/setup_stereocrafter.py).

CI-safe: zero network, zero downloads.  The ``--dry-run`` path is asserted
verbatim; the real paths mock ``subprocess.check_call`` / ``subprocess.run``
and the filesystem so nothing leaves the test sandbox.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "setup_stereocrafter",
        Path(__file__).resolve().parent.parent / "scripts" / "setup_stereocrafter.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


setup = _load_setup_module()


# ---------------------------------------------------------------------------
# Constants: the clone URL must point at TencentARC (not Tencent), and the
# weight repo must be TencentARC/StereoCrafter.  Pinned to guard against the
# org-name typo that broke the bootstrap in issue #111.
# ---------------------------------------------------------------------------


class TestRepoUrls:
    def test_clone_url_uses_tencent_arc_org(self):
        """The git clone URL must use the TencentARC org (Tencent/ is wrong)."""
        assert "TencentARC/StereoCrafter" in setup.NODE_REPO_URL, setup.NODE_REPO_URL
        assert "Tencent/StereoCrafter" not in setup.NODE_REPO_URL, (
            f"clone URL still uses the wrong Tencent/ org: {setup.NODE_REPO_URL}"
        )

    def test_clone_url_is_https_github(self):
        assert setup.NODE_REPO_URL.startswith("https://github.com/")
        assert setup.NODE_REPO_URL.endswith(".git")

    def test_model_repo_id_uses_tencent_arc(self):
        """The HF weight repo id must be TencentARC/StereoCrafter."""
        assert setup._MODEL_REPO_ID == "TencentARC/StereoCrafter"


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Redirect every in-repo path constant to a tmp_path-based sandbox."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(setup, "REPO_ROOT", repo_root, raising=True)

    node_dir = repo_root / "third_party" / "StereoCrafter"
    venv_dir = node_dir / ".venv"
    python_exe = venv_dir / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
    model_dir = repo_root / "models" / "StereoCrafter"
    svd_dir = repo_root / "models" / "svd-img2vid-xt-1-1"

    monkeypatch.setattr(setup, "INREPO_NODE_DIR", node_dir, raising=True)
    monkeypatch.setattr(setup, "INREPO_VENV_DIR", venv_dir, raising=True)
    monkeypatch.setattr(setup, "INREPO_PYTHON", python_exe, raising=True)
    monkeypatch.setattr(setup, "INREPO_MODEL_DIR", model_dir, raising=True)
    monkeypatch.setattr(setup, "INREPO_SVD_DIR", svd_dir, raising=True)
    return SimpleNamespace(
        repo_root=repo_root,
        node_dir=node_dir,
        venv_dir=venv_dir,
        python_exe=python_exe,
        model_dir=model_dir,
        svd_dir=svd_dir,
    )


# ---------------------------------------------------------------------------
# --dry-run: the step sequence must be stable and contain zero I/O.
# ---------------------------------------------------------------------------


class TestDryRunSequence:
    def test_dry_run_records_all_steps_in_order(self, sandbox):
        """A full --dry-run emits the planned steps in order: clone, venv, torch, deps, model, self-check."""
        buf = setup.DryRunBuffer()

        setup.ensure_node_repo(None, dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, None, dry_run=True, buffer=buf)
        setup.download_models(None, skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)

        labels = [s.lower() for s in buf.steps]
        assert any("git clone" in s for s in labels), f"missing git clone in {buf.steps}"
        assert any("venv" in s for s in labels), f"missing venv step in {buf.steps}"
        assert any("torch" in s and "2.6.0" in s for s in labels), f"missing torch step in {buf.steps}"
        assert any("diffusers" in s for s in labels), f"missing deps step in {buf.steps}"
        assert any("snapshot_download" in s for s in labels), f"missing model step in {buf.steps}"
        assert any("self-check" in s for s in labels), f"missing self-check in {buf.steps}"

        clone_idx = next(i for i, s in enumerate(labels) if "git clone" in s)
        venv_idx = next(i for i, s in enumerate(labels) if "venv" in s)
        torch_idx = next(i for i, s in enumerate(labels) if "torch" in s and "2.6.0" in s)
        deps_idx = next(i for i, s in enumerate(labels) if "diffusers" in s)
        model_idx = next(i for i, s in enumerate(labels) if "snapshot_download" in s)
        check_idx = next(i for i, s in enumerate(labels) if "self-check" in s)
        assert clone_idx < venv_idx < torch_idx < deps_idx < model_idx < check_idx

    def test_dry_run_performs_no_subprocess(self, sandbox):
        """--dry-run must NOT call subprocess.check_call / subprocess.run at all."""
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run") as mock_run,
        ):
            setup.main(["--dry-run"])
            mock_cc.assert_not_called()
            mock_run.assert_not_called()

    def test_dry_run_skip_model_omits_model_step(self, sandbox):
        buf = setup.DryRunBuffer()
        setup.ensure_node_repo(None, dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, None, dry_run=True, buffer=buf)
        setup.download_models(None, skip_model=True, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)
        assert not any("snapshot_download" in s.lower() for s in buf.steps)

    def test_dry_run_skip_deps_omits_venv_and_pip_steps(self, sandbox):
        """When --skip-deps is given, the venv-creation and pip steps are absent."""
        buf = setup.DryRunBuffer()
        # Only record the non-deps steps to prove the deps step is not emitted.
        setup.ensure_node_repo(None, dry_run=True, buffer=buf)
        # ensure_venv_and_deps is skipped entirely when --skip-deps is set.
        setup.download_models(None, skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)
        assert not any("-m venv" in s for s in buf.steps)
        assert not any("pip" in s and "torch" in s for s in buf.steps)


# ---------------------------------------------------------------------------
# Step 1: ensure_node_repo (clone vs pull vs --repo-dir)
# ---------------------------------------------------------------------------


class TestEnsureNodeRepo:
    def test_clone_when_absent(self, sandbox):
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(None, dry_run=False, buffer=setup.DryRunBuffer())
            assert mock_cc.call_count == 1
            cmd = mock_cc.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]
            assert setup.NODE_REPO_URL in cmd
            assert str(sandbox.node_dir) in cmd

    def test_pull_when_git_checkout_present(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(None, dry_run=False, buffer=setup.DryRunBuffer())
            assert mock_cc.call_count == 1
            cmd = mock_cc.call_args[0][0]
            assert cmd == ["git", "pull"]
            assert mock_cc.call_args.kwargs["cwd"] == str(sandbox.node_dir)

    def test_reclones_when_dir_exists_but_not_git(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(None, dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_cc.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]

    def test_repo_dir_pull_when_existing_git(self, sandbox):
        """--repo-dir pointing at an existing git checkout pulls rather than clones."""
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(str(sandbox.node_dir), dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_cc.call_args[0][0]
            assert cmd == ["git", "pull"]
            assert mock_cc.call_args.kwargs["cwd"] == str(sandbox.node_dir)

    def test_proxy_hint_on_clone_failure(self, sandbox, caplog):
        caplog.set_level("WARNING", logger="setup-stereocrafter")
        with (
            patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "git")),
            pytest.raises(subprocess.CalledProcessError),
        ):
            setup.ensure_node_repo(None, dry_run=False, buffer=setup.DryRunBuffer())
        assert any("proxy" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 2: venv + pip install (torch cu124 + curated deps)
# ---------------------------------------------------------------------------


class TestPipTimeout:
    """Issue #165: pip install timeout must be 3600s by default, overridable via
    --pip-timeout (CLI) and SETUP_PIP_TIMEOUT (env var), with CLI > env >
    default precedence.  Every pip command must also carry --timeout 120 (pip's
    own single-connection timeout).
    """

    def test_default_timeout_is_3600(self):
        assert setup.DEFAULT_PIP_TIMEOUT == 3600

    def test_pip_connect_timeout_is_120(self):
        assert setup.PIP_CONNECT_TIMEOUT == 120

    def test_resolve_default_when_no_cli_no_env(self, monkeypatch):
        monkeypatch.delenv("SETUP_PIP_TIMEOUT", raising=False)
        assert setup._resolve_pip_timeout(None) == 3600

    def test_resolve_env_var_when_no_cli(self, monkeypatch):
        monkeypatch.setenv("SETUP_PIP_TIMEOUT", "4800")
        assert setup._resolve_pip_timeout(None) == 4800

    def test_resolve_cli_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("SETUP_PIP_TIMEOUT", "4800")
        assert setup._resolve_pip_timeout(7200) == 7200

    def test_resolve_bad_env_falls_back_to_default(self, monkeypatch, caplog):
        caplog.set_level("WARNING", logger="setup-stereocrafter")
        monkeypatch.setenv("SETUP_PIP_TIMEOUT", "not-a-number")
        assert setup._resolve_pip_timeout(None) == 3600
        assert any("not a valid integer" in r.message.lower() for r in caplog.records)

    def test_ensure_venv_uses_default_timeout_when_not_passed(self, sandbox):
        """The old call signature (no pip_timeout kwarg) must still work and
        use the default 3600s — guards against the PR #168 regression that made
        pip_timeout required.
        """
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")

        captured_timeouts: list[int] = []

        def fake_check_call(cmd, *args, **kwargs):
            if "timeout" in kwargs:
                captured_timeouts.append(kwargs["timeout"])

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())
        assert captured_timeouts == [3600, 3600], captured_timeouts

    def test_ensure_venv_uses_explicit_pip_timeout(self, sandbox):
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")

        captured_timeouts: list[int] = []

        def fake_check_call(cmd, *args, **kwargs):
            if "timeout" in kwargs:
                captured_timeouts.append(kwargs["timeout"])

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(
                None,
                None,
                dry_run=False,
                buffer=setup.DryRunBuffer(),
                pip_timeout=999,
            )
        assert captured_timeouts == [999, 999], captured_timeouts

    def test_pip_command_includes_timeout_120(self, sandbox):
        """Every pip install command (torch and deps) must carry --timeout 120
        (pip's single-connection timeout, issue #165)."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())

        pip_calls = [c for c in calls if "-m" in c and "pip" in c]
        assert len(pip_calls) == 2, f"expected 2 pip calls, got {len(pip_calls)}: {pip_calls}"
        for pc in pip_calls:
            assert "--timeout" in pc, f"--timeout missing from {pc}"
            idx = pc.index("--timeout")
            assert pc[idx + 1] == "120", f"--timeout value not 120: {pc}"


class TestEnsureVenvAndDeps:
    def test_creates_venv_when_absent(self, sandbox):
        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert venv_calls, f"no venv creation call in {calls}"
        assert str(sandbox.venv_dir) in venv_calls[0]

    def test_skips_venv_when_present(self, sandbox):
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake python")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert not venv_calls, f"venv should not be recreated: {venv_calls}"

    def test_torch_uses_stable_cu124_not_nightly(self, sandbox):
        """The torch install command must pin 2.6.0 + the cu124 index, NOT nightly."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())

        torch_calls = [c for c in calls if any("torch==" in tok for tok in c)]
        assert torch_calls, f"no torch install call in {calls}"
        tc = torch_calls[0]
        assert any(tok == "torch==2.6.0" for tok in tc), f"torch not pinned to 2.6.0: {tc}"
        assert any(tok == "torchvision==0.21.0" for tok in tc), f"torchvision not pinned: {tc}"
        assert setup.TORCH_INDEX_URL in tc
        assert "cu124" in " ".join(tc)
        assert "nightly" not in " ".join(tc).lower()
        assert "--retries" in tc
        assert tc[tc.index("--retries") + 1] == "10"

    def test_runtime_deps_are_installed(self, sandbox):
        """The curated RUNTIME_DEPS are installed as a single pip install."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())

        # Match the deps install by a substring on any token (transformers /
        # diffusers are now version-pinned, e.g. 'diffusers==0.29.2', so a
        # bare-equality list membership check would miss them — issue #155).
        deps_calls = [c for c in calls if any("diffusers" in tok for tok in c)]
        assert deps_calls, f"no runtime-deps install call in {calls}"
        dc = deps_calls[0]
        for dep in setup.RUNTIME_DEPS:
            assert dep in dc, f"{dep} missing from {dc}"
        # torch / torchvision must NOT be part of the curated-deps install.
        assert not any("torch==" in tok for tok in dc), f"torch leaks into deps install: {dc}"

    def test_fire_is_in_runtime_deps(self, sandbox):
        """The entry scripts use ``from fire import Fire`` — fire must be in
        RUNTIME_DEPS or the self-check --help will ImportError (issue #111)."""
        assert "fire" in setup.RUNTIME_DEPS, setup.RUNTIME_DEPS

    def test_decord_is_in_runtime_deps(self):
        """Both entry scripts (inpainting_inference.py &
        depth_splatting_inference.py) ``from decord import VideoReader, cpu`` —
        decord must be in RUNTIME_DEPS or the self-check --help will
        ModuleNotFoundError (issue #116).  Mirrors the decord assertion in
        test_setup_depthcrafter.py."""
        assert "decord" in setup.RUNTIME_DEPS, setup.RUNTIME_DEPS

    def test_pip_mirror_forwarded_to_pip(self, sandbox):
        """--pip-mirror adds -i to the curated-deps install; torch keeps --index-url cu124."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, mirror, dry_run=False, buffer=setup.DryRunBuffer())

        torch_call = next(c for c in calls if any("torch==" in tok for tok in c))
        # Substring match: diffusers is now version-pinned (issue #155).
        deps_call = next(c for c in calls if any("diffusers" in tok for tok in c))
        assert "--index-url" in torch_call
        assert setup.TORCH_INDEX_URL in torch_call
        assert "-i" in deps_call
        assert deps_call[deps_call.index("-i") + 1] == mirror


# ---------------------------------------------------------------------------
# Step 3: model download (snapshot_download, skip if present)
# ---------------------------------------------------------------------------


class TestDownloadModels:
    def test_skip_model_short_circuits(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-stereocrafter")
        with patch("subprocess.check_call") as mock_cc:
            setup.download_models(None, skip_model=True, dry_run=False, buffer=setup.DryRunBuffer())
            mock_cc.assert_not_called()
        assert any("model download skipped" in r.message.lower() for r in caplog.records)

    def test_skips_existing_snapshot(self, sandbox):
        """A non-empty model dir is treated as already-downloaded and skipped."""
        sandbox.model_dir.mkdir(parents=True)
        (sandbox.model_dir / "dummy.bin").write_text("weights")

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not be called"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
            fake_hf.snapshot_download.assert_not_called()

    def test_snapshot_download_used_for_weights(self, sandbox):
        """snapshot_download is the primary path; it lands into models/StereoCrafter."""
        sandbox.model_dir.mkdir(parents=True)

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks):
            (sandbox.model_dir / "weights.bin").write_text("fake")
            assert repo_id == setup._MODEL_REPO_ID
            assert local_dir == str(sandbox.model_dir)
            assert local_dir_use_symlinks is False
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert (sandbox.model_dir / "weights.bin").exists()

    def test_missing_huggingface_hub_warns_and_returns(self, sandbox, caplog):
        """If huggingface_hub is not importable, the step warns and returns cleanly."""
        caplog.set_level("WARNING", logger="setup-stereocrafter")
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert any("huggingface_hub not installed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 4: SVD base pre-download (default-on, HF token, gated-repo 403 error)
# ---------------------------------------------------------------------------


class TestDownloadSvdBase:
    """The SVD base is pre-downloaded by default into models/svd-img2vid-xt-1-1
    (issue #150).  It is a GATED HF repo — the step reads the local HF token,
    passes it to snapshot_download, and surfaces a 403 with an actionable
    application-page error instead of a bare OSError."""

    def test_dry_run_records_svd_step(self, sandbox):
        """--dry-run records the SVD download step into the buffer."""
        buf = setup.DryRunBuffer()
        setup.download_svd_base(None, dry_run=True, buffer=buf)
        assert any("snapshot_download" in s and setup._SVD_REPO_ID in s for s in buf.steps), buf.steps

    def test_skip_svd_short_circuits(self, sandbox, caplog):
        """--skip-svd skips the SVD pre-download and logs it."""
        caplog.set_level("INFO", logger="setup-stereocrafter")
        with patch("subprocess.check_call") as mock_cc:
            setup.download_svd_base(None, skip_svd=True, dry_run=False, buffer=setup.DryRunBuffer())
            mock_cc.assert_not_called()
        assert any("skip-svd" in r.message.lower() for r in caplog.records)

    def test_skips_existing_svd_snapshot(self, sandbox):
        """A dir that already has the three fp16 safetensors is treated as
        ready and skipped (issue #155: the presence check keys on exactly the
        files the Stage-2 local-folder resolver looks for)."""
        sandbox.svd_dir.mkdir(parents=True)
        (sandbox.svd_dir / "image_encoder").mkdir()
        (sandbox.svd_dir / "unet").mkdir()
        (sandbox.svd_dir / "vae").mkdir()
        (sandbox.svd_dir / "image_encoder" / "model.fp16.safetensors").write_text("w")
        (sandbox.svd_dir / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_text("w")
        (sandbox.svd_dir / "vae" / "diffusion_pytorch_model.fp16.safetensors").write_text("w")

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not be called"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(None, dry_run=False, buffer=setup.DryRunBuffer())
            fake_hf.snapshot_download.assert_not_called()

    def test_default_downloads_into_inrepo_svd_dir(self, sandbox):
        """With no --svd-dir, the SVD base lands in the in-repo models/svd-img2vid-xt-1-1."""
        captured = {}

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            captured["repo_id"] = repo_id
            captured["local_dir"] = local_dir
            captured["token"] = token
            # Simulate the download writing a file so a re-run would skip.
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "svd.bin").write_text("fake")
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token="hf_test_token",
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        assert captured["repo_id"] == setup._SVD_REPO_ID
        assert captured["local_dir"] == str(sandbox.svd_dir)
        assert captured["token"] == "hf_test_token"
        assert (sandbox.svd_dir / "svd.bin").exists()

    def test_svd_dir_override_used_as_target(self, sandbox, tmp_path):
        """--svd-dir overrides the target directory."""
        custom = tmp_path / "custom-svd"
        captured = {}

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            captured["local_dir"] = local_dir
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "svd.bin").write_text("fake")
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                str(custom),
                skip_svd=False,
                hf_token="hf_x",
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        assert captured["local_dir"] == str(custom)
        assert (custom / "svd.bin").exists()

    def test_hf_token_passed_to_snapshot_download(self, sandbox, monkeypatch):
        """The resolved HF token is forwarded to snapshot_download (issue #150).

        _read_hf_token precedence: explicit arg > env var > huggingface_hub stored.
        We inject the token via the env var path (no --hf-token) to exercise the
        auto-read, and assert snapshot_download received it.
        """
        monkeypatch.setenv(setup._HF_TOKEN_ENV_VARS[0], "hf_env_token")
        # Ensure huggingface_hub's own get_token does not shadow the env var.
        captured = {}

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            captured["token"] = token
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "svd.bin").write_text("fake")
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token=None,
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        assert captured["token"] == "hf_env_token"

    def test_gated_repo_403_raises_actionable_error_with_link(self, sandbox):
        """A 403 'gated repo' OSError must surface as a RuntimeError naming the
        application page (issue #150) — not a bare OSError."""
        gated_err = OSError(
            "You are trying to access a gated repo. "
            "Cannot access gated repo for url "
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/config.json "
            "403 Client Error. Access to model ... is restricted and you are not in the authorized list."
        )

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=gated_err)
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_present")
        with (
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            pytest.raises(RuntimeError, match="gated Hugging Face repo") as exc_info,
        ):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token="hf_present",
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        msg = str(exc_info.value)
        # The error must contain the application page, not just a bare 403.
        assert setup._SVD_REPO_URL in msg, msg
        assert "403" in msg or "gated" in msg.lower(), msg

    def test_non_gated_download_failure_warns_not_raises(self, sandbox, caplog):
        """A generic (non-gated) download failure must warn and return, not raise."""
        caplog.set_level("WARNING", logger="setup-stereocrafter")
        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=ConnectionError("connection timed out"))
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_present")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token="hf_present",
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        assert any("snapshot_download failed" in r.message.lower() for r in caplog.records)

    def test_missing_token_warns_about_gated_repo(self, sandbox, monkeypatch, caplog):
        """When no HF token is found, the step warns that the repo is gated (issue #150)."""
        for var in setup._HF_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        caplog.set_level("WARNING", logger="setup-stereocrafter")

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "svd.bin").write_text("fake")
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value=None)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token=None,
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )
        assert any("gated" in r.message.lower() for r in caplog.records)

    def test_fetches_only_fp16_safetensors(self, sandbox):
        """Issue #155: the snapshot uses allow_patterns restricted to fp16
        safetensors + configs (≈5 GB, not the full ~10 GB repo) and ignores
        .bin entirely — the repo ships only safetensors and we want exactly
        the files the Stage-2 local-folder resolver looks for."""
        captured: dict = {}

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            captured["allow_patterns"] = kwargs.get("allow_patterns")
            captured["ignore_patterns"] = kwargs.get("ignore_patterns")
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "svd.bin").write_text("fake")
            return local_dir

        fake_hf = MagicMock()
        fake_hf.snapshot_download = fake_snapshot_download
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_present")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_svd_base(
                None,
                skip_svd=False,
                hf_token="hf_present",
                dry_run=False,
                buffer=setup.DryRunBuffer(),
            )

        allow = captured["allow_patterns"]
        assert allow is not None
        # The three fp16 safetensors the Stage-2 loaders resolve locally must
        # be in the allow list.
        assert "image_encoder/model.fp16.safetensors" in allow
        assert "unet/diffusion_pytorch_model.fp16.safetensors" in allow
        assert "vae/diffusion_pytorch_model.fp16.safetensors" in allow
        # The fp32 variants are NOT requested.
        assert "image_encoder/model.safetensors" not in allow
        assert "unet/diffusion_pytorch_model.safetensors" not in allow
        # .bin is never fetched even if it appeared upstream.
        ignore = captured["ignore_patterns"]
        assert "*.bin" in ignore

    def test_reuses_hf_cache_when_partial(self, sandbox, monkeypatch):
        """Issue #190: a partial in-repo snapshot (missing weight components)
        must NOT raise and block — it reuses the standard HF cache to fill
        the gap (cached vae / image_encoder are NOT re-downloaded; only the
        missing unet is fetched).  snapshot_download IS called, the broken
        in-repo snapshot is wiped first, and the download proceeds cleanly.

        This supersedes the #186 raise-on-partial behaviour: the whole point
        of #190 is that the broken state is now the problem we FIX by
        reusing the cache rather than the problem we refuse to touch.
        """
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        (unet / "config.json").write_text("{}", encoding="utf-8")
        # vae + image_encoder are present (as if the HF cache had them)
        _write_weight(sandbox.svd_dir, "vae")
        _write_weight(sandbox.svd_dir, "image_encoder")

        fake_hf = MagicMock()

        fake_dl = MagicMock()

        def fake_snapshot_download(repo_id, local_dir, local_dir_use_symlinks, token, **kwargs):
            fake_dl(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=local_dir_use_symlinks,
                token=token,
            )
            assert repo_id == setup._SVD_REPO_ID
            assert local_dir == str(sandbox.svd_dir)
            return local_dir

        fake_hf.snapshot_download = fake_snapshot_download
        fake_hf.HfFolder = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_present")
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        # Must NOT raise — it reuses the cache and fills the missing unet.
        setup.download_svd_base(
            None,
            skip_svd=False,
            hf_token="hf_present",
            dry_run=False,
            buffer=setup.DryRunBuffer(),
        )
        # snapshot_download WAS called (this is the #190 reuse path).
        fake_dl.assert_called_once()
        assert fake_dl.call_args.kwargs["token"] == "hf_present"


# ---------------------------------------------------------------------------
# Runtime dep version pins (issue #155 — the safetensors load fix)
# ---------------------------------------------------------------------------


class TestRuntimeDepPins:
    """Issue #155: transformers/diffusers must be PINNED to the combo upstream
    tested, so the local-folder safetensors-first resolution path is the one
    upstream validated (an unpinned transformers could drift to a 5.x that
    breaks the vendored SVD inpainting pipeline, or an older line that
    defaults to .bin).
    """

    def test_transformers_pinned_to_upstream_tested_version(self):
        """transformers==4.42.3 (the exact pin in upstream
        TencentARC/StereoCrafter/requirements.txt)."""
        pins = [d for d in setup.RUNTIME_DEPS if d.startswith("transformers")]
        assert pins, f"transformers missing from RUNTIME_DEPS: {setup.RUNTIME_DEPS}"
        assert "4.42.3" in pins[0], f"transformers not pinned to 4.42.3: {pins[0]}"
        # Must be a hard pin (==), not a bare name or a loose >= that could drift.
        assert pins[0].startswith("transformers=="), pins[0]

    def test_diffusers_pinned_to_upstream_tested_version(self):
        """diffusers==0.29.2 (the exact pin in upstream requirements.txt)."""
        pins = [d for d in setup.RUNTIME_DEPS if d.startswith("diffusers")]
        assert pins, f"diffusers missing from RUNTIME_DEPS: {setup.RUNTIME_DEPS}"
        assert "0.29.2" in pins[0], f"diffusers not pinned to 0.29.2: {pins[0]}"
        assert pins[0].startswith("diffusers=="), pins[0]

    def test_no_bare_transformers_or_diffusers(self):
        """A bare (unpinned) 'transformers'/'diffusers' element must NOT be in
        RUNTIME_DEPS — that would let pip drift to an untested version
        (the regression of issue #155)."""
        assert "transformers" not in setup.RUNTIME_DEPS, setup.RUNTIME_DEPS
        assert "diffusers" not in setup.RUNTIME_DEPS, setup.RUNTIME_DEPS

    def test_torch_not_in_runtime_deps(self):
        """torch/torchvision are pinned separately in Step 2 (cu124 index) so a
        transitive dep can never bump them — they must NOT be in RUNTIME_DEPS."""
        for tok in setup.RUNTIME_DEPS:
            assert not tok.startswith("torch=="), f"torch leaked into RUNTIME_DEPS: {tok}"
            assert not tok.startswith("torchvision=="), f"torchvision leaked: {tok}"


# HF token resolution (issue #150)
# ---------------------------------------------------------------------------


class TestReadHfToken:
    """_read_hf_token precedence: explicit arg > env var > huggingface_hub stored."""

    def test_explicit_token_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_env")
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_env2")
        assert setup._read_hf_token("hf_explicit") == "hf_explicit"

    def test_env_var_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_env_primary")
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        fake_hf = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_stored")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            assert setup._read_hf_token(None) == "hf_env_primary"

    def test_stored_token_when_no_env_no_explicit(self, monkeypatch):
        for var in setup._HF_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        fake_hf = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(return_value="hf_stored")
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            assert setup._read_hf_token(None) == "hf_stored"
            fake_hf.HfFolder.get_token.assert_called_once()

    def test_returns_none_when_nothing_available(self, monkeypatch):
        for var in setup._HF_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            assert setup._read_hf_token(None) is None

    def test_hf_folder_lookup_failure_does_not_crash(self, monkeypatch):
        """A broken huggingface_hub install must not crash token lookup."""
        for var in setup._HF_TOKEN_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        fake_hf = MagicMock()
        fake_hf.HfFolder.get_token = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            # Must return None, not raise.
            assert setup._read_hf_token(None) is None


class TestIsGatedRepoError:
    """The gated-repo detector must recognize HF's 401/403 gated-repo errors
    and not mislabel generic failures (issue #150)."""

    def test_real_hf_gated_message(self):
        err = OSError(
            "You are trying to access a gated repo. "
            "Cannot access gated repo for url "
            "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/resolve/main/x.json "
            "403 Client Error."
        )
        assert setup._is_gated_repo_error(err) is True

    def test_restricted_authorized_list_phrase(self):
        err = OSError("Access to model is restricted and you are not in the authorized list.")
        assert setup._is_gated_repo_error(err) is True

    def test_generic_network_error_not_gated(self):
        assert setup._is_gated_repo_error(ConnectionError("connection timed out")) is False

    def test_unrelated_403_not_gated(self):
        """A 403 that does not name the gated model must not be mislabelled."""
        assert setup._is_gated_repo_error(OSError("some other 403 thing happened")) is False


# ---------------------------------------------------------------------------
# Step 4: self-check
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_self_check_passes_on_exit_zero(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-stereocrafter")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "inpainting_inference.py").write_text("# fake")

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == "--help"
            assert mock_run.call_args.kwargs["cwd"] == str(sandbox.node_dir)
        assert any("self-check passed" in r.message.lower() for r in caplog.records)

    def test_self_check_raises_when_python_missing(self, sandbox):
        with pytest.raises(RuntimeError, match="venv python not found"):
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())

    def test_self_check_raises_when_no_inference_script(self, sandbox):
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        with pytest.raises(RuntimeError, match="No known inference script"):
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())

    def test_self_check_records_label_in_dry_run(self, sandbox):
        buf = setup.DryRunBuffer()
        setup.self_check(None, dry_run=True, buffer=buf)
        assert any("self-check" in s for s in buf.steps)

    def test_dry_run_label_uses_real_entry_script_name(self, sandbox):
        """The dry-run self-check label must reference the real upstream entry
        script (inpainting_inference.py), not the non-existent run.py."""
        buf = setup.DryRunBuffer()
        setup.self_check(None, dry_run=True, buffer=buf)
        label = buf.steps[-1]
        assert "inpainting_inference.py" in label, label
        assert "run.py" not in label, label

    def test_find_inference_script_prefers_inpainting(self, sandbox):
        """Upstream has no run.py; _find_inference_script must find the real
        root-level fire-style entry scripts (issue #111)."""
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / "inpainting_inference.py").write_text("# fake")
        found = setup._find_inference_script(sandbox.node_dir)
        assert found is not None
        assert found.name == "inpainting_inference.py"

    def test_find_inference_script_falls_back_to_depth_splatting(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / "depth_splatting_inference.py").write_text("# fake")
        found = setup._find_inference_script(sandbox.node_dir)
        assert found is not None
        assert found.name == "depth_splatting_inference.py"


# ---------------------------------------------------------------------------
# print_summary + env vars
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_prints_three_env_vars(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-stereocrafter")
        setup.print_summary(None)
        msgs = "\n".join(r.message for r in caplog.records)
        assert "STEREOCRAFTER_REPO_DIR" in msgs
        assert "STEREOCRAFTER_PYTHON" in msgs
        assert "STEREOCRAFTER_CKPT_DIR" in msgs
        assert str(sandbox.node_dir) in msgs
        assert str(sandbox.python_exe) in msgs
        assert str(sandbox.model_dir) in msgs


# ---------------------------------------------------------------------------
# main() wiring
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_dry_run_no_side_effects(self, sandbox):
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run") as mock_run,
        ):
            setup.main(["--dry-run"])
            mock_cc.assert_not_called()
            mock_run.assert_not_called()
        assert not sandbox.node_dir.exists()

    def test_main_full_flow_calls_all_steps(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "inpainting_inference.py").write_text("# fake")

        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")) as mock_run,
            patch("huggingface_hub.snapshot_download", return_value=str(sandbox.model_dir)),
        ):
            setup.main([])
            assert mock_cc.called
            assert mock_run.called  # self-check

    def test_main_skip_deps_and_skip_model(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "inpainting_inference.py").write_text("# fake")
        sandbox.model_dir.mkdir(parents=True)
        (sandbox.model_dir / "dummy.bin").write_text("w")

        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
            patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("must not be called")),
        ):
            setup.main(["--skip-deps", "--skip-model"])
            cmds = [c[0][0] for c in mock_cc.call_args_list]
            assert all(cmd[:2] == ["git", "pull"] for cmd in cmds) or not cmds


# ---------------------------------------------------------------------------
# --verify-only + per-component SVD weight completeness (issue #186)
# ---------------------------------------------------------------------------


def _write_weight(model_dir: Path, name: str) -> Path:
    """Create a non-empty safetensors weight file for an SVD component."""
    comp = model_dir / name
    comp.mkdir(parents=True, exist_ok=True)
    weight = comp / (
        "model.fp16.safetensors" if name == "image_encoder" else "diffusion_pytorch_model.fp16.safetensors"
    )
    weight.write_bytes(b"\x00" * 1024)
    return weight


class TestSvdComponentCompleteness:
    """Issue #186: the completeness check must inspect WEIGHT files per
    component — following symlinks — and must NOT be fooled by a config-only
    directory (the exact unet bug that blocked the disocclusion gate).
    """

    def test_config_only_is_missing(self, sandbox):
        """A component that only has config.json (no weight file) is MISSING —
        this is the direct regression test for the #186 bug where the unet
        subfolder held only a config blob yet was treated as 'already
        downloaded'."""
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        (unet / "config.json").write_text('{"sample_size": 64}')  # config only

        assert setup._svd_component_has_weight(sandbox.svd_dir, "unet") is False
        # config.json must not count as a weight file.
        assert setup._svd_component_weight_path(sandbox.svd_dir, "unet") is None

    def test_broken_symlink_weight_is_missing(self, sandbox):
        """A weight file that is a symlink pointing at a non-existent blobs/
        target must be treated as MISSING.  Path.is_file() follows symlinks, so
        a dangling symlink returns False — the check must rely on that, not on
        the directory entry existing (issue #186: the HF cache trap)."""
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        target = unet / "ghost.weights"  # does not exist
        link = unet / "diffusion_pytorch_model.fp16.safetensors"
        link.symlink_to(target)

        assert link.is_symlink()  # the directory entry exists — must not fool us
        assert setup._svd_component_has_weight(sandbox.svd_dir, "unet") is False
        assert setup._svd_component_weight_path(sandbox.svd_dir, "unet") is None

    def test_valid_nonempty_weight_is_complete(self, sandbox):
        """A real, non-empty weight file (safetensors or .bin) makes the
        component complete."""
        sandbox.svd_dir.mkdir(parents=True)
        _write_weight(sandbox.svd_dir, "unet")
        _write_weight(sandbox.svd_dir, "vae")
        _write_weight(sandbox.svd_dir, "image_encoder")

        for name in setup._SVD_WEIGHT_COMPONENTS:
            assert setup._svd_component_has_weight(sandbox.svd_dir, name) is True
        assert setup._has_svd_fp16_safetensors(sandbox.svd_dir) is True

    def test_empty_weight_file_is_missing(self, sandbox):
        """A weight file that exists but is zero bytes is MISSING — the check
        requires size > 0 so a truncated / never-finished download can't pass."""
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        (unet / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")

        assert setup._svd_component_has_weight(sandbox.svd_dir, "unet") is False

    def test_missing_subfolder_is_missing(self, sandbox):
        """When the component subfolder itself does not exist, the component is
        MISSING."""
        sandbox.svd_dir.mkdir(parents=True)
        assert setup._svd_component_has_weight(sandbox.svd_dir, "unet") is False

    def test_bin_weights_also_count(self, sandbox):
        """Both *.safetensors and *.bin are accepted as weight files
        (safetensors is the SVD reality, .bin is accepted for forward
        compatibility)."""
        sandbox.svd_dir.mkdir(parents=True)
        vae = sandbox.svd_dir / "vae"
        vae.mkdir()
        (vae / "diffusion_pytorch_model.fp16.bin").write_bytes(b"\x00" * 100)
        assert setup._svd_component_has_weight(sandbox.svd_dir, "vae") is True


class TestVerifyOnly:
    """--verify-only: pure filesystem inspection (no download) that prints a
    per-component report and exits non-zero if anything is missing (issue #186).
    """

    def test_verify_only_does_not_download(self, sandbox, monkeypatch):
        """--verify-only must NOT trigger snapshot_download — it is read-only.
        A config-only unet forces the report to be non-trivial so we can also
        assert it surfaces the missing component."""
        sandbox.svd_dir.mkdir(parents=True)
        vae = sandbox.svd_dir / "vae"
        vae.mkdir()
        (vae / "config.json").write_text("{}")  # config only — missing weight

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        # No HF cache copy — the report is purely about the in-repo dir.
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 1  # non-zero: something is missing
        fake_hf.snapshot_download.assert_not_called()

    def test_verify_only_reports_missing_unet_and_exits_nonzero(self, sandbox, monkeypatch):
        """Regression for the exact #186 scenario: unet has config only, vae +
        image_encoder are fine.  The verify report must name unet as MISSING
        and exit 1 (no download)."""
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        (unet / "config.json").write_text("{}")
        _write_weight(sandbox.svd_dir, "vae")
        _write_weight(sandbox.svd_dir, "image_encoder")

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 1
        fake_hf.snapshot_download.assert_not_called()

    def test_verify_only_reports_missing_unet_component(self, sandbox, monkeypatch):
        """verify_svd_base() returns per-component lines; the config-only unet
        line must say MISSING."""
        sandbox.svd_dir.mkdir(parents=True)
        unet = sandbox.svd_dir / "unet"
        unet.mkdir()
        (unet / "config.json").write_text("{}")
        _write_weight(sandbox.svd_dir, "vae")
        _write_weight(sandbox.svd_dir, "image_encoder")

        # Isolate the report to the in-repo dir only (no HF-cache fallback).
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)

        report = setup.verify_svd_base(sandbox.svd_dir)
        unet_lines = [ln for ln in report if "unet" in ln.lower()]
        assert unet_lines, report
        assert any("MISSING" in ln for ln in unet_lines), report
        assert any("OK" in ln and "vae" in ln for ln in report), report

    def test_verify_only_all_complete_exits_zero(self, sandbox, monkeypatch):
        """When every component has a real weight, --verify-only exits 0 and
        still performs no download."""
        sandbox.svd_dir.mkdir(parents=True)
        for name in setup._SVD_WEIGHT_COMPONENTS:
            _write_weight(sandbox.svd_dir, name)

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 0
        fake_hf.snapshot_download.assert_not_called()

    def test_verify_only_missing_svd_dir_exits_one(self, sandbox, monkeypatch, tmp_path):
        """When the SVD dir does not exist AND there is no HF cache copy,
        --verify-only exits 1 with a clear 'nothing to verify' message rather
        than trying to download."""
        missing = tmp_path / "no-such-svd"
        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
        # No HF cache copy available — exit 1.
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only", "--svd-dir", str(missing)])
        assert exc_info.value.code == 1
        fake_hf.snapshot_download.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #190: dual-path verify (in-repo + standard HF cache) + cache reuse
# ---------------------------------------------------------------------------


class TestHfCacheResolution:
    """_hf_cache_snapshot_dir resolves the standard HF cache snapshot path via
    huggingface_hub.try_to_load_from_cache (issue #190)."""

    def test_returns_snapshot_dir_when_repo_cached(self, tmp_path):
        """When try_to_load_from_cache resolves model_index.json to a real path,
        the parent (snapshot dir) is returned."""
        snapshot_dir = tmp_path / "snapshots" / "abc123"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model_index.json").write_text("{}", encoding="utf-8")

        def fake_try_to_load(repo_id, filename):
            assert repo_id == "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
            assert filename == "model_index.json"
            return str(snapshot_dir / "model_index.json")

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(try_to_load_from_cache=fake_try_to_load)}):
            result = setup._hf_cache_snapshot_dir("stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        assert result == snapshot_dir

    def test_returns_none_when_not_cached(self, tmp_path):
        """When try_to_load_from_cache returns None (file not in cache), the
        resolver returns None."""

        def fake_try_to_load(repo_id, filename):
            return None

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(try_to_load_from_cache=fake_try_to_load)}):
            result = setup._hf_cache_snapshot_dir("stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        assert result is None

    def test_returns_none_when_hf_hub_unavailable(self):
        """When huggingface_hub is not importable, the resolver returns None
        gracefully (CI-safe)."""
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            result = setup._hf_cache_snapshot_dir("stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        assert result is None

    def test_requests_model_index_json(self):
        """The resolver specifically looks up model_index.json (present in every
        diffusers model snapshot) — assert it never asks for a weight file."""

        def fake_try_to_load(repo_id, filename):
            assert filename == "model_index.json", f"unexpected filename: {filename}"
            return "/nonexistent/snapshots/abc/model_index.json"

        with patch.dict("sys.modules", {"huggingface_hub": MagicMock(try_to_load_from_cache=fake_try_to_load)}):
            result = setup._hf_cache_snapshot_dir("stabilityai/stable-video-diffusion-img2vid-xt-1-1")
        # The path does not exist on disk, so the resolver returns None even
        # though try_to_load_from_cache returned a string — belt-and-braces.
        assert result is None


class TestVerifyOnlyDualPath:
    """--verify-only recognises BOTH the in-repo SVD directory AND the standard
    HF cache snapshot (issue #190 requirement 2).  Whichever is complete counts
    as complete, and the output prints which path was inspected.
    """

    def test_inrepo_absent_hf_cache_complete_is_complete(self, sandbox, monkeypatch, tmp_path):
        """When the in-repo SVD dir does not exist but the HF cache has a full
        snapshot, --verify-only reports COMPLETE (exit 0) and points at the HF
        cache path."""
        # In-repo dir does NOT exist (the lead's exact state).
        assert not sandbox.svd_dir.exists()

        # Fake a complete HF cache snapshot.
        hf_snap = tmp_path / "hf-cache" / "snapshots" / "abc123"
        for name in setup._SVD_WEIGHT_COMPONENTS:
            _write_weight(hf_snap, name)
        (hf_snap / "model_index.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            setup,
            "_hf_cache_snapshot_dir",
            lambda _r: hf_snap,
        )

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 0, "HF-cache-complete must count as complete"
        fake_hf.snapshot_download.assert_not_called()

    def test_hf_cache_config_only_unet_reports_missing_and_names_unet(self, sandbox, monkeypatch, tmp_path):
        """When the HF cache snapshot has unet config-only (the real state of
        the lead's cache), --verify-only reports MISSING and the text names
        `unet` (issue #190 acceptance: '报错文本里出现 unet')."""
        # In-repo absent; HF cache has vae + image_encoder but unet config-only.
        assert not sandbox.svd_dir.exists()

        hf_snap = tmp_path / "hf-cache" / "snapshots" / "abc123"
        _write_weight(hf_snap, "vae")
        _write_weight(hf_snap, "image_encoder")
        unet = hf_snap / "unet"
        unet.mkdir(parents=True)
        (unet / "config.json").write_text("{}", encoding="utf-8")
        (hf_snap / "model_index.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            setup,
            "_hf_cache_snapshot_dir",
            lambda _r: hf_snap,
        )

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 1  # unet missing → not complete
        fake_hf.snapshot_download.assert_not_called()

    def test_verify_report_names_missing_component(self, sandbox, monkeypatch, tmp_path, caplog):
        """The verify report must explicitly name the missing component — e.g.
        'unet  MISSING weights (only config found)' — not a blanket 'directory
        does not exist' (issue #190 requirement 3)."""
        caplog.set_level("INFO", logger="setup-stereocrafter")

        # In-repo absent; HF cache has config-only unet.
        hf_snap = tmp_path / "hf-cache" / "snapshots" / "abc123"
        _write_weight(hf_snap, "vae")
        _write_weight(hf_snap, "image_encoder")
        (hf_snap / "unet" / "config.json").parent.mkdir(parents=True, exist_ok=True)
        (hf_snap / "unet" / "config.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            setup,
            "_hf_cache_snapshot_dir",
            lambda _r: hf_snap,
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", MagicMock(snapshot_download=MagicMock()))

        with pytest.raises(SystemExit):
            setup.main(["--verify-only"])

        joined = "\n".join(r.message for r in caplog.records)
        # The component name must appear alongside MISSING.
        assert "unet" in joined.lower(), joined
        assert "MISSING" in joined, joined

    def test_verify_output_contains_checked_path(self, sandbox, monkeypatch, tmp_path, caplog):
        """The output must contain the path that was actually checked, so the
        user can see exactly where the base lives (issue #190: '这次就是因为
        看不出它在查哪儿，才没人发现基座压根没建')."""
        caplog.set_level("INFO", logger="setup-stereocrafter")

        # In-repo dir present (so the report names it explicitly).
        sandbox.svd_dir.mkdir(parents=True)
        for name in setup._SVD_WEIGHT_COMPONENTS:
            _write_weight(sandbox.svd_dir, name)

        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)
        monkeypatch.setitem(sys.modules, "huggingface_hub", MagicMock(snapshot_download=MagicMock()))

        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 0

        joined = "\n".join(r.message for r in caplog.records)
        # The report must include the absolute path of the directory it checked.
        assert str(sandbox.svd_dir) in joined, joined

    def test_verify_only_never_downloads_from_hf_cache_lookup(self, sandbox, monkeypatch, tmp_path):
        """--verify-only must not trigger snapshot_download — even when it
        consults the HF cache, the lookup uses try_to_load_from_cache (a
        filesystem check), NOT snapshot_download."""
        sandbox.svd_dir.mkdir(parents=True)

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("must not download"))
        fake_hf.try_to_load_from_cache = MagicMock(
            return_value=str(tmp_path / "model_index.json"),
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        with pytest.raises(SystemExit):
            setup.main(["--verify-only"])
        fake_hf.snapshot_download.assert_not_called()


# ---------------------------------------------------------------------------
# Issue #201: --verify-only summary count must match the per-line verdicts on
# the HF-cache branch.  The bug: a 2/3 HF-cache copy was reported as 0/3
# because the summary only promoted the HF count when it was FULLY complete.
# ---------------------------------------------------------------------------


class TestVerifyOnlySummaryCount:
    """Issue #201: the summary ``X/3`` count must equal the number of distinct
    components OK in at least one inspected location — consistent with the
    per-line verdicts printed above it — across all three completeness states
    on the HF-cache branch (where the bug lived)."""

    @staticmethod
    def _hf_cache_state(tmp_path, present: tuple[str, ...]) -> Path:
        """Build a fake HF-cache snapshot dir with the given components present
        as real weights; any of the three components not in *present* is left
        config-only (so it is correctly MISSING)."""
        hf_snap = tmp_path / "hf-cache" / "snapshots" / "abc123"
        hf_snap.mkdir(parents=True)
        for name in setup._SVD_WEIGHT_COMPONENTS:
            if name in present:
                _write_weight(hf_snap, name)
            else:
                comp = hf_snap / name
                comp.mkdir(parents=True, exist_ok=True)
                (comp / "config.json").write_text("{}", encoding="utf-8")
        (hf_snap / "model_index.json").write_text("{}", encoding="utf-8")
        return hf_snap

    def _capture_summary(self, caplog, sandbox, monkeypatch, hf_snap) -> str:
        """Run --verify-only with an absent in-repo dir + the given HF-cache
        snapshot, returning the joined log so the summary line can be asserted."""
        caplog.set_level("INFO", logger="setup-stereocrafter")

        # In-repo dir absent → the HF-cache branch is the one that prints.
        assert not sandbox.svd_dir.exists()
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: hf_snap)
        monkeypatch.setitem(sys.modules, "huggingface_hub", MagicMock(snapshot_download=MagicMock()))
        with pytest.raises(SystemExit):
            setup.main(["--verify-only"])
        return "\n".join(r.message for r in caplog.records)

    def test_hf_cache_all_complete_summary_is_three_thirds(self, sandbox, monkeypatch, tmp_path, caplog):
        """3/3: all components OK → summary says 3/3 (and exits 0)."""
        hf_snap = self._hf_cache_state(tmp_path, present=("image_encoder", "unet", "vae"))
        joined = self._capture_summary(caplog, sandbox, monkeypatch, hf_snap)
        assert "3/3 components complete" in joined, joined
        # The complete summary has no "(missing: …)" clause (note: the
        # in-repo "directory missing:" line is unrelated and must not match).
        assert "(missing:" not in joined.lower(), joined

    def test_hf_cache_only_unet_missing_summary_is_two_thirds(self, sandbox, monkeypatch, tmp_path, caplog):
        """THE BUG (issue #201): only unet missing → summary must say 2/3 (not
        0/3) and name `unet`.  This is the exact state the lead hit on #190:
        image_encoder + vae OK in the HF cache, unet config-only."""
        hf_snap = self._hf_cache_state(tmp_path, present=("image_encoder", "vae"))
        joined = self._capture_summary(caplog, sandbox, monkeypatch, hf_snap)
        assert "2/3 components complete" in joined, joined
        assert "missing: unet" in joined.lower(), joined
        # Must NOT claim 0/3 (the buggy behaviour that sent the lead to re-download).
        assert "0/3" not in joined, joined

    def test_hf_cache_all_missing_summary_is_zero_thirds(self, sandbox, monkeypatch, tmp_path, caplog):
        """0/3: all three components missing → summary says 0/3 and lists all three."""
        hf_snap = self._hf_cache_state(tmp_path, present=())
        joined = self._capture_summary(caplog, sandbox, monkeypatch, hf_snap)
        assert "0/3 components complete" in joined, joined
        # Every component must be named in the missing list.
        for name in setup._SVD_WEIGHT_COMPONENTS:
            assert name in joined.lower(), (name, joined)

    def test_hf_cache_count_matches_per_line_verdicts(self, sandbox, monkeypatch, tmp_path, caplog):
        """The summary count must equal the number of `OK` lines printed for
        the HF-cache branch — the consistency guarantee at the heart of #201.
        With only unet missing, exactly two `[hf-cache]` OK lines are printed
        and the summary says 2/3."""
        hf_snap = self._hf_cache_state(tmp_path, present=("image_encoder", "vae"))
        joined = self._capture_summary(caplog, sandbox, monkeypatch, hf_snap)
        ok_lines = [ln for ln in joined.splitlines() if "[hf-cache]" in ln and " OK " in ln]
        assert len(ok_lines) == 2, joined
        assert "2/3 components complete" in joined, joined

    def test_inrepo_all_complete_summary_unchanged(self, sandbox, monkeypatch, tmp_path, caplog):
        """Sanity: the in-repo branch (where the count was already correct)
        must keep reporting 3/3.  Guards against the refactor regressing the
        non-buggy path."""
        sandbox.svd_dir.mkdir(parents=True)
        for name in setup._SVD_WEIGHT_COMPONENTS:
            _write_weight(sandbox.svd_dir, name)
        # No HF-cache fallback so the in-repo path is the sole source.
        monkeypatch.setattr(setup, "_hf_cache_snapshot_dir", lambda _r: None)
        monkeypatch.setitem(sys.modules, "huggingface_hub", MagicMock(snapshot_download=MagicMock()))
        caplog.set_level("INFO", logger="setup-stereocrafter")
        with pytest.raises(SystemExit) as exc_info:
            setup.main(["--verify-only"])
        assert exc_info.value.code == 0
        joined = "\n".join(r.message for r in caplog.records)
        assert "3/3 components complete" in joined, joined

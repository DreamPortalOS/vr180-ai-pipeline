"""Tests for the in-repo DepthCrafter bootstrap (scripts/setup_depthcrafter.py).

CI-safe: zero network, zero downloads.  The ``--dry-run`` path is asserted
verbatim; the real paths mock ``subprocess.check_call`` / ``subprocess.run``
and the filesystem so nothing leaves the test sandbox.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _looks_like_path(tok: str) -> bool:
    """Heuristic: a token is a filesystem path if it contains a dir separator."""
    return any(sep in tok for sep in ("/", "\\"))


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "setup_depthcrafter",
        Path(__file__).resolve().parent.parent / "scripts" / "setup_depthcrafter.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


setup = _load_setup_module()


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Redirect every in-repo path constant to a tmp_path-based sandbox."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(setup, "REPO_ROOT", repo_root, raising=True)

    node_dir = repo_root / "third_party" / "DepthCrafter"
    venv_dir = node_dir / ".venv"
    python_exe = venv_dir / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
    model_dir = repo_root / "models" / "DepthCrafter"

    # INREPO_NODE_DIR / INREPO_VENV_DIR / INREPO_PYTHON / INREPO_MODEL_DIR are
    # derived from REPO_ROOT inside the script via module-level Path math, so we
    # patch the derived names too (they're assigned at import time).
    monkeypatch.setattr(setup, "INREPO_NODE_DIR", node_dir, raising=True)
    monkeypatch.setattr(setup, "INREPO_VENV_DIR", venv_dir, raising=True)
    monkeypatch.setattr(setup, "INREPO_PYTHON", python_exe, raising=True)
    monkeypatch.setattr(setup, "INREPO_MODEL_DIR", model_dir, raising=True)
    return SimpleNamespace(
        repo_root=repo_root,
        node_dir=node_dir,
        venv_dir=venv_dir,
        python_exe=python_exe,
        model_dir=model_dir,
    )


# ---------------------------------------------------------------------------
# --dry-run: the step sequence must be stable and contain zero I/O.
# ---------------------------------------------------------------------------


class TestDryRunSequence:
    def test_dry_run_records_all_steps_in_order(self, sandbox, caplog):
        """A full --dry-run emits the planned steps in order."""
        buf = setup.DryRunBuffer()

        setup.ensure_node_repo(None, dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, None, dry_run=True, buffer=buf)
        setup.download_models(None, skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)

        labels = [s.lower() for s in buf.steps]
        assert any("git clone" in s for s in labels), f"missing git clone in {buf.steps}"
        assert any("venv" in s for s in labels), f"missing venv step in {buf.steps}"
        assert any("torch" in s and "2.6.0" in s for s in labels), f"missing torch step in {buf.steps}"
        # Node-deps step is now a curated-list ``pip install`` (no -e, no -r).
        # It must contain the curated packages and must NOT be an editable install.
        assert any("pip install" in s and "fire" in s and "diffusers" in s for s in labels), (
            f"missing curated node-deps install step in {buf.steps}: {buf.steps}"
        )
        assert not any("pip install -e" in s for s in labels), f"must not use editable install (-e) in {buf.steps}"
        assert any("snapshot_download" in s for s in labels), f"missing model step in {buf.steps}"
        assert any("self-check" in s for s in labels), f"missing self-check in {buf.steps}"

        # Order: clone before venv before torch before node-deps before model before self-check.
        clone_idx = next(i for i, s in enumerate(labels) if "git clone" in s)
        venv_idx = next(i for i, s in enumerate(labels) if "venv" in s)
        torch_idx = next(i for i, s in enumerate(labels) if "torch" in s)
        node_dep_idx = next(i for i, s in enumerate(labels) if "pip install" in s and "fire" in s)
        model_idx = next(i for i, s in enumerate(labels) if "snapshot_download" in s)
        check_idx = next(i for i, s in enumerate(labels) if "self-check" in s)
        assert clone_idx < venv_idx < torch_idx < node_dep_idx < model_idx < check_idx

    def test_dry_run_performs_no_subprocess(self, sandbox):
        """--dry-run must NOT call subprocess.check_call / subprocess.run at all."""
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run") as mock_run,
        ):
            setup.main(["--dry-run"])
            mock_cc.assert_not_called()
            mock_run.assert_not_called()

    def test_dry_run_skips_deps_omits_venv_steps(self, sandbox):
        buf = setup.DryRunBuffer()
        setup.download_models(None, skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)
        assert not any("-m venv" in s for s in buf.steps)
        assert not any("pip" in s and "torch" in s for s in buf.steps)

    def test_dry_run_skip_model_omits_model_steps(self, sandbox):
        buf = setup.DryRunBuffer()
        setup.ensure_node_repo(None, dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, None, dry_run=True, buffer=buf)
        setup.download_models(None, skip_model=True, dry_run=True, buffer=buf)
        setup.self_check(None, dry_run=True, buffer=buf)
        assert not any("snapshot_download" in s.lower() for s in buf.steps)


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

    def test_repo_dir_uses_existing_checkout(self, sandbox, tmp_path):
        """--repo-dir pointing at an existing git checkout pulls instead of cloning."""
        external = tmp_path / "D_DepthCrafter"
        external.mkdir()
        (external / ".git").mkdir()
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(str(external), dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_cc.call_args[0][0]
            assert cmd == ["git", "pull"]
            assert mock_cc.call_args.kwargs["cwd"] == str(external)

    def test_proxy_hint_on_clone_failure(self, sandbox, caplog):
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        with (
            patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "git")),
            pytest.raises(subprocess.CalledProcessError),
        ):
            setup.ensure_node_repo(None, dry_run=False, buffer=setup.DryRunBuffer())
        assert any("proxy" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 2: venv + pip install (torch cu124 + requirements)
# ---------------------------------------------------------------------------


class TestPipTimeout:
    """Issue #166 (mirroring #165): pip install timeout must be 3600s by
    default, overridable via --pip-timeout (CLI) and SETUP_PIP_TIMEOUT (env
    var), with CLI > env > default precedence.  Every pip command must also
    carry --timeout 120 (pip's own single-connection timeout).
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
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        monkeypatch.setenv("SETUP_PIP_TIMEOUT", "not-a-number")
        assert setup._resolve_pip_timeout(None) == 3600
        assert any("not a valid integer" in r.message.lower() for r in caplog.records)

    def test_ensure_venv_uses_default_timeout_when_not_passed(self, sandbox):
        """The old call signature (no pip_timeout kwarg) must still work and
        use the default 3600s — mirrors the PR #168 regression guard on #165.
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
        (pip's single-connection timeout, issue #166 mirroring #165)."""
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

    def test_timeout_expired_prints_mirror_rerun_hint(self, sandbox, caplog):
        """A pip subprocess timeout surfaces the mirror/re-run hint (issue
        #166): the hint mentions --pip-mirror and that re-running skips done
        steps (idempotent), then exits non-zero."""
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_hf = MagicMock()
        with (
            patch(
                "subprocess.check_call",
                side_effect=subprocess.TimeoutExpired(cmd=["pip"], timeout=3600),
            ),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            pytest.raises(SystemExit) as exc_info,
        ):
            setup.main([])
        assert exc_info.value.code == 1
        msgs = "\n".join(r.message for r in caplog.records)
        assert "--pip-mirror" in msgs
        assert "重跑" in msgs
        assert "已完成步骤会跳过" in msgs


class TestEnsureVenvAndDeps:
    def _run_venv_step(self, sandbox, pip_mirror=None):
        """Set up a fake venv and capture check_call invocations for the dep step."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, pip_mirror, dry_run=False, buffer=setup.DryRunBuffer())
        return calls

    def test_creates_venv_when_absent(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, None, dry_run=False, buffer=setup.DryRunBuffer())
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert venv_calls, f"no venv creation call in {calls}"
        assert str(sandbox.venv_dir) in venv_calls[0]

    def test_skips_venv_when_present(self, sandbox):
        calls = self._run_venv_step(sandbox)
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert not venv_calls, f"venv should not be recreated: {venv_calls}"

    def test_torch_uses_stable_cu124_not_nightly(self, sandbox):
        calls = self._run_venv_step(sandbox)

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

    def test_node_deps_use_curated_list_not_editable_install(self, sandbox):
        """The node-deps step must be a curated ``pip install RUNTIME_DEPS``:
        no ``-e`` (flat-layout setuptools rejects editable), no ``-r`` (legacy
        manifest path), and it must contain fire/diffusers/transformers/etc."""
        calls = self._run_venv_step(sandbox)

        torch_call = next(c for c in calls if any("torch==" in tok for tok in c))
        node_call = next(
            (c for c in calls if c is not torch_call and "pip" in " ".join(c)),
            None,
        )
        assert node_call is not None, f"no node-deps pip install call in {calls}"

        # Must NOT be an editable or legacy-manifest install.
        assert "-e" not in node_call, f"must not use -e (editable install): {node_call}"
        assert "-r" not in node_call, f"must not use -r (requirements.txt path): {node_call}"
        # And the command body (everything after `-m pip install`) must be the
        # curated dep list + --retries + optional mirror, never a repo path.
        try:
            body_start = node_call.index("install") + 1
        except ValueError:  # pragma: no cover - defensive
            body_start = 0
        body = node_call[body_start:]
        # Any token that looks like a filesystem path is a leaked repo/manifest arg.
        assert all(not _looks_like_path(tok) for tok in body), (
            f"node-deps command body must contain only deps/flags, not a repo path: {body}"
        )

        # Must contain the curated runtime packages.
        for dep in setup.RUNTIME_DEPS:
            assert dep in node_call, f"curated dep {dep!r} missing from {node_call}"
        assert "--retries" in node_call
        assert node_call[node_call.index("--retries") + 1] == "10"

    def test_node_deps_do_not_include_torch(self, sandbox):
        """The curated node-deps command must NOT contain torch/torchvision —
        those are pinned in step (a) so a transitive dep can never bump them
        off the cu124 pairing."""
        calls = self._run_venv_step(sandbox)
        torch_call = next(c for c in calls if any("torch==" in tok for tok in c))
        node_call = next(
            (c for c in calls if c is not torch_call and "pip" in " ".join(c)),
            None,
        )
        assert node_call is not None
        node_text = " ".join(node_call).lower()
        assert "torch==" not in node_text, f"node-deps must not re-install torch: {node_call}"
        assert "torchvision==" not in node_text, f"node-deps must not re-install torchvision: {node_call}"

    def test_runtime_deps_constant_is_curated(self):
        """RUNTIME_DEPS should be the curated subset: include fire/diffusers,
        exclude the heavy demo/dev deps and the torch pairing."""
        deps = setup.RUNTIME_DEPS
        for required in ("fire", "diffusers", "transformers", "accelerate", "huggingface-hub", "mediapy", "decord"):
            assert required in deps, f"RUNTIME_DEPS missing {required!r}: {deps}"
        for excluded in ("torch", "torchvision", "xformers", "gradio", "pytest", "matplotlib"):
            assert excluded not in deps, f"RUNTIME_DEPS must not include {excluded!r}: {deps}"

    def test_pip_mirror_forwarded_to_both_pip_installs(self, sandbox):
        mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
        calls = self._run_venv_step(sandbox, pip_mirror=mirror)

        torch_call = next(c for c in calls if any("torch==" in tok for tok in c))
        node_call = next(
            (c for c in calls if c is not torch_call and "pip" in " ".join(c)),
            None,
        )
        assert node_call is not None

        assert "--index-url" in torch_call
        assert setup.TORCH_INDEX_URL in torch_call
        assert "-i" in torch_call
        assert torch_call[torch_call.index("-i") + 1] == mirror
        assert "-i" in node_call
        assert node_call[node_call.index("-i") + 1] == mirror


# ---------------------------------------------------------------------------
# Step 3: model weights via snapshot_download (skip existing; hf primary)
# ---------------------------------------------------------------------------


class TestDownloadModels:
    def test_skip_model_short_circuits(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-depthcrafter")
        with patch("subprocess.check_call") as mock_cc:
            setup.download_models(None, skip_model=True, dry_run=False, buffer=setup.DryRunBuffer())
            mock_cc.assert_not_called()
        assert any("model download skipped" in r.message.lower() for r in caplog.records)

    def test_skips_when_snapshot_present(self, sandbox, caplog):
        """A model dir that already contains files is not re-downloaded."""
        caplog.set_level("INFO", logger="setup-depthcrafter")
        sandbox.model_dir.mkdir(parents=True)
        (sandbox.model_dir / "model.safetensors").write_text("fake weights")

        fake_hf = MagicMock()
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        fake_hf.snapshot_download.assert_not_called()
        assert any("already present" in r.message.lower() for r in caplog.records)

    def test_snapshot_download_used(self, sandbox):
        """When huggingface_hub is importable, snapshot_download is the primary path."""
        sandbox.model_dir.mkdir(parents=True)

        fake_hf = MagicMock()

        def fake_snapshot(**kwargs):
            # Simulate the snapshot landing on disk.
            out = sandbox.model_dir / kwargs.get("filename", "model.safetensors")
            out.write_text("fake")

        fake_hf.snapshot_download = MagicMock(side_effect=fake_snapshot)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        fake_hf.snapshot_download.assert_called_once()
        call_kwargs = fake_hf.snapshot_download.call_args.kwargs
        assert call_kwargs["repo_id"] == setup._MODEL_REPO_ID
        assert str(sandbox.model_dir) in call_kwargs["local_dir"]

    def test_snapshot_download_failure_warns(self, sandbox, caplog):
        """If snapshot_download raises, the script warns (does not crash)."""
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        sandbox.model_dir.mkdir(parents=True)

        fake_hf = MagicMock()
        fake_hf.snapshot_download = MagicMock(side_effect=RuntimeError("network down"))
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert any("snapshot_download failed" in r.message.lower() for r in caplog.records)

    def test_missing_huggingface_hub_warns(self, sandbox, caplog):
        """If huggingface_hub is not installed, the step warns and returns."""
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        sandbox.model_dir.mkdir(parents=True)

        # Force ImportError on huggingface_hub.
        import builtins

        orig_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "huggingface_hub":
                raise ImportError("no hf")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            setup.download_models(None, skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert any("huggingface_hub not installed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 4: self-check
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_self_check_passes_on_exit_zero(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-depthcrafter")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == "--help"
            assert "run.py" in cmd
            assert mock_run.call_args.kwargs["cwd"] == str(sandbox.node_dir)
        assert any("self-check passed" in r.message.lower() for r in caplog.records)

    def test_self_check_raises_when_python_missing(self, sandbox, caplog):
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        with (
            patch("subprocess.run") as mock_run,
            pytest.raises(RuntimeError) as exc_info,
        ):
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())
            mock_run.assert_not_called()
        assert "venv python not found" in str(exc_info.value).lower()

    def test_self_check_raises_when_run_py_missing(self, sandbox, caplog):
        caplog.set_level("WARNING", logger="setup-depthcrafter")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        # No run.py created.
        with (
            patch("subprocess.run") as mock_run,
            pytest.raises(RuntimeError) as exc_info,
        ):
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())
            mock_run.assert_not_called()
        assert "run.py not found" in str(exc_info.value).lower()

    def test_self_check_raises_nonzero_exit_and_surface_stderr(self, sandbox, caplog):
        """A failing run.py --help must raise and include the real stderr."""
        caplog.set_level("INFO", logger="setup-depthcrafter")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = (
            "Traceback (most recent call last):\n  File 'run.py', line 5\nImportError: No module named 'fire'"
        )
        fake_result.stdout = ""
        with (
            patch("subprocess.run", return_value=fake_result) as mock_run,
            pytest.raises(RuntimeError) as exc_info,
        ):
            setup.self_check(None, dry_run=False, buffer=setup.DryRunBuffer())
        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == "--help"
        assert "run.py" in cmd
        err_msg = str(exc_info.value)
        assert "exit code 1" in err_msg
        assert "No module named 'fire'" in err_msg


# ---------------------------------------------------------------------------
# print_summary + env vars
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_prints_env_vars_and_first_run_note(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-depthcrafter")
        setup.print_summary(None)
        msgs = "\n".join(r.message for r in caplog.records)
        assert "DEPTHCRAFTER_REPO_DIR" in msgs
        assert "DEPTHCRAFTER_PYTHON" in msgs
        assert "DEPTHCRAFTER_MODEL_DIR" in msgs
        assert str(sandbox.node_dir) in msgs
        assert str(sandbox.python_exe) in msgs
        assert str(sandbox.model_dir) in msgs
        # First-run ~10 GB download note + base model repo id.
        assert "stable-video-diffusion-img2vid-xt" in msgs


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
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_hf = MagicMock()
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")) as mock_run,
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
        ):
            setup.main([])
            assert mock_cc.called  # git pull, torch, reqs
            assert mock_run.called  # self-check
            fake_hf.snapshot_download.assert_called_once()  # model download

    def test_main_skip_deps_and_skip_model(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        fake_hf = MagicMock()
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("tests.test_setup_depthcrafter.setup.self_check"),  # no venv/run.py here
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
        ):
            setup.main(["--skip-deps", "--skip-model"])
            # Only git pull (no venv/torch/reqs, no model download).
            cmds = [c[0][0] for c in mock_cc.call_args_list]
            assert all(cmd[:2] == ["git", "pull"] for cmd in cmds) or not cmds
            fake_hf.snapshot_download.assert_not_called()

    def test_main_repo_dir_forwarded(self, sandbox, tmp_path):
        """--repo-dir drives every step against the external checkout."""
        external = tmp_path / "D_DepthCrafter"
        external.mkdir()
        (external / ".git").mkdir()
        ext_venv = external / ".venv"
        ext_venv.mkdir()
        ext_python = ext_venv / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
        ext_python.parent.mkdir(parents=True, exist_ok=True)
        ext_python.write_text("# fake")
        (external / "run.py").write_text("# fake")

        fake_hf = MagicMock()
        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")) as mock_run,
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
        ):
            setup.main(["--repo-dir", str(external)])
            # venv was created inside the external dir, not the in-repo path.
            venv_calls = [c[0][0] for c in mock_cc.call_args_list if c[0][0][1:3] == ["-m", "venv"]]
            assert not venv_calls  # venv already exists
            # git pull ran in the external dir.
            pull_calls = [c for c in mock_cc.call_args_list if c[0][0] == ["git", "pull"]]
            assert pull_calls
            assert pull_calls[0].kwargs["cwd"] == str(external)
            # self-check ran against the external run.py.
            assert mock_run.call_args.kwargs["cwd"] == str(external)

    def test_main_exits_nonzero_when_self_check_fails(self, sandbox, caplog):
        """When run.py --help returns non-zero, main() must exit non-zero
        (propagate the failure), not print a success summary."""
        caplog.set_level("INFO", logger="setup-depthcrafter")
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_hf = MagicMock()
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stderr = "ImportError: No module named 'fire'"
        fake_result.stdout = ""

        with (
            patch("subprocess.check_call"),
            patch("subprocess.run", return_value=fake_result),
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
            pytest.raises(SystemExit) as exc_info,
        ):
            setup.main([])

        assert exc_info.value.code == 1, "bootstrap must exit non-zero on self-check failure"
        fake_hf.snapshot_download.assert_called_once()
        msgs = "\n".join(r.message for r in caplog.records)
        assert "failed" in msgs.lower() or "exit code" in msgs.lower()

    def test_main_succeeds_without_node_manifest(self, sandbox, caplog):
        """The curated deps list is a module constant, so the bootstrap no
        longer depends on the checkout shipping a pyproject.toml or
        requirements.txt.  A minimal checkout (no manifest) must complete
        successfully — the old fatal 'no pyproject.toml' path is gone."""
        caplog.set_level("INFO", logger="setup-depthcrafter")
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        # Deliberately: no pyproject.toml, no requirements.txt.
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "run.py").write_text("# fake")

        fake_hf = MagicMock()
        with (
            patch("subprocess.check_call"),
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")),
            patch.dict("sys.modules", {"huggingface_hub": fake_hf}),
        ):
            setup.main([])  # must not raise / SystemExit
        fake_hf.snapshot_download.assert_called_once()

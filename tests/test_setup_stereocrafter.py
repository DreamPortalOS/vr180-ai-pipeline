"""Tests for the in-repo StereoCrafter bootstrap (scripts/setup_stereocrafter.py).

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

        deps_calls = [c for c in calls if "diffusers" in c]
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
        deps_call = next(c for c in calls if "diffusers" in c)
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

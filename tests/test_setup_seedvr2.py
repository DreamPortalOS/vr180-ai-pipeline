"""Tests for the in-repo SeedVR2 bootstrap (scripts/setup_seedvr2.py).

CI-safe: zero network, zero downloads.  The ``--dry-run`` path is asserted
verbatim; the real paths mock ``subprocess.check_call`` / ``subprocess.run``
and the filesystem so nothing leaves the test sandbox.
"""

from __future__ import annotations

import importlib

# Import the module by path (it is a standalone script, not a package member).
import importlib.util
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_setup_module():
    spec = importlib.util.spec_from_file_location(
        "setup_seedvr2",
        Path(__file__).resolve().parent.parent / "scripts" / "setup_seedvr2.py",
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

    node_dir = repo_root / "third_party" / "seedvr2_videoupscaler"
    venv_dir = node_dir / ".venv"
    python_exe = venv_dir / (Path("Scripts") / "python.exe" if os.name == "nt" else Path("bin") / "python")
    model_dir = repo_root / "models" / "SEEDVR2"

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
        """A full --dry-run emits exactly the 5 planned steps in order."""
        buf = setup.DryRunBuffer()

        setup.ensure_node_repo(dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, dry_run=True, buffer=buf)
        setup.download_models(skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(dry_run=True, buffer=buf)

        # 1 (clone) + 2 (venv, torch, reqs) + 2 (model downloads) + 1 (self-check)
        labels = [s.lower() for s in buf.steps]
        assert any("git clone" in s for s in labels), f"missing git clone in {buf.steps}"
        assert any("venv" in s for s in labels), f"missing venv step in {buf.steps}"
        assert any("torch" in s and "2.6.0" in s for s in labels), f"missing torch step in {buf.steps}"
        assert any("requirements.txt" in s for s in labels), f"missing requirements step in {buf.steps}"
        assert any("hf_hub_download" in s for s in labels), f"missing model step in {buf.steps}"
        assert any("self-check" in s for s in labels), f"missing self-check in {buf.steps}"

        # Order: clone before venv before torch before reqs before model before self-check
        clone_idx = next(i for i, s in enumerate(labels) if "git clone" in s)
        venv_idx = next(i for i, s in enumerate(labels) if "venv" in s)
        torch_idx = next(i for i, s in enumerate(labels) if "torch" in s)
        reqs_idx = next(i for i, s in enumerate(labels) if "requirements.txt" in s)
        model_idx = next(i for i, s in enumerate(labels) if "hf_hub_download" in s)
        check_idx = next(i for i, s in enumerate(labels) if "self-check" in s)
        assert clone_idx < venv_idx < torch_idx < reqs_idx < model_idx < check_idx

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
        # When --skip-deps is given, ensure_venv_and_deps is not called at all,
        # so the venv-creation step and torch/requirements pip installs are absent.
        # (self_check's label contains .venv in the python path — that's expected;
        #  we only forbid the explicit "-m venv" creation step and torch installs.)
        setup.download_models(skip_model=False, dry_run=True, buffer=buf)
        setup.self_check(dry_run=True, buffer=buf)
        assert not any("-m venv" in s for s in buf.steps)
        assert not any("pip" in s and "torch" in s for s in buf.steps)

    def test_dry_run_skip_model_omits_model_steps(self, sandbox):
        buf = setup.DryRunBuffer()
        setup.ensure_node_repo(dry_run=True, buffer=buf)
        setup.ensure_venv_and_deps(None, dry_run=True, buffer=buf)
        setup.download_models(skip_model=True, dry_run=True, buffer=buf)
        setup.self_check(dry_run=True, buffer=buf)
        assert not any("hf_hub_download" in s.lower() for s in buf.steps)
        assert not any("curl" in s.lower() for s in buf.steps)


# ---------------------------------------------------------------------------
# Step 1: ensure_node_repo (clone vs pull)
# ---------------------------------------------------------------------------


class TestEnsureNodeRepo:
    def test_clone_when_absent(self, sandbox):
        """If the node dir doesn't exist, git clone is invoked."""
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(dry_run=False, buffer=setup.DryRunBuffer())
            # First (and only) call should be the git clone.
            assert mock_cc.call_count == 1
            cmd = mock_cc.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]
            assert setup.NODE_REPO_URL in cmd
            assert str(sandbox.node_dir) in cmd

    def test_pull_when_git_checkout_present(self, sandbox):
        """If the node dir exists and is a git checkout, git pull is run instead."""
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()  # marks it as a git checkout
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(dry_run=False, buffer=setup.DryRunBuffer())
            assert mock_cc.call_count == 1
            cmd = mock_cc.call_args[0][0]
            assert cmd == ["git", "pull"]
            assert mock_cc.call_args.kwargs["cwd"] == str(sandbox.node_dir)

    def test_reclones_when_dir_exists_but_not_git(self, sandbox):
        """If the dir exists but isn't a git checkout, clone again."""
        sandbox.node_dir.mkdir(parents=True)
        # No .git dir → not a checkout → should clone
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_node_repo(dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_cc.call_args[0][0]
            assert cmd[:2] == ["git", "clone"]

    def test_proxy_hint_on_clone_failure(self, sandbox, caplog):
        """A failed git clone surfaces a proxy hint and re-raises."""
        caplog.set_level("WARNING", logger="setup-seedvr2")
        with (
            patch("subprocess.check_call", side_effect=subprocess.CalledProcessError(1, "git")),
            pytest.raises(subprocess.CalledProcessError),
        ):
            setup.ensure_node_repo(dry_run=False, buffer=setup.DryRunBuffer())
        # The proxy hint is logged as a warning.
        assert any("proxy" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 2: venv + pip install (torch cu124 + requirements)
# ---------------------------------------------------------------------------


class TestEnsureVenvAndDeps:
    def test_creates_venv_when_absent(self, sandbox):
        """If the venv python doesn't exist, ``python -m venv`` is the first call."""
        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, dry_run=False, buffer=setup.DryRunBuffer())
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert venv_calls, f"no venv creation call in {calls}"
        assert str(sandbox.venv_dir) in venv_calls[0]

    def test_skips_venv_when_present(self, sandbox):
        """If the venv python exists, no ``-m venv`` call is made."""
        sandbox.venv_dir.mkdir(parents=True)
        # Create the python marker file so INREPO_PYTHON.is_file() is True.
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake python")
        # And a requirements.txt so the deps step doesn't bail early.
        (sandbox.node_dir / "requirements.txt").write_text("numpy\n")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, dry_run=False, buffer=setup.DryRunBuffer())
        venv_calls = [c for c in calls if c[1:3] == ["-m", "venv"]]
        assert not venv_calls, f"venv should not be recreated: {venv_calls}"

    def test_torch_uses_stable_cu124_not_nightly(self, sandbox):
        """The torch install command must pin 2.6.0 + the cu124 index, NOT nightly."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "requirements.txt").write_text("numpy\n")

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(None, dry_run=False, buffer=setup.DryRunBuffer())

        torch_calls = [c for c in calls if any("torch==" in tok for tok in c)]
        assert torch_calls, f"no torch install call in {calls}"
        tc = torch_calls[0]
        # Must pin 2.6.0
        assert any(tok == "torch==2.6.0" for tok in tc), f"torch not pinned to 2.6.0: {tc}"
        assert any(tok == "torchvision==2.6.0" for tok in tc), f"torchvision not pinned: {tc}"
        # Must use the stable cu124 index — NOT nightly
        assert setup.TORCH_INDEX_URL in tc
        assert "cu124" in " ".join(tc)
        assert "nightly" not in " ".join(tc).lower()
        # Must include --retries 10
        assert "--retries" in tc
        assert tc[tc.index("--retries") + 1] == "10"

    def test_pip_mirror_forwarded_to_pip_not_torch_index(self, sandbox):
        """--pip-mirror adds -i to the requirements install but torch keeps --index-url."""
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "requirements.txt").write_text("numpy\n")
        mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"

        calls: list[list[str]] = []

        def fake_check_call(cmd, *args, **kwargs):
            calls.append(list(cmd))

        with patch("subprocess.check_call", side_effect=fake_check_call):
            setup.ensure_venv_and_deps(mirror, dry_run=False, buffer=setup.DryRunBuffer())

        torch_call = next(c for c in calls if any("torch==" in tok for tok in c))
        reqs_call = next(c for c in calls if "-r" in c)
        # requirements call gets the mirror as -i
        assert "-i" in reqs_call
        assert reqs_call[reqs_call.index("-i") + 1] == mirror
        # torch call keeps its own cu124 index-url AND the mirror as -i
        assert "--index-url" in torch_call
        assert setup.TORCH_INDEX_URL in torch_call
        assert "-i" in torch_call
        assert torch_call[torch_call.index("-i") + 1] == mirror

    def test_missing_requirements_warns_and_skips(self, sandbox, caplog):
        """If requirements.txt is absent, the deps step warns and doesn't crash."""
        caplog.set_level("WARNING", logger="setup-seedvr2")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        with patch("subprocess.check_call") as mock_cc:
            setup.ensure_venv_and_deps(None, dry_run=False, buffer=setup.DryRunBuffer())
        # Only the torch call should have happened (no -r call).
        reqs_calls = [c[0][0] for c in mock_cc.call_args_list if "-r" in c[0][0]]
        assert reqs_calls == []
        assert any("requirements.txt not found" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Step 3: model download (skip existing >1 GB; hf → curl fallback)
# ---------------------------------------------------------------------------


class TestDownloadModels:
    def test_skip_model_short_circuits(self, sandbox, caplog):
        """--skip-model logs and does nothing."""
        caplog.set_level("INFO", logger="setup-seedvr2")
        with patch("subprocess.check_call") as mock_cc:
            setup.download_models(skip_model=True, dry_run=False, buffer=setup.DryRunBuffer())
            mock_cc.assert_not_called()
        assert any("model download skipped" in r.message.lower() for r in caplog.records)

    def test_skips_existing_large_model(self, sandbox):
        """A model file >1 GB already on disk is not re-downloaded."""
        sandbox.model_dir.mkdir(parents=True)
        # Write a >1 GB file via os.truncate (sparse on most FS, no real disk use).
        big = sandbox.model_dir / setup._MODELS[0][0]
        big.write_bytes(b"")
        with open(big, "ab"):
            os.truncate(big, setup._MODEL_SKIP_SIZE + 1)

        with patch.object(setup, "_download_with_hf_hub") as mock_hf:
            setup.download_models(skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
            # The big existing file is skipped → only the VAE download runs.
            mock_hf.assert_called_once()
            called_filename = mock_hf.call_args.kwargs.get("filename") or mock_hf.call_args.args[0]
            assert called_filename == setup._MODELS[1][0]  # the VAE, not the DIT

    def test_hf_hub_download_used_first(self, sandbox):
        """When hf_hub_download import succeeds, it is the primary path."""
        sandbox.model_dir.mkdir(parents=True)

        # Inject a fake huggingface_hub module so the import inside succeeds.
        fake_hf = MagicMock()

        def fake_download(**kwargs):
            # Simulate the file landing on disk and return its path.
            out = sandbox.model_dir / kwargs["filename"]
            out.write_text("fake weights")
            return str(out)

        fake_hf.hf_hub_download = fake_download
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf}):
            setup.download_models(skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        # Both model files should now exist (the fake writes them).
        assert (sandbox.model_dir / setup._MODELS[0][0]).exists()
        assert (sandbox.model_dir / setup._MODELS[1][0]).exists()

    def test_curl_fallback_on_hf_failure(self, sandbox, caplog):
        """If hf_hub_download raises, curl resume download is used instead."""
        sandbox.model_dir.mkdir(parents=True)

        def boom(*a, **k):
            raise RuntimeError("network down")

        curl_calls: list[list[str]] = []

        def fake_curl(filename, *, dry_run, buffer):
            curl_calls.append(filename)

        with (
            patch.object(setup, "_download_with_hf_hub", side_effect=boom),
            patch.object(setup, "_download_with_curl", side_effect=fake_curl),
        ):
            setup.download_models(skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert sorted(curl_calls) == sorted([m[0] for m in setup._MODELS])
        assert any("falling back to curl" in r.message.lower() for r in caplog.records)

    def test_curl_command_is_list_form_no_shell(self, sandbox):
        """The curl invocation must be a subprocess list, never shell=True."""
        sandbox.model_dir.mkdir(parents=True)
        captured: list[list[str]] = []
        captured_kwargs: list[dict] = []

        def fake_check_call(cmd, *args, **kwargs):
            captured.append(list(cmd))
            captured_kwargs.append(kwargs)
            # touch the output file so a follow-up size-check would pass
            out = next(p for p in cmd if p.endswith(".safetensors"))
            Path(out).touch()

        with (
            patch.object(setup, "_download_with_hf_hub", side_effect=RuntimeError("no hf")),
            patch("subprocess.check_call", side_effect=fake_check_call),
        ):
            setup.download_models(skip_model=False, dry_run=False, buffer=setup.DryRunBuffer())
        assert captured, "curl was never invoked"
        for cmd, kw in zip(captured, captured_kwargs, strict=True):
            assert cmd[0] == "curl"
            assert "-C" in cmd and "-" in cmd  # resume flag
            assert "-o" in cmd
            assert all(isinstance(t, str) for t in cmd)  # list form, not a shell string
            assert not kw.get("shell", False), f"shell=True must never be set: {kw}"


# ---------------------------------------------------------------------------
# Step 4: self-check
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_self_check_passes_on_exit_zero(self, sandbox, caplog):
        """inference_cli.py --help returning exit 0 logs success."""
        caplog.set_level("INFO", logger="setup-seedvr2")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "inference_cli.py").write_text("# fake")

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stderr = ""
        with patch("subprocess.run", return_value=fake_result) as mock_run:
            setup.self_check(dry_run=False, buffer=setup.DryRunBuffer())
            cmd = mock_run.call_args[0][0]
            assert cmd[-1] == "--help"
            assert "inference_cli.py" in cmd
            assert mock_run.call_args.kwargs["cwd"] == str(sandbox.node_dir)
        assert any("self-check passed" in r.message.lower() for r in caplog.records)

    def test_self_check_skips_when_python_missing(self, sandbox, caplog):
        """If the venv python is absent, self-check is skipped with a warning."""
        caplog.set_level("WARNING", logger="setup-seedvr2")
        with patch("subprocess.run") as mock_run:
            setup.self_check(dry_run=False, buffer=setup.DryRunBuffer())
            mock_run.assert_not_called()
        assert any("skipping self-check" in r.message.lower() for r in caplog.records)

    def test_self_check_skips_when_cli_missing(self, sandbox, caplog):
        caplog.set_level("WARNING", logger="setup-seedvr2")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        # No inference_cli.py created
        with patch("subprocess.run") as mock_run:
            setup.self_check(dry_run=False, buffer=setup.DryRunBuffer())
            mock_run.assert_not_called()
        assert any("skipping self-check" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# print_summary + env vars
# ---------------------------------------------------------------------------


class TestSummary:
    def test_summary_prints_three_env_vars(self, sandbox, caplog):
        caplog.set_level("INFO", logger="setup-seedvr2")
        setup.print_summary()
        msgs = "\n".join(r.message for r in caplog.records)
        assert "SEEDVR2_NODE_DIR" in msgs
        assert "SEEDVR2_PYTHON" in msgs
        assert "SEEDVR2_MODEL_DIR" in msgs
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
        # Nothing was created on disk.
        assert not sandbox.node_dir.exists()

    def test_main_full_flow_calls_all_steps(self, sandbox):
        """A non-dry-run main() invokes each step's real (mocked) subprocess."""
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        (sandbox.node_dir / "requirements.txt").write_text("numpy\n")
        sandbox.venv_dir.mkdir(parents=True)
        sandbox.python_exe.parent.mkdir(parents=True, exist_ok=True)
        sandbox.python_exe.write_text("# fake")
        (sandbox.node_dir / "inference_cli.py").write_text("# fake")

        with (
            patch("subprocess.check_call") as mock_cc,
            patch("subprocess.run", return_value=MagicMock(returncode=0, stderr="")) as mock_run,
            patch.object(setup, "_download_with_hf_hub") as mock_hf,
        ):
            setup.main([])
            # git pull, venv (skipped, exists), torch, reqs, self-check(run)
            assert mock_cc.called
            assert mock_run.called  # self-check
            assert mock_hf.called  # model download

    def test_main_skip_deps_and_skip_model(self, sandbox):
        sandbox.node_dir.mkdir(parents=True)
        (sandbox.node_dir / ".git").mkdir()
        with (
            patch("subprocess.check_call") as mock_cc,
            patch.object(setup, "_download_with_hf_hub") as mock_hf,
        ):
            setup.main(["--skip-deps", "--skip-model"])
            # No venv/torch/reqs calls, no model calls — only git pull.
            cmds = [c[0][0] for c in mock_cc.call_args_list]
            assert all(cmd[:2] == ["git", "pull"] for cmd in cmds) or not cmds
            mock_hf.assert_not_called()

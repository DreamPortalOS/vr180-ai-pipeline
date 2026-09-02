"""K-15 (#205): ``scripts/`` entry points must import without PYTHONPATH.

The lead hit this as a pure stumbling block: invoking
``python scripts/make_comparison.py --help`` from the repo root crashed with
``ModuleNotFoundError: No module named 'pipeline'`` unless the caller manually
exported ``PYTHONPATH=<repo>``.  Every entry script now bootstraps the repo
root onto ``sys.path`` itself (see the ``_REPO_ROOT`` block at the top of each
``scripts/*.py``), so direct invocation works from any CWD.

This test proves that fix the way the lead experienced it: a **subprocess**
with ``PYTHONPATH`` *stripped* from its environment, running ``--help`` and
asserting exit 0.  ``--help`` is the right probe — argparse exits 0 only after
the module has finished importing and the parser is built, so a non-zero exit
here means an import-time crash (exactly the ``ModuleNotFoundError`` class of
bug), not an argparse complaint.

subprocess is list-form (no ``shell=True``) per repo discipline.  This test
complements ``tests/test_cli_smoke.py`` (which *sets* ``PYTHONPATH`` to assert
the CLI contracts hold under the package-import path); this one asserts the
**direct-invocation** path works without it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Entry scripts that carry the K-15 sys.path bootstrap.  Each must run
# ``--help`` to exit 0 when invoked directly with no PYTHONPATH.
SCRIPTS: list[str] = [
    "make_comparison.py",
    "run_pipeline.py",
    "e2e_smoke.py",
    "batch_runner.py",
    "generate.py",
    "vr180_qa.py",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _env_without_pythonpath() -> dict[str, str]:
    """A copy of the current environment with PYTHONPATH removed.

    The whole point of K-15 is that the caller does NOT set PYTHONPATH, so the
    test env must reflect that — copying os.environ would leak the test
    runner's own PYTHONPATH (e.g. ``PYTHONPATH=.`` from the CI gate) and mask
    the very regression this test exists to catch.
    """
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_help_without_pythonpath(script: str) -> None:
    """Each entry script's ``--help`` exits 0 with no PYTHONPATH set.

    A non-zero exit means the script crashed on import (e.g. the
    ``ModuleNotFoundError: No module named 'pipeline'`` that K-15 fixed) — the
    bootstrap is missing or broken.  The failure message names the script and
    the captured stderr so the nightly reviewer can localise the break.
    """
    script_path = SCRIPTS_DIR / script
    proc = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True,
        text=True,
        env=_env_without_pythonpath(),
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"[{script}] '--help' crashed without PYTHONPATH (exit={proc.returncode}). "
        f"This is the K-15 regression — the sys.path bootstrap is missing/broken. "
        f"stderr:\n{proc.stderr[-600:]}"
    )

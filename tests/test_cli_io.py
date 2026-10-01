"""Issue #441 — CLI console hardening regression tests.

On Windows a bare ``python scripts/<cli>.py`` gives ``sys.stdout`` /
``sys.stderr`` the *locale* codec (cp936/GBK on a zh-CN box) unless the caller
remembers ``PYTHONUTF8=1`` or ``PYTHONIOENCODING``.  That codec cannot encode
plenty of characters the CLIs print on purpose — ``²`` in the
``run_pipeline.py`` ``--quality`` help ("1920²/eye"), ``⇄`` in
``dome_frame_tools.py``, emoji in operator-facing messages — so even
``--help`` died with::

    UnicodeEncodeError: 'gbk' codec can't encode character '\\u00b2'

before argparse had finished printing.  The operator's only escape hatch was to
know about ``PYTHONUTF8``, which is not a fix.

The fix is :func:`scripts.cli_io.configure_console`: it flips the *error
handler* of the live ``sys.stdout`` / ``sys.stderr`` objects to
``backslashreplace`` and leaves their encoding alone.  These tests pin the
properties that make that safe:

1. a GBK console renders ``²`` / ``⇄`` / emoji as ASCII escapes instead of
   raising, while every character GBK *can* represent still comes out verbatim;
2. the console's own encoding is untouched, and a UTF-8 console still carries
   the exact unicode (no escaping, no mangling);
3. capture / substitute streams without ``reconfigure`` (``io.StringIO``) are
   skipped — never re-wrapped, never closed — and a stream that refuses the
   call does not take the CLI down either;
4. importing ``scripts.cli_io`` changes nothing, and each CLI calls the helper
   as the first statement of ``main()`` so the guard cannot silently rot;

plus the end-to-end regression: the three real CLIs exit 0 on ``--help`` in a
subprocess with ``PYTHONIOENCODING=gbk`` and ``PYTHONUTF8`` removed, captured
as **bytes** and decoded as GBK here.

The subprocess environment is built by *editing* a copy of the real
environment instead of using a "clean env" fixture: an ambient ``PYTHONUTF8=1``
is exactly what would mask this bug, so the test removes that variable
explicitly rather than laundering the environment into a shape where every CLI
looks fixed.
"""

from __future__ import annotations

import ast
import importlib
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.cli_io import CONSOLE_ERRORS, configure_console

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"

#: The three CLI entry points that carry the #441 guard.
CLI_SCRIPTS = ("run_pipeline.py", "generate.py", "dome_frame_tools.py")

#: Characters GBK cannot represent, so a hardened GBK stream must escape them
#: (these two are precisely what made ``--help`` crash in the issue).
SQUARED = "\u00b2"
HARPOON = "\u21c4"
EMOJI = "\U0001f600"

#: Characters GBK *can* represent — they must survive byte-for-byte rather than
#: being escaped or dropped.
ARROW = "\u2192"
DOME = "穹顶"

#: Help-text escapes each CLI is expected to emit on a GBK console.  ``generate``
#: is absent on purpose: its help is fully GBK-representable, so it exercises the
#: "no escaping needed, still exit 0" path.
KNOWN_HELP_ESCAPES = {
    "run_pipeline.py": r"\xb2",
    "dome_frame_tools.py": r"\u21c4",
}


# ---------------------------------------------------------------------------
# Stream factories — exactly what a Windows console hands a script
# ---------------------------------------------------------------------------


def _gbk_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    """A GBK stream with ``errors="strict"``, i.e. the crash-prone default."""
    buf = io.BytesIO()
    return io.TextIOWrapper(buf, encoding="gbk", errors="strict"), buf


def _utf8_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    """A UTF-8 stream with ``errors="strict"`` — the CI / Linux default."""
    buf = io.BytesIO()
    return io.TextIOWrapper(buf, encoding="utf-8", errors="strict"), buf


def _install_streams(monkeypatch: pytest.MonkeyPatch, out: object, err: object) -> None:
    """Point ``sys.stdout`` / ``sys.stderr`` at *out* / *err* for one test."""
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)


class _RefusingStream:
    """A stream that advertises ``reconfigure`` but refuses the call.

    Stands in for a closed / detached wrapper: the guard must swallow the
    failure, not surface it as a CLI crash.
    """

    def __init__(self) -> None:
        self.calls = 0

    def reconfigure(self, **kwargs: object) -> None:
        self.calls += 1
        raise ValueError("I/O operation on closed file")


# ---------------------------------------------------------------------------
# 1. GBK console: escapes instead of a traceback
# ---------------------------------------------------------------------------


def test_gbk_console_writes_emoji_squared_and_arrow(monkeypatch: pytest.MonkeyPatch) -> None:
    """'😀' / '²' / '→' can be written to a GBK console without crashing."""
    out, out_buf = _gbk_stream()
    err, err_buf = _gbk_stream()
    _install_streams(monkeypatch, out, err)

    configure_console()

    text = f"preview = 1920{SQUARED}/eye {ARROW} {EMOJI} {DOME}"
    print(text)  # must not raise
    print(text, file=sys.stderr)  # stderr is hardened too
    out.flush()
    err.flush()

    for rendered in (out_buf.getvalue().decode("gbk"), err_buf.getvalue().decode("gbk")):
        # Unencodable → ASCII escape (that is what backslashreplace buys us).
        assert r"\xb2" in rendered
        assert r"\U0001f600" in rendered
        # Encodable in GBK → written verbatim, not gratuitously escaped.
        assert ARROW in rendered
        assert DOME in rendered
    # Nothing was closed underneath us.
    assert not out.closed
    assert not err.closed


# ---------------------------------------------------------------------------
# 2. Encoding is preserved; UTF-8 output keeps its full unicode
# ---------------------------------------------------------------------------


def test_console_encoding_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only ``errors`` changes — the console's own codec stays what it was."""
    gbk, _ = _gbk_stream()
    utf8, _ = _utf8_stream()
    _install_streams(monkeypatch, gbk, utf8)

    configure_console()

    assert gbk.encoding.lower().replace("-", "") == "gbk"
    assert utf8.encoding.lower().replace("-", "") == "utf8"
    assert gbk.errors == CONSOLE_ERRORS
    assert utf8.errors == CONSOLE_ERRORS


def test_utf8_console_keeps_the_exact_unicode(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a UTF-8 console the hardening is a no-op: unicode round-trips exactly."""
    out, out_buf = _utf8_stream()
    err, _ = _utf8_stream()
    _install_streams(monkeypatch, out, err)

    configure_console()

    text = f"{DOME} {SQUARED} {ARROW} {EMOJI} 1920{SQUARED}/eye"
    print(text)
    out.flush()

    decoded = out_buf.getvalue().decode("utf-8")
    # TextIOWrapper translates "\n" to os.linesep (Windows: "\r\n"); normalise
    # that away so the assertion is about characters, not line endings.
    assert decoded.replace("\r\n", "\n") == text + "\n"


# ---------------------------------------------------------------------------
# 3. Capture / substitute streams are skipped, never re-wrapped or closed
# ---------------------------------------------------------------------------


def test_stream_without_reconfigure_is_skipped_not_rewrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """``io.StringIO`` has no ``reconfigure``: leave it exactly as it was."""
    buf = io.StringIO()
    _install_streams(monkeypatch, buf, buf)

    assert not hasattr(buf, "reconfigure")  # the case under test
    configure_console()  # must not raise

    assert sys.stdout is buf
    assert sys.stderr is buf  # never re-wrapped
    assert not buf.closed  # never closed
    print("still usable")
    assert buf.getvalue() == "still usable\n"


def test_refusing_stream_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stream whose ``reconfigure`` raises is swallowed, not propagated."""
    refusing = _RefusingStream()
    _install_streams(monkeypatch, refusing, refusing)

    configure_console()  # must not raise

    assert refusing.calls == 2  # stdout + stderr were attempted...
    assert sys.stdout is refusing  # ...and the object was left in place


# ---------------------------------------------------------------------------
# 4. No import-time side effects; every CLI calls the guard from main()
# ---------------------------------------------------------------------------


def test_import_does_not_touch_the_global_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing (and re-importing) the module must not reconfigure anything."""
    stream, _ = _gbk_stream()
    _install_streams(monkeypatch, stream, stream)

    import scripts.cli_io as cli_io

    importlib.reload(cli_io)

    assert stream.errors == "strict"  # an import-time side effect would show here
    assert sys.stdout is stream
    assert cli_io.CONSOLE_ERRORS == CONSOLE_ERRORS


def _main_body(path: Path) -> list[ast.stmt]:
    """Return the statements of the module-level ``main()``, minus its docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return body
    raise AssertionError(f"{path.name} has no module-level main()")


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_main_configures_the_console_first(script: str) -> None:
    """``configure_console()`` is the first statement of every CLI's ``main()``.

    Being first is the point: it must be in place before ``parse_args`` prints
    help/errors, before ``logging.basicConfig`` and before any pipeline print —
    otherwise the guard protects nothing.
    """
    path = SCRIPTS / script
    assert "from scripts.cli_io import configure_console" in path.read_text(encoding="utf-8")

    first = _main_body(path)[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), ast.dump(first)
    func = first.value.func
    assert isinstance(func, ast.Name) and func.id == "configure_console"


# ---------------------------------------------------------------------------
# End-to-end: the three CLIs on a GBK console
# ---------------------------------------------------------------------------


def _gbk_subprocess_env() -> dict[str, str]:
    """An environment that reproduces a bare GBK Windows console.

    ``os.environ.copy()`` plus explicit edits, *not* a scrubbed "clean" env: the
    bug lives in "nothing forced UTF-8", so the two variables that would mask it
    are removed by name and everything else is left as the runner had it.
    ``PYTHONPATH`` goes too — that is the direct-invocation path (K-15) users
    actually hit.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "gbk"  # cp936 console, errors=strict
    env.pop("PYTHONUTF8", None)  # UTF-8 mode would override PYTHONIOENCODING
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_help_exits_zero_on_a_gbk_console(script: str) -> None:
    """``python scripts/<cli>.py --help`` exits 0 with a GBK console codec.

    Byte capture (no ``text=True``) on purpose: the CLI's output *is* GBK bytes,
    so this test decodes them as GBK itself.  Before the fix the child died with
    a ``UnicodeEncodeError`` traceback and returned 1.
    """
    proc = subprocess.run(  # list form, no shell=True (repo discipline)
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        env=_gbk_subprocess_env(),
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"[{script}] '--help' exited {proc.returncode} on a GBK console; "
        f"stderr:\n{proc.stderr.decode('gbk', errors='replace')[-800:]}"
    )

    stdout = proc.stdout.decode("gbk")  # decodes only if the bytes really are GBK
    assert "usage" in stdout.lower()
    assert not proc.stderr.strip()  # a clean help run writes nothing to stderr

    # None of these glyphs can be written to a GBK console, so none of them may
    # show up raw; an ASCII escape in their place is the degraded-but-visible
    # form we want.
    for unencodable in (SQUARED, HARPOON):
        assert unencodable not in stdout

    expected_escape = KNOWN_HELP_ESCAPES.get(script)
    if expected_escape is not None:
        # The unencodable help text is still *shown* rather than dropped — that
        # is the difference between "degraded" and "lost".
        assert expected_escape in stdout

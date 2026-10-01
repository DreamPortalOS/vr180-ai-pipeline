"""Shared console hardening for the repo's CLI entry points (issue #441).

On Windows a bare ``python scripts/...py`` picks the *locale* codec for
``sys.stdout`` / ``sys.stderr`` (cp936/GBK on a zh-CN box, cp1252 elsewhere)
unless the caller remembers ``PYTHONUTF8=1`` or ``PYTHONIOENCODING``.  That
codec cannot encode plenty of characters the CLIs legitimately print — ``²`` in
the ``run_pipeline.py --quality`` help, ``⇄`` in ``dome_frame_tools.py``, any
emoji in an operator-facing message — and the resulting ``UnicodeEncodeError``
kills the process mid-write.  On a GBK console even ``--help`` died with a
traceback before argparse had finished printing.

:func:`configure_console` is the single shared fix.  It rewrites the *error
handler* of the current ``sys.stdout`` / ``sys.stderr`` to
``backslashreplace``, so a character the console codec cannot represent is
written as an ASCII escape (``\\xb2``, ``\\u21c4``) instead of raising.  The
stream keeps whatever encoding the console actually speaks, which is the whole
point: the bytes the console *can* represent still round-trip unchanged.

Three constraints are load-bearing:

* **The encoding is never touched.**  Forcing UTF-8 onto a GBK console is what
  produces mojibake on the operator's terminal; ``reconfigure(errors=...)``
  leaves the codec exactly as the interpreter negotiated it.
* **Streams are mutated in place, never re-wrapped and never closed.**  The
  module-level ``logging.basicConfig`` handlers in ``run_pipeline.py`` /
  ``generate.py`` capture the ``sys.stderr`` *object* at import time, so
  replacing ``sys.stderr`` would leave those handlers writing into the old,
  unhardened stream.
* **No import-time side effects.**  Importing this module changes nothing; each
  CLI calls :func:`configure_console` itself as the first statement of its
  ``main()``, which is early enough to also cover argparse's own help/error
  printing and every print/log line the pipeline emits afterwards.

Streams without a working ``reconfigure`` (``io.StringIO``, pytest's capture
objects, pipes wrapped by test harnesses) are skipped rather than replaced or
closed — they either cannot raise ``UnicodeEncodeError`` for a stray character
or belong to a harness that owns its own encoding.
"""

from __future__ import annotations

import sys

#: Error handler applied to the CLI console streams.  ``backslashreplace``
#: keeps everything the console codec can represent and escapes the rest, so a
#: write can never blow up halfway through a line.
CONSOLE_ERRORS = "backslashreplace"


def configure_console() -> None:
    """Harden ``sys.stdout`` / ``sys.stderr`` against the console's codec.

    Call this as the first statement of a CLI ``main()`` — before ``parse_args``
    (argparse prints help and errors itself), before ``logging.basicConfig`` and
    before any ``print``, so that neither the CLI nor the pipeline it drives can
    die on a character the console cannot encode.

    Idempotent, side-effect free with respect to the process environment, and
    deliberately silent: a stream it cannot reconfigure is left exactly as it
    was found.
    """
    for stream in (sys.stdout, sys.stderr):
        _harden_stream(stream)


def _harden_stream(stream: object) -> None:
    """Set ``errors='backslashreplace'`` on *stream* when it supports it.

    Streams are reconfigured in place so callers that already hold a reference
    (e.g. a ``logging.StreamHandler`` created at import time) keep working.  A
    stream without ``reconfigure`` is skipped; a stream that refuses the call
    (closed, detached, or otherwise not reconfigurable) keeps its previous
    settings.  Neither is ever re-wrapped or closed.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if not callable(reconfigure):
        return
    try:
        reconfigure(errors=CONSOLE_ERRORS)
    except (AttributeError, OSError, TypeError, ValueError):
        # ``io.UnsupportedOperation`` is both an OSError and a ValueError, so
        # the closed / detached / non-seekable cases are all covered here.
        return

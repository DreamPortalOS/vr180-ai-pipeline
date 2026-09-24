"""Source-level guards for studio/static/app.js regressions found in browser QA."""

from __future__ import annotations

import re
from pathlib import Path

APP_JS = Path(__file__).resolve().parent.parent / "studio" / "static" / "app.js"


def _function_body(src: str, name: str) -> str:
    start = src.index(f"async function {name}(")
    brace = src.index("{", start)
    depth = 0
    for i in range(brace, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[brace : i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_upload_file_reads_the_response_body_once() -> None:
    """#417: a second ``res.json()`` threw "body stream already read" on every
    successful upload, so dropped files never became input nodes."""
    body = _function_body(APP_JS.read_text(encoding="utf-8"), "uploadFile")
    reads = re.findall(r"\bres\.(json|text|blob|arrayBuffer)\(\)", body)
    assert len(reads) == 1, f"uploadFile must read the response once, found {reads}"

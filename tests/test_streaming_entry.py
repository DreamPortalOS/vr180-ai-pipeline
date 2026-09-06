"""F-5 (#268): streaming entry-point hygiene.

The module-level ``run_streaming_pipeline()`` convenience wrapper in
``pipeline/streaming_pipeline.py`` was dead code: it forwarded a
``flip_vertical`` kwarg that :class:`StreamingPipeline.__init__` does not accept
(so every call raised ``TypeError``) and nothing in the repo called it.  It has
been deleted; the CLI streaming branch in ``scripts/run_pipeline.py`` is the
single supported entry point.

These tests keep it that way and, more importantly, guard against the root
cause recurring -- a constructor and its only call site drifting apart:

  - ``run_streaming_pipeline`` must not reappear on the module, and the
    constructor must still reject the ``flip_vertical`` kwarg it used to pass.
  - The set of keyword args ``scripts/run_pipeline.py`` passes to
    ``StreamingPipeline(...)`` must equal the set of parameters
    ``StreamingPipeline.__init__`` accepts.  Either side gaining or losing a
    knob without the other is exactly the #120 / #243 / #268 defect class.

CPU-only, no model download, no inference: ``scripts/run_pipeline.py`` is
parsed with :mod:`ast` rather than imported.
"""

import ast
import inspect
import os
import sys
import unittest

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pipeline.streaming_pipeline as streaming_pipeline  # noqa: E402
from pipeline.streaming_pipeline import StreamingPipeline  # noqa: E402

RUN_PIPELINE_PATH = os.path.join(PROJECT_ROOT, "scripts", "run_pipeline.py")


def _streaming_pipeline_call_kwargs(source_path: str) -> list[set[str]]:
    """Return the kwarg-name set of every ``StreamingPipeline(...)`` call in *source_path*.

    A ``**splat`` keyword is recorded as ``"**"`` so callers can reject it --
    a splat would hide constructor drift from this guard.
    """
    with open(source_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=source_path)

    found: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "StreamingPipeline":
            continue
        found.append({kw.arg if kw.arg is not None else "**" for kw in node.keywords})
    return found


class TestDeadWrapperStaysRemoved(unittest.TestCase):
    def test_run_streaming_pipeline_is_gone(self):
        self.assertFalse(hasattr(streaming_pipeline, "run_streaming_pipeline"))

    def test_constructor_rejects_flip_vertical(self):
        # The exact kwarg the dead wrapper forwarded.  Python rejects it at
        # call binding, before __init__ runs, so no device/model work happens.
        # If flip_vertical is ever added deliberately, update this test then.
        with self.assertRaises(TypeError):
            StreamingPipeline(flip_vertical=True)


class TestConstructorMatchesCliCallSite(unittest.TestCase):
    """#268 acceptance: ``StreamingPipeline.__init__`` parameter names ==
    keyword args at the single ``scripts/run_pipeline.py`` construction site."""

    def _call_site_kwargs(self) -> set[str]:
        calls = _streaming_pipeline_call_kwargs(RUN_PIPELINE_PATH)
        self.assertEqual(
            len(calls),
            1,
            f"expected exactly one StreamingPipeline(...) call in scripts/run_pipeline.py, found {len(calls)}",
        )
        return calls[0]

    def test_exactly_one_call_site_in_run_pipeline(self):
        self._call_site_kwargs()

    def test_no_kwargs_splat_at_call_site(self):
        self.assertNotIn("**", self._call_site_kwargs(), "a **splat would hide constructor drift from this guard")

    def test_init_has_no_var_keyword_param(self):
        # A **kwargs on the constructor would likewise make set-equality
        # meaningless (any name would be "accepted").
        kinds = {p.kind for p in inspect.signature(StreamingPipeline.__init__).parameters.values()}
        self.assertNotIn(inspect.Parameter.VAR_KEYWORD, kinds)
        self.assertNotIn(inspect.Parameter.VAR_POSITIONAL, kinds)

    def test_kwarg_set_equals_init_param_set(self):
        call_kwargs = self._call_site_kwargs()
        init_params = set(inspect.signature(StreamingPipeline.__init__).parameters) - {"self"}

        missing_at_call_site = init_params - call_kwargs
        unknown_at_call_site = call_kwargs - init_params
        self.assertEqual(
            (missing_at_call_site, unknown_at_call_site),
            (set(), set()),
            "StreamingPipeline.__init__ and scripts/run_pipeline.py have drifted:\n"
            f"  accepted by __init__ but not passed by run_pipeline: {sorted(missing_at_call_site)}\n"
            f"  passed by run_pipeline but not accepted by __init__: {sorted(unknown_at_call_site)}",
        )


if __name__ == "__main__":
    unittest.main()

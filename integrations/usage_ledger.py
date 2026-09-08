"""Generation usage ledger + budget soft-gate (issue #328, W-11).

Before this module the only trace of a paid generation was a log line
(``usage={'completion_tokens': 48400}``) that vanished with the terminal
scrollback.  Nothing accumulated, so nothing could warn — and a single
4k/10s run costs an order of magnitude more than the 480p/5s baseline
(the measured incident: 1,952,100 tokens in one shot).  The owner's Ark
quota was burned through with no signal at all.

What this module adds:

* **Ledger** — one JSON line per *successful* generation appended to
  ``~/.vr180/usage_ledger.jsonl`` (override with ``VR180_LEDGER_PATH``).
  Never inside the repo.
* **Pricing** — deliberately *not* hard-coded.  ``VR180_ARK_PRICE_PER_MTOKEN``
  (元 per million tokens) is the only source; with it unset the ledger records
  token counts and says so instead of inventing a number.
* **Soft gate** — a cumulative cap (``--budget-cap`` / ``VR180_BUDGET_CAP``,
  or ``VR180_BUDGET_CAP_TOKENS`` when no unit price is configured).  At 80%
  it warns; at 100% it raises :class:`BudgetExceededError` **before the submit
  request is sent**, which is the entire point — blocking after the POST
  would cost exactly as much as not blocking at all.

The real hard limit lives in the Ark console; this is the local guard rail
that fires long before the console one does.

Read-only inspection::

    python -m integrations.usage_ledger --summary

Two deliberate non-negotiables in here:

* **A ledger failure must never fail a generation.**  The generation was
  already paid for; losing the result over a bookkeeping problem is strictly
  worse than losing the bookkeeping.  :func:`record_generation` therefore
  swallows every exception and returns ``None``.
* **A test run must never touch the operator's real ledger.**  See
  :func:`_writes_blocked`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration surface (environment variables — no magic numbers in code)
# ---------------------------------------------------------------------------

#: Override for the ledger file location. Tests **must** set this.
ENV_LEDGER_PATH = "VR180_LEDGER_PATH"

#: Unit price in 元 per million tokens.  Intentionally unset by default: the
#: Ark price list changes (tiered by model / resolution / promo window), and a
#: stale hard-coded number is worse than an honest "not configured".
ENV_PRICE_PER_MTOKEN = "VR180_ARK_PRICE_PER_MTOKEN"

#: Cumulative spend cap in 元. Requires a configured unit price.
ENV_BUDGET_CAP = "VR180_BUDGET_CAP"

#: Cumulative token cap — the fallback gate when no unit price is configured.
ENV_BUDGET_CAP_TOKENS = "VR180_BUDGET_CAP_TOKENS"

#: Fraction of the cap at which a loud warning is printed before generating.
WARN_RATIO = 0.8

#: Fraction of the cap at which submission is refused outright.
BLOCK_RATIO = 1.0

_TOKENS_PER_MTOKEN = 1_000_000

#: Ledger modes. ``amount`` = 元 accounting, ``tokens`` = token accounting
#: (no unit price configured), ``off`` = no cap configured, warn/block never fire.
MODE_AMOUNT = "amount"
MODE_TOKENS = "tokens"
MODE_OFF = "off"

_UNIT_YUAN = "元"
_UNIT_TOKENS = "tokens"


class BudgetExceededError(RuntimeError):
    """Raised *before* a generation is submitted when the cap is reached.

    Subclasses :class:`RuntimeError` so existing provider/CLI error handling
    treats it as a normal failure (non-zero exit) even where it is not
    caught by name.
    """


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_ledger_path() -> Path:
    """The operator's real ledger: ``~/.vr180/usage_ledger.jsonl``.

    Outside the repo on purpose — usage history is machine state, not source.
    """
    return Path.home() / ".vr180" / "usage_ledger.jsonl"


def ledger_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the ledger file: explicit argument > env var > default."""
    if path is not None:
        return Path(path).expanduser()
    override = os.environ.get(ENV_LEDGER_PATH, "").strip()
    if override:
        return Path(override).expanduser()
    return default_ledger_path()


def _writes_blocked(explicit_path: str | os.PathLike[str] | None) -> bool:
    """Return True when a write to the *default* ledger must be suppressed.

    Under pytest, provider code paths that record usage are exercised with
    mocked HTTP — including tests owned by other cards that cannot be edited
    from here.  Without this interlock those runs would append to the
    operator's real ``~/.vr180/usage_ledger.jsonl``, corrupting live budget
    accounting from a test suite (and violating the "never write the real
    home dir" rule the whole test suite is built around).

    The interlock is narrow by design: it only fires when *no* explicit path
    and *no* ``VR180_LEDGER_PATH`` were given.  A test that actually wants to
    exercise ledger writes points it at ``tmp_path`` and is unaffected.
    """
    if explicit_path is not None:
        return False
    if os.environ.get(ENV_LEDGER_PATH, "").strip():
        return False
    return "PYTEST_CURRENT_TEST" in os.environ


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def _positive_float(raw: object, source: str) -> float | None:
    """Parse *raw* into a positive float, or return None with a warning."""
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        log.warning("%s 的值 %r 不是数字，已忽略。", source, raw)
        return None
    if value <= 0:
        log.warning("%s 必须为正数，收到 %r，已忽略。", source, raw)
        return None
    return value


def price_per_mtoken(override: float | str | None = None) -> float | None:
    """Unit price in 元 per million tokens, or ``None`` when unconfigured.

    There is **no** built-in default: guessing a price would produce
    authoritative-looking numbers that are silently wrong.  Callers must
    handle ``None`` by reporting tokens only.
    """
    raw = override if override is not None else os.environ.get(ENV_PRICE_PER_MTOKEN)
    return _positive_float(raw, ENV_PRICE_PER_MTOKEN)


def estimate_cost(tokens: float | None, price: float | None) -> float | None:
    """Convert a token count to 元, or ``None`` when the price is unknown."""
    if tokens is None or price is None:
        return None
    return float(tokens) / _TOKENS_PER_MTOKEN * price


# ---------------------------------------------------------------------------
# Reading the ledger
# ---------------------------------------------------------------------------


def read_records(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
    """Return every ledger record, tolerating a missing or damaged file.

    A ledger is bookkeeping, not a database: one corrupt line (half-written
    during a power cut) must not make the whole budget gate unusable, so bad
    lines are logged and skipped rather than raised.
    """
    target = ledger_path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        log.warning("读取用量账本失败（%s）：%s", target, exc)
        return []

    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log.warning("用量账本第 %d 行不是合法 JSON，已跳过：%s", lineno, target)
            continue
        if isinstance(entry, dict):
            records.append(entry)
    return records


def record_tokens(record: dict[str, Any]) -> int:
    """Billable token count of one record (``total_tokens``, else completion)."""
    for key in ("total_tokens", "completion_tokens"):
        try:
            value = int(record.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value:
            return value
    return 0


def total_tokens(records: Iterable[dict[str, Any]]) -> int:
    """Cumulative billable tokens across *records*."""
    return sum(record_tokens(r) for r in records)


def estimate_next_tokens(
    records: Sequence[dict[str, Any]],
    *,
    model: str | None = None,
    resolution: str | None = None,
    duration: object = None,
) -> float | None:
    """Predict this run's token usage from history, or ``None`` with no history.

    Ark does not quote a price up front, so "本次预估" can only come from what
    comparable past runs actually cost.  Progressively looser matches are
    tried (model+resolution+duration → resolution+duration → resolution →
    everything) so the first generation at a new tier still gets a rough
    number instead of nothing.
    """
    if not records:
        return None

    wanted: dict[str, object] = {"model": model, "resolution": resolution, "duration": duration}
    for keys in (("model", "resolution", "duration"), ("resolution", "duration"), ("resolution",), ()):
        if any(wanted[k] is None for k in keys):
            continue
        subset = [r for r in records if all(str(r.get(k)) == str(wanted[k]) for k in keys)]
        if subset:
            return total_tokens(subset) / len(subset)
    return None


# ---------------------------------------------------------------------------
# Budget status
# ---------------------------------------------------------------------------


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "未知"
    if unit == _UNIT_YUAN:
        return f"{value:.2f} 元"
    return f"{round(value):,} tokens"


@dataclass(frozen=True)
class BudgetStatus:
    """A snapshot of cumulative usage against the configured cap."""

    mode: str
    used: float
    cap: float | None
    estimate: float | None
    unit: str
    path: Path
    used_tokens: int
    price: float | None
    note: str | None = None

    @property
    def ratio(self) -> float:
        """Fraction of the cap already consumed (0.0 when no cap is set)."""
        if not self.cap:
            return 0.0
        return self.used / self.cap

    @property
    def warn(self) -> bool:
        """True at >= 80% of the cap but below the blocking threshold."""
        return self.cap is not None and WARN_RATIO <= self.ratio < BLOCK_RATIO

    @property
    def blocked(self) -> bool:
        """True at >= 100% of the cap: submission must be refused."""
        return self.cap is not None and self.ratio >= BLOCK_RATIO

    def headline(self) -> str:
        """The one line every message shares: 已用 / 上限 / 本次预估."""
        estimate = _fmt(self.estimate, self.unit) if self.estimate is not None else "未知（账本无可比历史记录）"
        return (
            f"已用 {_fmt(self.used, self.unit)} / 上限 {_fmt(self.cap, self.unit)}"
            f"（{self.ratio * 100:.1f}%），本次预估 {estimate}"
        )

    def warn_message(self) -> str:
        return f"⚠️  预算软闸：{self.headline()}。账本 {self.path}"

    def block_message(self) -> str:
        cap_hint = (
            f"提高上限：--budget-cap <元> 或 {ENV_BUDGET_CAP}=<元>"
            if self.mode == MODE_AMOUNT
            else f"提高上限：{ENV_BUDGET_CAP_TOKENS}=<tokens>"
        )
        return (
            f"❌ 预算软闸拦截：{self.headline()}。\n"
            f"   本次生成已在**提交前**拒绝，未消耗任何额度。\n"
            f"   {cap_hint}\n"
            f"   查看账本：python -m integrations.usage_ledger --summary\n"
            f"   清空账本：删除 {self.path}"
        )


def check_budget(
    *,
    cap: float | str | None = None,
    cap_tokens: float | str | None = None,
    model: str | None = None,
    resolution: str | None = None,
    duration: object = None,
    path: str | os.PathLike[str] | None = None,
    price: float | str | None = None,
) -> BudgetStatus:
    """Compute cumulative usage against the cap. Pure — logs nothing, raises nothing.

    Mode selection:

    * a 元 cap **and** a configured unit price → amount mode;
    * otherwise a token cap → token mode (this is the documented fallback when
      the price is unknown, so the gate still works);
    * otherwise off — usage is still reported, nothing is ever blocked.
    """
    records = read_records(path)
    used_tokens = total_tokens(records)
    estimated_tokens = estimate_next_tokens(records, model=model, resolution=resolution, duration=duration)
    target = ledger_path(path)

    unit_price = price_per_mtoken(price)
    amount_cap = _positive_float(cap if cap is not None else os.environ.get(ENV_BUDGET_CAP), ENV_BUDGET_CAP)
    token_cap = _positive_float(
        cap_tokens if cap_tokens is not None else os.environ.get(ENV_BUDGET_CAP_TOKENS),
        ENV_BUDGET_CAP_TOKENS,
    )

    if amount_cap is not None and unit_price is not None:
        return BudgetStatus(
            mode=MODE_AMOUNT,
            used=estimate_cost(used_tokens, unit_price) or 0.0,
            cap=amount_cap,
            estimate=estimate_cost(estimated_tokens, unit_price),
            unit=_UNIT_YUAN,
            path=target,
            used_tokens=used_tokens,
            price=unit_price,
        )

    note = None
    if amount_cap is not None and unit_price is None:
        note = (
            f"未配置单价（{ENV_PRICE_PER_MTOKEN}），无法估算金额，"
            f"金额上限 {amount_cap} 元无法生效；闸门退回 token 模式（{ENV_BUDGET_CAP_TOKENS}）。"
        )

    return BudgetStatus(
        mode=MODE_TOKENS if token_cap is not None else MODE_OFF,
        used=float(used_tokens),
        cap=token_cap,
        estimate=estimated_tokens,
        unit=_UNIT_TOKENS,
        path=target,
        used_tokens=used_tokens,
        price=unit_price,
        note=note,
    )


def enforce_budget(
    *,
    cap: float | str | None = None,
    cap_tokens: float | str | None = None,
    model: str | None = None,
    resolution: str | None = None,
    duration: object = None,
    path: str | os.PathLike[str] | None = None,
    price: float | str | None = None,
) -> BudgetStatus:
    """Warn at 80%, refuse at 100%. **Call this before spending anything.**

    Raises
    ------
    BudgetExceededError
        When cumulative usage has reached the configured cap.  Callers must
        let this propagate past any submit call — the value of this function
        is entirely in *not* having sent the request yet.
    """
    status = check_budget(
        cap=cap,
        cap_tokens=cap_tokens,
        model=model,
        resolution=resolution,
        duration=duration,
        path=path,
        price=price,
    )
    if status.note:
        log.warning("%s", status.note)
    if status.blocked:
        message = status.block_message()
        log.error("%s", message)
        raise BudgetExceededError(message)
    if status.warn:
        log.warning("%s", status.warn_message())
    return status


# ---------------------------------------------------------------------------
# Writing the ledger
# ---------------------------------------------------------------------------


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_generation(
    *,
    provider: str,
    usage: dict[str, Any] | None = None,
    task_id: str | None = None,
    model: object = None,
    resolution: object = None,
    duration: object = None,
    path: str | os.PathLike[str] | None = None,
    price: float | str | None = None,
) -> dict[str, Any] | None:
    """Append one usage record; return it, or ``None`` if nothing was written.

    **Never raises.**  A full disk, a read-only path or a permission error
    must not turn a paid, already-completed generation into a failure — the
    money is spent either way, and the video is the thing worth keeping.  All
    failure modes degrade to a WARNING and ``None``.
    """
    try:
        return _record_generation(
            provider=provider,
            usage=usage,
            task_id=task_id,
            model=model,
            resolution=resolution,
            duration=duration,
            path=path,
            price=price,
        )
    except Exception as exc:  # bookkeeping must never break a paid generation
        log.warning("用量账本写入失败（已忽略，本次生成结果不受影响）：%s", exc)
        return None


def _record_generation(
    *,
    provider: str,
    usage: dict[str, Any] | None,
    task_id: str | None,
    model: object,
    resolution: object,
    duration: object,
    path: str | os.PathLike[str] | None,
    price: float | str | None,
) -> dict[str, Any] | None:
    if _writes_blocked(path):
        log.debug("跳过用量记账：测试环境未指定 %s，拒绝写入真实账本。", ENV_LEDGER_PATH)
        return None

    usage = usage or {}
    completion = _as_int(usage.get("completion_tokens"))
    total = _as_int(usage.get("total_tokens")) or completion

    target = ledger_path(path)
    existing = read_records(target)

    # A timed-out task is recovered with --resume-task, which polls the *same*
    # task and sees the *same* usage block.  Without this guard the recovery
    # path would double-charge the budget for one generation (issue #325/#326).
    if task_id and any(str(r.get("task_id")) == str(task_id) for r in existing):
        log.info("用量账本已有 task_id=%s 的记录，跳过重复记账。", task_id)
        return None

    unit_price = price_per_mtoken(price)
    if unit_price is None:
        log.warning(
            "未配置单价（%s），本次只记录 token 用量，无法估算金额。参考方舟官方定价页填入 元/百万token。",
            ENV_PRICE_PER_MTOKEN,
        )

    cumulative_tokens = total_tokens(existing) + total
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "provider": provider,
        "model": None if model is None else str(model),
        "resolution": None if resolution is None else str(resolution),
        "duration": duration,
        "task_id": task_id,
        "completion_tokens": completion,
        "total_tokens": total,
        "price_per_mtoken": unit_price,
        "estimated_cost": estimate_cost(total, unit_price),
        "cumulative_tokens": cumulative_tokens,
        "cumulative_cost": estimate_cost(cumulative_tokens, unit_price),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if unit_price is None:
        log.info(
            "用量已记账：%s tokens（累计 %s tokens，未配置单价）→ %s",
            f"{total:,}",
            f"{cumulative_tokens:,}",
            target,
        )
    else:
        log.info(
            "用量已记账：%s tokens ≈ %.2f 元（累计 %s tokens ≈ %.2f 元）→ %s",
            f"{total:,}",
            record["estimated_cost"],
            f"{cumulative_tokens:,}",
            record["cumulative_cost"],
            target,
        )
    return record


# ---------------------------------------------------------------------------
# Read-only summary CLI
# ---------------------------------------------------------------------------


def summary_text(path: str | os.PathLike[str] | None = None, limit: int = 10) -> str:
    """Render cumulative usage plus the last *limit* records as plain text."""
    target = ledger_path(path)
    records = read_records(target)
    unit_price = price_per_mtoken()
    cumulative = total_tokens(records)

    lines = [
        "VR180 生成用量账本",
        f"  账本路径 : {target}",
        f"  记录条数 : {len(records)}",
        f"  累计 token: {cumulative:,}",
    ]
    if unit_price is None:
        lines.append(f"  累计金额 : 未配置单价（{ENV_PRICE_PER_MTOKEN}），无法估算金额")
    else:
        lines.append(f"  单价     : {unit_price:g} 元/百万token（{ENV_PRICE_PER_MTOKEN}）")
        lines.append(f"  累计金额 : 约 {estimate_cost(cumulative, unit_price):.2f} 元")

    status = check_budget(path=path)
    if status.cap is None:
        lines.append(f"  预算上限 : 未配置（{ENV_BUDGET_CAP} / {ENV_BUDGET_CAP_TOKENS}）")
    else:
        lines.append(f"  预算上限 : {_fmt(status.cap, status.unit)}（已用 {status.ratio * 100:.1f}%）")

    if not records:
        lines.append("")
        lines.append("  （账本为空，尚无生成记录）")
        return "\n".join(lines)

    shown = records[-limit:] if limit > 0 else records
    lines.append("")
    lines.append(f"最近 {len(shown)} 条记录：")
    for entry in shown:
        cost = entry.get("estimated_cost")
        cost_text = "金额未知" if cost is None else f"≈{float(cost):.2f}元"
        lines.append(
            "  {ts}  {provider}/{model}  {res} {dur}s  {tokens:,} tokens  {cost}  {task}".format(
                ts=entry.get("timestamp", "?"),
                provider=entry.get("provider", "?"),
                model=entry.get("model") or "?",
                res=entry.get("resolution") or "?",
                dur=entry.get("duration", "?"),
                tokens=record_tokens(entry),
                cost=cost_text,
                task=entry.get("task_id") or "-",
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m integrations.usage_ledger",
        description="生成用量账本（只读查询）。默认打印累计用量与最近 10 条记录。",
        epilog=(
            "环境变量：\n"
            f"  {ENV_LEDGER_PATH}            账本路径（默认 ~/.vr180/usage_ledger.jsonl）\n"
            f"  {ENV_PRICE_PER_MTOKEN}  单价，元/百万token（未配置则只记 token）\n"
            f"  {ENV_BUDGET_CAP}            预算上限（元）\n"
            f"  {ENV_BUDGET_CAP_TOKENS}     预算上限（token，未配置单价时使用）\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="打印累计用量与最近若干条记录（默认行为，写出来更明确）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="最近 N 条记录（默认 10；<=0 表示全部）。",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        metavar="FILE",
        help=f"账本路径，覆盖 {ENV_LEDGER_PATH} 与默认位置。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m integrations.usage_ledger``. Read-only."""
    args = build_parser().parse_args(argv)
    print(summary_text(path=args.path, limit=args.limit))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    sys.exit(main())

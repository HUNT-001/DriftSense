"""Trace sink for the agent: one :class:`driftsense.contracts.TraceEvent` per line.

The trace is the Agentic-Vision evidence. Every step the runner takes appends a
``TraceEvent`` here; the file convention is ``traces/<run_id>.jsonl`` (one JSON
object per line), which the service uploads to S3 under the key
``traces/<run_id>.jsonl`` (see the PR handoff). The writer never mutates a
contract model -- it only constructs them -- so the schema stays owned by
``contracts.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftsense.contracts import TraceEvent


class TraceWriter:
    """Append-only sink that builds and records :class:`TraceEvent` lines.

    Parameters
    ----------
    run_id : stable id for the episode; also names the ``traces/<run_id>.jsonl``.
    path   : file to append JSONL to; ``None`` keeps the trace in memory only.
    """

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path is not None else None
        self._events: list[TraceEvent] = []
        self._step = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate any stale trace for this run so a rerun is clean.
            self.path.write_text("", encoding="utf-8")

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    @property
    def next_step(self) -> int:
        return self._step

    def emit(self, kind: str, **fields: Any) -> TraceEvent:
        """Record one event, auto-assigning the monotonic ``step`` and ``ts``."""
        ev = TraceEvent(
            run_id=self.run_id,
            step=self._step,
            ts=datetime.now(UTC),
            kind=kind,  # type: ignore[arg-type]  # validated by the contract model
            **fields,
        )
        self._step += 1
        self._events.append(ev)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(ev.model_dump_json() + "\n")
        return ev


def read_trace(path: str | Path) -> list[TraceEvent]:
    """Read a ``traces/<run_id>.jsonl`` file back into contract events."""
    out: list[TraceEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(TraceEvent.model_validate_json(line))
    return out

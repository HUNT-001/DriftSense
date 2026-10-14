from __future__ import annotations
import importlib, os
from driftsense.contracts import Budget, LocalizeResult, Mode

_ENGINES = {
    "fake": "driftsense.engines.fake:localize",
    "v1": "driftsense.engines.v1_adapter:localize",
    "v2": "driftsense.v2.engine:localize",
}
DEFAULT_ENGINE = "v1"   # flipped to "v2" by A in int/g2-default-v2 after g2-pass


def available_engines() -> list[str]:
    out = []
    for name, target in _ENGINES.items():
        try:
            importlib.import_module(target.split(":")[0]); out.append(name)
        except Exception:
            pass
    return out


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode = Mode.AUTO,
             budget: Budget | None = None, *, engine: str | None = None) -> LocalizeResult:
    name = engine or os.environ.get("DRIFTSENSE_ENGINE", DEFAULT_ENGINE)
    if name not in _ENGINES:
        raise ValueError(f"unknown engine {name!r}; choose from {sorted(_ENGINES)}")
    mod, fn = _ENGINES[name].split(":")
    return getattr(importlib.import_module(mod), fn)(template_bytes, search_bytes, mode, budget or Budget())

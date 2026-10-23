#!/usr/bin/env python3
"""Render docs/templates/*.md.j2 -> docs/*.md with every number pulled from results/**/*.json.

Usage:
  python scripts/render_docs.py [--results-dir results] [--templates docs/templates]
                                [--out docs] [--include-readme] [--readme README.md]
                                [--check] [--allow-missing] [--strict]

Numbers enter templates only through m(), n() and meta(); each emits an invisible
provenance marker <!--src:results/<key>.json#<metric> ...--> that scripts/check_claims.py
verifies. Template syntax: docs/TEMPLATES.md.

Exit codes: 0 ok; 1 render error, invalid results file, stale output (--check) or,
with --strict, a template still waiting for results.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _docs_common import (
    CANONICAL_TEMPLATES,
    HEADER_TEMPLATE,
    Q_FORMATS,
    TEMPLATE_SUFFIX,
    DocsError,
    MissingSource,
    Ref,
    ci_verdict_of,
    dataset_of,
    dev_sealed_violations,
    is_generated,
    key_to_path,
    load_results,
    parse_jsonpath,
    read_text,
    render_any,
    resolve_fmt,
    resolve_jsonpath,
    source_exists,
    utf8_stdio,
)

REQUIRES_RE = re.compile(r"\{#-?\s*requires:(?P<keys>[^#]*?)-?#\}", re.DOTALL)
README_TEMPLATE = "README" + TEMPLATE_SUFFIX
SUBMISSION_TEMPLATE = "SUBMISSION" + TEMPLATE_SUFFIX
# Templates that render to the repo root (next to --readme) instead of --out. They
# are rendered only with --include-readme, or when the target already carries the
# GENERATED header, so a legacy hand-written file is never clobbered by accident.
ROOT_TEMPLATES = (README_TEMPLATE, SUBMISSION_TEMPLATE)

# ---- A-grammar placeholders:  {{ <relpath>.json | <jsonpath> | <fmt?> }}
# (Person A's REPORT.md / PRIOR_WORK.md convention). A pre-pass rewrites each one
# into {{ q("<relpath>.json", "<jsonpath>", "<fmt>") }} before Jinja parses the
# template, so a template may mix this grammar with m()/n()/meta().
A_PLACEHOLDER = re.compile(
    r"(?P<lt>`?)\{\{\s*(?P<rel>[A-Za-z0-9_][A-Za-z0-9_./-]*\.json)\s*\|"
    r"\s*(?P<path>[^|{}\n]+?)\s*(?:\|\s*(?P<fmt>[A-Za-z0-9_]+)\s*)?\}\}(?P<rt>`?)"
)
# Spans the pre-pass never rewrites: Jinja comments, raw blocks, HTML comments.
_PROTECTED = re.compile(r"\{#.*?#\}|\{%-?\s*raw\s*-?%\}.*?\{%-?\s*endraw\s*[-+]?%\}|<!--.*?-->", re.DOTALL)


def _rewrite_placeholders(seg: str, name: str, line0: int) -> str:
    out, pos = [], 0
    for mt in A_PLACEHOLDER.finditer(seg):
        line_start = seg.rfind("\n", 0, mt.start()) + 1
        before = seg[line_start : mt.start()]
        code = bool(mt.group("lt")) and bool(mt.group("rt"))
        ticks_before = before.count("`") + (0 if code else len(mt.group("lt")))
        if ticks_before % 2:
            line = line0 + seg.count("\n", 0, mt.start())
            raise DocsError(
                f"{CANONICAL_TEMPLATES}/{name}:{line}: a placeholder inside inline code "
                "must be the whole code span: `{{ x.json | path }}`"
            )
        args = [json.dumps(mt.group("rel")), json.dumps(mt.group("path").strip())]
        args.append(json.dumps(mt.group("fmt")) if mt.group("fmt") else "none")
        call = "{{ q(" + ", ".join(args) + (", code=true" if code else "") + ") }}"
        if not code:
            call = mt.group("lt") + call + mt.group("rt")
        out.append(seg[pos : mt.start()] + call)
        pos = mt.end()
    out.append(seg[pos:])
    return "".join(out)


def prepass(source: str, name: str = "?") -> str:
    """Rewrite A-grammar placeholders into q() calls (line numbers are preserved).

    HTML comments are emitted verbatim: if one contains Jinja delimiters (e.g. the
    grammar description in A's header comment) it is wrapped in {% raw %}."""
    out, pos = [], 0
    for mt in _PROTECTED.finditer(source):
        out.append(_rewrite_placeholders(source[pos : mt.start()], name, source.count("\n", 0, pos) + 1))
        block = mt.group(0)
        if block.startswith("<!--") and re.search(r"\{\{|\{%|\{#", block):
            block = "{% raw %}" + block + "{% endraw +%}"
        out.append(block)
        pos = mt.end()
    out.append(_rewrite_placeholders(source[pos:], name, source.count("\n", 0, pos) + 1))
    return "".join(out)


@dataclass
class RenderResult:
    outputs: dict[Path, str] = field(default_factory=dict)  # out path -> content
    sources: dict[Path, str] = field(default_factory=dict)  # out path -> template name
    pending: dict[str, list[str]] = field(default_factory=dict)  # template -> missing keys
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _jinja():
    try:
        import jinja2
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocsError("jinja2 is required: pip install jinja2") from exc
    return jinja2


def list_templates(templates_dir: Path) -> list[str]:
    """Renderable template names (posix, relative). Names starting with '_' are partials."""
    if not templates_dir.is_dir():
        return []
    names = []
    for p in sorted(templates_dir.rglob("*" + TEMPLATE_SUFFIX)):
        rel = p.relative_to(templates_dir)
        if any(part.startswith("_") for part in rel.parts):
            continue
        names.append(rel.as_posix())
    return names


def output_path(name: str, out_dir: Path, readme_path: Path) -> Path:
    if name == README_TEMPLATE:
        return readme_path
    if name in ROOT_TEMPLATES:
        return readme_path.parent / name[: -len(".j2")]
    return out_dir / name[: -len(".j2")]


def requires_of(source: str) -> list[str]:
    keys: list[str] = []
    for m in REQUIRES_RE.finditer(source):
        keys += [k for k in re.split(r"[\s,]+", m.group("keys")) if k]
    return keys


def _caller_location() -> tuple[str | None, int | None]:
    """Template name and line of the innermost template frame calling m()/n()/meta()."""
    f = sys._getframe(1)
    while f is not None:
        t = f.f_globals.get("__jinja_template__")
        if t is not None:
            return t.name, t.get_corresponding_lineno(f.f_lineno)
        f = f.f_back
    return None, None


def _located(msg: str) -> DocsError:
    name, line = _caller_location()
    where = f"{CANONICAL_TEMPLATES}/{name}:{line}: " if name else ""
    return DocsError(where + msg)


def _exc_location(exc: BaseException, name: str) -> str:
    line = None
    for fr in traceback.extract_tb(exc.__traceback__):
        if fr.filename.replace("\\", "/").endswith(name) or fr.filename.endswith(".j2"):
            line = fr.lineno
    if line is None:
        line = getattr(exc, "lineno", None)
    return f"{CANONICAL_TEMPLATES}/{name}:{line if line is not None else '?'}"


def make_env(templates_dir: Path, results: dict[str, Any], allow_missing: bool):
    jinja2 = _jinja()

    class _Loader(jinja2.FileSystemLoader):
        def get_source(self, environment, template):
            src, filename, uptodate = super().get_source(environment, template)
            return prepass(src, template), filename, uptodate

    raw = getattr(results, "raw", {})
    env = jinja2.Environment(
        loader=_Loader(str(templates_dir)),
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def _emit(ref: Ref) -> str:
        if not source_exists(ref, results):
            if allow_missing:
                return ref.missing()
            avail = sorted(raw) if ref.q else sorted(results)
            raise _located(
                f"results key {ref.key!r} -> {ref.path} not found (available: {', '.join(avail) or 'none'})"
            )
        try:
            text = render_any(ref, results)
        except DocsError as exc:
            if allow_missing and isinstance(exc, MissingSource):
                return ref.missing()
            raise _located(str(exc)) from None
        return text + ref.marker()

    def _key(key: Any) -> str:
        if not isinstance(key, str) or not key or key.endswith(".json"):
            raise _located(f"results key must look like 'accuracy/g2', got {key!r}")
        return key.removeprefix("results/")

    def m(key, metric, fmt=None, ci=False, label=True):
        k = _key(key)
        try:
            if fmt is not None:
                resolve_fmt(str(fmt), "")
            elif k in results and metric in results[k].metrics:
                fmt = resolve_fmt(None, results[k].metrics[metric].unit)
        except DocsError as exc:
            raise _located(str(exc)) from None
        return _emit(Ref(k, str(metric), "value", fmt, bool(ci), bool(label)))

    def n(key, metric, label=True):
        return _emit(Ref(_key(key), str(metric), "n", "int", False, bool(label)))

    def meta(key, path, fmt=None):
        return _emit(Ref(_key(key), "@" + str(path), "value", fmt, False, False))

    def q(relpath, jsonpath, fmt=None, code=False):
        """A-grammar placeholder: a value at a JSON path in results/<relpath>.json
        (ResultsFile or plain JSON). Bare text, no DEV/SEALED label."""
        rel = str(relpath).removeprefix("results/")
        if not rel.endswith(".json") or rel.startswith("/") or ".." in rel.split("/"):
            raise _located(f"placeholder path must be <dir>/<name>.json, got {relpath!r}")
        try:
            parse_jsonpath(str(jsonpath))
            if fmt is not None and fmt not in Q_FORMATS:
                raise DocsError(f"unknown placeholder fmt {fmt!r}; choose from {', '.join(Q_FORMATS)}")
        except DocsError as exc:
            raise _located(str(exc)) from None
        return _emit(Ref(rel[: -len(".json")], str(jsonpath), fmt=fmt, label=False, q=True, code=bool(code)))

    # ---- logic-only helpers: they never print numbers ----

    def _metric(key, metric, what: str):
        """Return the Metric, or None when allowed to be missing."""
        k = _key(key)
        rf = results.get(k)
        if rf is None:
            if allow_missing:
                return None
            raise _located(f"{what}: results key {k!r} -> {key_to_path(k)} not found")
        mt = rf.metrics.get(str(metric))
        if mt is None:
            if allow_missing:
                return None
            raise _located(f"{what}: metric {metric!r} not in {key_to_path(k)}")
        return mt

    def has(key, metric=None):
        rf = results.get(_key(key))
        if rf is None:
            return False
        return metric is None or str(metric) in rf.metrics

    def metrics(key, prefix=""):
        rf = results.get(_key(key))
        if rf is None:
            if allow_missing:
                return []
            raise _located(f"metrics(): results key {key!r} not found")
        return sorted(name for name in rf.metrics if name.startswith(str(prefix)))

    def val(key, metric, field="value"):
        if field not in ("value", "ci_low", "ci_high", "n"):
            raise _located(f"val(): field must be value/ci_low/ci_high/n, got {field!r}")
        mt = _metric(key, metric, "val()")
        v = None if mt is None else getattr(mt, field)
        return Hidden(float("nan") if v is None else float(v), f"{key}#{metric}.{field}")

    def ci_verdict(key, metric, null=1.0):
        mt = _metric(key, metric, "ci_verdict()")
        return "no_ci" if mt is None else ci_verdict_of(mt, float(null))

    def exists(key, path=None):
        """True if results/<key>.json exists (any JSON, incl. plain documents) and,
        when given, the JSON path resolves. Never errors, never prints a number."""
        k = _key(str(key).removesuffix(".json"))
        if k not in raw:
            return False
        if path is None:
            return True
        try:
            resolve_jsonpath(raw[k], str(path), k)
        except DocsError:
            return False
        return True

    def flag(relpath, jsonpath):
        """A JSON boolean for template logic (e.g. decisions.json overall_pass).
        Returns True/False, or None for JSON null (e.g. an undecided rule);
        anything else is an error."""
        k = _key(str(relpath).removesuffix(".json"))
        if k not in raw:
            raise _located(f"flag(): results/{k}.json not found")
        try:
            v, _ = resolve_jsonpath(raw[k], str(jsonpath), f"results/{k}.json")
        except DocsError as exc:
            raise _located(str(exc)) from None
        if v is not None and not isinstance(v, bool):
            raise _located(f"flag(): {k}#{jsonpath} is not a JSON boolean or null")
        return v

    def keys(relpath, jsonpath=None):
        """Sorted keys of a JSON object (names, never values), e.g. decision rule IDs."""
        k = _key(str(relpath).removesuffix(".json"))
        if k not in raw:
            if allow_missing:
                return []
            raise _located(f"keys(): results/{k}.json not found")
        obj = raw[k]
        if jsonpath:
            try:
                obj, _ = resolve_jsonpath(obj, str(jsonpath), f"results/{k}.json")
            except DocsError as exc:
                raise _located(str(exc)) from None
        if isinstance(obj, list):
            return [str(i) for i in range(len(obj))]
        if not isinstance(obj, dict):
            raise _located(f"keys(): {k}#{jsonpath} is not a JSON object or array")
        return list(obj)

    def kind(relpath, jsonpath):
        """JSON type at a path: object, array, string, number, bool, null or
        missing. For template branching only (it never reveals a value)."""
        k = _key(str(relpath).removesuffix(".json"))
        if k not in raw:
            return "missing"
        try:
            v, _ = resolve_jsonpath(raw[k], str(jsonpath), f"results/{k}.json")
        except DocsError:
            return "missing"
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, (int, float)):
            return "number"
        if isinstance(v, str):
            return "string"
        return "array" if isinstance(v, list) else "object"

    env.globals.update(
        m=m,
        n=n,
        meta=meta,
        q=q,
        has=has,
        exists=exists,
        flag=flag,
        keys=keys,
        kind=kind,
        metrics=metrics,
        val=val,
        ci_verdict=ci_verdict,
    )
    return env


class Hidden:
    """A results number usable in template logic (comparisons, truthiness) that
    refuses to be printed, so every printed number still goes through m()/n()."""

    __slots__ = ("_v", "_what")
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, v: float, what: str) -> None:
        self._v, self._what = v, what

    def _o(self, other: Any) -> float:
        return other._v if isinstance(other, Hidden) else float(other)

    def __lt__(self, other):
        return self._v < self._o(other)

    def __le__(self, other):
        return self._v <= self._o(other)

    def __gt__(self, other):
        return self._v > self._o(other)

    def __ge__(self, other):
        return self._v >= self._o(other)

    def __eq__(self, other):
        return self._v == self._o(other)

    def __ne__(self, other):
        return self._v != self._o(other)

    def __bool__(self) -> bool:
        return self._v == self._v and self._v != 0  # NaN (missing) is falsy

    def __str__(self) -> str:
        raise _located(
            f"val({self._what}) cannot be printed; use m()/n() so the number carries a provenance marker"
        )

    def __repr__(self) -> str:
        return f"<hidden {self._what}>"

    def __format__(self, spec: str) -> str:
        return str(self)


def render_all(
    templates_dir: Path,
    results: dict[str, Any],
    out_dir: Path,
    readme_path: Path,
    *,
    include_readme: bool = False,
    allow_missing: bool = False,
) -> RenderResult:
    res = RenderResult()
    names = list_templates(templates_dir)
    if not names:
        return res
    jinja2 = _jinja()
    env = make_env(templates_dir, results, allow_missing)

    def ds_of(path: str) -> str | None:
        return dataset_of(results, path)

    for name in names:
        if name in ROOT_TEMPLATES and not include_readme:
            target = output_path(name, out_dir, readme_path)
            if not (target.is_file() and is_generated(read_text(target))):
                res.skipped.append(name)
                continue
        source = read_text(templates_dir / name)
        raw_keys = getattr(results, "raw", {})
        missing = [k for k in requires_of(source) if k not in results and k not in raw_keys]
        if missing and not allow_missing:
            res.pending[name] = [key_to_path(k) for k in missing]
            continue
        try:
            body = env.get_template(name).render()
        except DocsError as exc:
            res.errors.append(str(exc))
            continue
        except jinja2.TemplateSyntaxError as exc:
            res.errors.append(f"{CANONICAL_TEMPLATES}/{name}:{exc.lineno}: syntax error: {exc.message}")
            continue
        except jinja2.UndefinedError as exc:
            res.errors.append(f"{_exc_location(exc, name)}: undefined: {exc.message}")
            continue
        except Exception as exc:  # noqa: BLE001 - report any template failure, keep going
            res.errors.append(f"{_exc_location(exc, name)}: {type(exc).__name__}: {exc}")
            continue
        header = HEADER_TEMPLATE.format(src=f"{CANONICAL_TEMPLATES}/{name}")
        content = header + "\n" + body.replace("\r\n", "\n").rstrip("\n") + "\n"
        bad = dev_sealed_violations(content, ds_of)
        if bad:
            for line, msg in bad:
                res.errors.append(
                    f"{CANONICAL_TEMPLATES}/{name} (rendered line {line}): {msg}; "
                    "keep DEV and SEALED numbers in separate sentences"
                )
            continue
        out = output_path(name, out_dir, readme_path)
        res.outputs[out] = content
        res.sources[out] = name
    return res


def find_generated(out_dir: Path, readme_path: Path, templates_dir: Path) -> list[Path]:
    """Existing files carrying the GENERATED header."""
    found = []
    if out_dir.is_dir():
        tdir = templates_dir.resolve()
        for p in sorted(out_dir.rglob("*.md")):
            if tdir in p.resolve().parents:
                continue
            if is_generated(read_text(p)):
                found.append(p)
    for name in ROOT_TEMPLATES:
        p = output_path(name, out_dir, readme_path)
        if p.is_file() and is_generated(read_text(p)) and p not in found:
            found.append(p)
    return found


def stale_report(
    res: RenderResult, out_dir: Path, readme_path: Path, templates_dir: Path
) -> list[tuple[Path, str]]:
    """(path, reason) for every output that differs from disk, is missing, or is orphaned."""
    problems: list[tuple[Path, str]] = []
    for out, content in sorted(res.outputs.items()):
        if not out.is_file():
            problems.append((out, "not rendered yet (run `make docs`)"))
            continue
        disk = read_text(out)
        if disk != content:
            diff = difflib.unified_diff(
                disk.splitlines(),
                content.splitlines(),
                "on disk",
                "rendered",
                lineterm="",
                n=0,
            )
            snippet = "\n".join(list(diff)[2:8])
            problems.append((out, "stale: differs from a fresh render\n" + snippet))
    pending_outputs = {output_path(name, out_dir, readme_path).resolve(): name for name in res.pending}
    expected = {p.resolve() for p in res.outputs}
    for p in find_generated(out_dir, readme_path, templates_dir):
        rp = p.resolve()
        if rp in expected:
            continue
        if rp in pending_outputs:
            problems.append(
                (
                    p,
                    f"stale: template {pending_outputs[rp]} is waiting for results that no longer exist",
                )
            )
        elif any(p == output_path(nm, out_dir, readme_path) for nm in res.skipped):
            problems.append((p, f"{p.name} is generated: re-run with --include-readme"))
        else:
            problems.append((p, "orphan: carries the GENERATED header but no template produces it"))
    return problems


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--templates", default="docs/templates", type=Path)
    ap.add_argument("--out", default="docs", type=Path)
    ap.add_argument("--readme", default="README.md", type=Path, help="README output path")
    ap.add_argument(
        "--include-readme",
        action="store_true",
        help="also render the repo-root templates (README.md.j2, SUBMISSION.md.j2) "
        "over hand-written files; already-generated root files are always re-rendered",
    )
    ap.add_argument("--check", action="store_true", help="render in memory; exit 1 if disk differs")
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="render ⟨missing: ...⟩ placeholders instead of failing (drafts only)",
    )
    ap.add_argument("--strict", action="store_true", help="templates waiting for results are errors")
    a = ap.parse_args(argv)

    try:
        results, load_errors = load_results(a.results_dir)
        if load_errors:
            for e in load_errors:
                print(f"error: {e}", file=sys.stderr)
            return 1
        if not results:
            print(
                f"note: no results files under {a.results_dir.as_posix()}/",
                file=sys.stderr,
            )
        res = render_all(
            a.templates,
            results,
            a.out,
            a.readme,
            include_readme=a.include_readme,
            allow_missing=a.allow_missing,
        )
    except DocsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for e in res.errors:
        print(f"error: {e}", file=sys.stderr)
    for name, missing in sorted(res.pending.items()):
        print(
            f"PENDING {CANONICAL_TEMPLATES}/{name}: waiting for {', '.join(missing)}",
            file=sys.stderr,
        )
    for name in res.skipped:
        print(
            f"skipped {CANONICAL_TEMPLATES}/{name} (pass --include-readme)",
            file=sys.stderr,
        )
    if res.errors:
        return 1
    if a.allow_missing:
        print(
            "warning: --allow-missing output is a draft; check_claims will fail on it",
            file=sys.stderr,
        )

    rc = 1 if (a.strict and res.pending) else 0
    if a.check:
        problems = stale_report(res, a.out, a.readme, a.templates)
        for path, why in problems:
            print(f"{path.as_posix()}: {why}", file=sys.stderr)
        if problems:
            return 1
        print(f"docs up to date ({len(res.outputs)} rendered, {len(res.pending)} pending)")
        return rc

    for out, content in sorted(res.outputs.items()):
        if out.is_file() and read_text(out) == content:
            print(f"unchanged {out.as_posix()}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        print(f"wrote {out.as_posix()}")
    for path, why in stale_report(res, a.out, a.readme, a.templates):
        if "orphan" in why or "waiting" in why:
            print(f"warning: {path.as_posix()}: {why}", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())

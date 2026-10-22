#!/usr/bin/env python3
"""Fail if any number in the generated docs is not traceable to results/**/*.json.

Scans docs/**/*.md (except docs/templates/) plus README.md and SUBMISSION.md.
Files whose first line is the render_docs GENERATED header are checked strictly
(findings are errors); every other markdown file is "legacy" and only gets
warnings unless --strict-legacy is passed. A non-generated file whose first line is
`<!-- check_claims: skip (reason) -->` (e.g. docs/TEMPLATES.md) is not scanned.

Codes (rules are spelled out in docs/TEMPLATES.md, "What check_claims enforces"):
  CL001 unmarked numeric claim          CL007 TODO / TBD / XX% left in the text
  CL002 displayed value != JSON value   CL008 stale or orphaned render
  CL003 marker source missing/invalid   CL009 invalid results file
  CL004 banned label recall@60          CL010 marker inside a code block
  CL005 'sealed' next to a DEV number   CL011 template waiting for results (warning)
  CL006 ⟨missing: ...⟩ placeholder       CL012 template syntax / render error

Output: `path:line: CODE message` (warnings say `CODE warning: message`) and a summary.
Exit 1 on any error (or any warning with --strict-legacy); 0 otherwise.
"""

from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _docs_common import (
    COMMENT_RE,
    MARKER_RE,
    DocsError,
    MissingSource,
    dataset_of,
    dev_sealed_violations,
    dumps_json,
    fenced_lines,
    is_generated,
    load_results,
    parse_marker,
    read_text,
    render_any,
    source_exists,
    utf8_stdio,
)

# --------------------------------------------------------------------------- rules

SKIP_RE = re.compile(r"<!-- check_claims: skip\b")
BANNED_RECALL = re.compile(r"recall@60", re.IGNORECASE)
TODO_RE = re.compile(r"\bTODO\b|\bTBD\b|\bXX(?:\.X+)?\s?%")
MISSING_RE = re.compile(r"⟨missing:[^⟩\n]*⟩?")

INLINE_CODE = re.compile(r"(`+)(?:(?!\1).)+?\1")
URL_RE = re.compile(r"\bhttps?://[^\s)>\]]+")
LINK_TARGET = re.compile(r"\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
HEADING_NUM = re.compile(r"^(\s{0,3}#{1,6}\s+)((?:\d+\.)*\d+\.?)(?=\s)")
LIST_NUM = re.compile(r"^(\s*(?:>\s*)?)(\d+[.)])(?=\s)")

CITATION = re.compile(r"(?:\\?\[\d+\\?\])+[.,;:]?")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?|\d{4}/\d{2}/\d{2}")
TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?(?:[–-]\d{1,2}:\d{2}(?::\d{2})?)?")
YEAR = re.compile(r"(?:19|20)\d{2}")
VERSION = re.compile(r"v?\d+(?:\.\d+){2,}|v\d+(?:\.\d+)*", re.IGNORECASE)
NUMERIC = re.compile(
    r"(?P<pre>[-+−±~≈<>≤≥]*\$?)"
    r"(?P<num>\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?(?:[–-]\$?\d[\d,]*(?:\.\d+)?)?)"
    r"(?P<suf>%|×|x|ms|s|nm|px|µs|us|ns|k|gb|mb|kb|h|min|fps)?",
    re.IGNORECASE,
)
PRODUCTS = {
    "opencv",
    "python",
    "py",
    "cool",
    "node",
    "node.js",
    "nodejs",
    "cuda",
    "ubuntu",
    "numpy",
    "pydantic",
    "jinja2",
    "jinja",
    "cdk",
    "kleidicv",
    "fastapi",
    "pandoc",
    "ruff",
    "pytest",
    "version",
    "release",
    "gcc",
    "debian",
    "linux",
    "graviton",
    "tls",
    "http",
    "mermaid",
    "java",
    "go",
    "rust",
    "aws",
    "industry",
    "web",
}
CONTEXT_WORDS = {
    "prompt",
    "section",
    "sec",
    "step",
    "phase",
    "gate",
    "stage",
    "figure",
    "fig",
    "table",
    "appendix",
    "day",
    "rule",
    "level",
    "item",
    "issue",
    "pr",
    "part",
    "chapter",
    "task",
    "line",
    "page",
    "row",
    "column",
    "priority",
    "week",
    "§",
}
MONTHS = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
UNIT_WORDS = {
    "ms",
    "s",
    "sec",
    "secs",
    "second",
    "seconds",
    "nm",
    "px",
    "pixel",
    "pixels",
    "µs",
    "us",
    "ns",
    "min",
    "mins",
    "minute",
    "minutes",
    "h",
    "hr",
    "hrs",
    "hour",
    "hours",
    "day",
    "days",
    "gb",
    "mb",
    "kb",
    "tb",
    "%",
    "×",
    "x",
    "fps",
    "usd",
    "mhz",
    "ghz",
}
_STRIP = "*_~\"'()[]{}<>|,;:!?\\"


@dataclass
class Finding:
    path: str
    line: int
    code: str
    severity: str  # "error" | "warning"
    message: str

    def render(self) -> str:
        sev = "" if self.severity == "error" else "warning: "
        return f"{self.path}:{self.line}: {self.code} {sev}{self.message}"


def _clean(tok: str) -> str:
    t = tok.strip(_STRIP)
    while t.endswith(".") and not t.endswith(".."):
        t = t[:-1].strip(_STRIP)
    return t


def is_claim(raw: str, prev: str, nxt: str) -> bool:
    """Decide whether one whitespace-delimited token is a measured numeric claim."""
    if CITATION.fullmatch(raw.strip()):
        return False
    t = _clean(raw)
    if not re.search(r"\d", t):
        return False
    if t.startswith("§") or (t.startswith("#") and t[1:].isdigit()):
        return False
    if t == "95%" and nxt == "ci":
        return False  # the interval level ("95% CI"), fixed by the protocol: not a result
    if DATE.fullmatch(t) or TIME.fullmatch(t) or VERSION.fullmatch(t):
        return False
    if YEAR.fullmatch(t) and nxt not in UNIT_WORDS:
        return False
    if (prev in PRODUCTS or prev in CONTEXT_WORDS) and re.fullmatch(r"\d+(?:\.\d+)*\+?", t):
        return False  # OpenCV 5, Python 3.12, Python 3.10+, Prompt 15, Industry 4.0
    m = NUMERIC.fullmatch(t)
    if not m:
        if re.search(r"[A-Za-z]", t):  # identifier: c8g.large, S0–S9, p95, x86, top-1
            return False
        return bool(re.search(r"\d[.,]\d|\d\d|[%×$]", t))
    num = m.group("num")
    plain = not m.group("suf") and "$" not in m.group("pre")
    if plain and (prev in MONTHS or nxt in MONTHS) and re.fullmatch(r"\d{1,2}(?:[–-]\d{1,2})?", num):
        return False
    if not plain or nxt in UNIT_WORDS:
        return True
    if "." in num or "e" in num.lower():
        return True
    return re.search(r"\d\d", num.replace(",", "")) is not None


def claim_tokens(line: str) -> list[tuple[int, str]]:
    """Unmarked claim tokens on one already-masked, non-code line."""

    def blank(m: re.Match) -> str:
        return " " * len(m.group(0))

    s = INLINE_CODE.sub(blank, line)
    s = URL_RE.sub(blank, s)
    s = LINK_TARGET.sub(lambda m: "]" + " " * (len(m.group(0)) - 1), s)
    s = HTML_TAG.sub(blank, s)
    s = HEADING_NUM.sub(lambda m: m.group(1) + " " * len(m.group(2)), s)
    s = LIST_NUM.sub(lambda m: m.group(1) + " " * len(m.group(2)), s)
    toks = [(m.start(), m.group(0)) for m in re.finditer(r"\S+", s)]
    out = []
    for i, (col, raw) in enumerate(toks):
        if not re.search(r"\d", raw):
            continue
        prev = _clean(toks[i - 1][1]).lower() if i else ""
        nxt = _clean(toks[i + 1][1]).lower() if i + 1 < len(toks) else ""
        if is_claim(raw, prev, nxt):
            out.append((col, _clean(raw)))
    return out


_TAIL_CLAIM = re.compile(r"(?:[-+]?\$?\d[\d.,]*\S*(?: \(95% CI [^)]*\))?(?: \[(?:DEV|SEALED)\])?|\S+)$")


def scan_text(rel: str, text: str, results: dict[str, Any], generated: bool) -> list[Finding]:
    sev = "error" if generated else "warning"
    findings: list[Finding] = []

    def add(line: int, code: str, msg: str) -> None:
        findings.append(Finding(rel, line, code, sev, msg))

    lines = text.split("\n")
    code = fenced_lines(lines)
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln) + 1)

    def line_of(off: int) -> int:
        return bisect.bisect_right(starts, off)

    # (c) banned labels and drafting leftovers (anywhere, incl. code)
    for i, line in enumerate(lines, 1):
        visible = MARKER_RE.sub("", line)
        if BANNED_RECALL.search(visible):
            add(i, "CL004", "banned metric name 'recall@60' (use candidate_set_size)")
        if "⟨missing:" in visible:
            add(
                i,
                "CL006",
                "unresolved ⟨missing: ...⟩ placeholder (rendered with --allow-missing)",
            )
        for t in TODO_RE.finditer(visible):
            add(i, "CL007", f"unfinished text {t.group(0)!r}")

    chars = list(text)

    def blank(a: int, b: int) -> None:
        for j in range(max(a, 0), b):
            if chars[j] != "\n":
                chars[j] = " "

    # (b) every marker must resolve and match the text right before it
    for mk in MARKER_RE.finditer(text):
        ln = line_of(mk.start())
        blank(mk.start(), mk.end())
        before = text[starts[ln - 1] : mk.start()]
        tail = _TAIL_CLAIM.search(before)
        tail_text = tail.group(0) if tail else ""
        if code[ln - 1]:
            add(ln, "CL010", "provenance marker inside a code block renders literally")
            continue
        try:
            ref = parse_marker(mk)
        except DocsError as exc:
            add(ln, "CL003", f"malformed marker: {exc}")
            blank(mk.start() - len(tail_text), mk.start())
            continue
        if not source_exists(ref, results):
            add(ln, "CL003", f"marker source {ref.path} does not exist")
            blank(mk.start() - len(tail_text), mk.start())
            continue
        try:
            expected = render_any(ref, results)
        except MissingSource as exc:
            add(ln, "CL003", str(exc))
            blank(mk.start() - len(tail_text), mk.start())
            continue
        except DocsError as exc:
            add(ln, "CL002", f"{ref.path}#{ref.field}: {exc}")
            blank(mk.start() - len(tail_text), mk.start())
            continue
        k = len(before) - len(expected)
        ok = before.endswith(expected) and (
            k == 0 or not (before[k - 1].isalnum() or before[k - 1] in ".,$-+")
        )
        if ok:
            blank(mk.start() - len(expected), mk.start())
        else:
            add(
                ln,
                "CL002",
                f"displayed {tail_text!r} does not match {ref.path}#{ref.field} (expected {expected!r})",
            )
            blank(mk.start() - len(tail_text), mk.start())

    # (c) DEV numbers in a 'sealed' sentence
    def ds_of(path: str) -> str | None:
        return dataset_of(results, path)

    for ln, msg in dev_sealed_violations(text, ds_of):
        add(ln, "CL005", msg)

    # (a) unmarked digit-bearing claims
    for rx in (MISSING_RE, COMMENT_RE):
        for m in rx.finditer(text):
            blank(m.start(), m.end())
    masked = "".join(chars).split("\n")
    for i, (line, is_code) in enumerate(zip(masked, code, strict=True), 1):
        if is_code:
            continue
        for _col, tok in claim_tokens(line):
            add(
                i,
                "CL001",
                f"unmarked numeric claim {tok!r}: render it with m()/n() from results",
            )
    findings.sort(key=lambda f: (f.line, f.code))
    return findings


# --------------------------------------------------------------------------- driver


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def collect_docs(docs_dir: Path, templates_dir: Path, extra: list[Path]) -> list[Path]:
    out: list[Path] = []
    tdir = templates_dir.resolve()
    if docs_dir.is_dir():
        for p in sorted(docs_dir.rglob("*.md")):
            if tdir == p.resolve().parent or tdir in p.resolve().parents:
                continue
            out.append(p)
    seen = {p.resolve() for p in out}
    for p in extra:
        if p.is_file() and p.resolve() not in seen:
            out.append(p)
            seen.add(p.resolve())
    return out


def template_findings(templates_dir: Path, root: Path) -> list[Finding]:
    """Syntax-check every template (cheap; needs jinja2)."""
    import render_docs

    out: list[Finding] = []
    names = render_docs.list_templates(templates_dir)
    if not names:
        return out
    try:
        import jinja2
    except ImportError:
        return [
            Finding(
                _rel(templates_dir, root),
                0,
                "CL012",
                "warning",
                "jinja2 missing: templates not checked",
            )
        ]
    env = jinja2.Environment()
    for name in names:
        p = templates_dir / name
        try:
            env.parse(render_docs.prepass(read_text(p), name))
        except jinja2.TemplateSyntaxError as exc:
            out.append(
                Finding(
                    _rel(p, root),
                    exc.lineno or 0,
                    "CL012",
                    "error",
                    f"syntax error: {exc.message}",
                )
            )
        except DocsError as exc:
            out.append(Finding(_rel(p, root), 0, "CL012", "error", str(exc)))
    return out


def render_findings(
    templates_dir: Path, results: dict, docs_dir: Path, readme: Path, root: Path
) -> list[Finding]:
    import render_docs

    out: list[Finding] = []
    try:
        # root templates (README/SUBMISSION) are included iff their target is generated
        res = render_docs.render_all(templates_dir, results, docs_dir, readme)
    except DocsError as exc:
        return [Finding(_rel(templates_dir, root), 0, "CL012", "error", str(exc))]
    for e in res.errors:
        out.append(Finding(_rel(templates_dir, root), 0, "CL012", "error", e))
    for path, why in render_docs.stale_report(res, docs_dir, readme, templates_dir):
        out.append(Finding(_rel(path, root), 1, "CL008", "error", why.splitlines()[0]))
    for name, missing in sorted(res.pending.items()):
        target = render_docs.output_path(name, docs_dir, readme)
        if not target.is_file():
            out.append(
                Finding(
                    _rel(templates_dir / name, root),
                    1,
                    "CL011",
                    "warning",
                    f"not rendered yet: waiting for {', '.join(missing)}",
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root",
        default=".",
        type=Path,
        help="repo root (other paths are relative to it)",
    )
    ap.add_argument("--docs", default="docs", type=Path)
    ap.add_argument("--templates", default="docs/templates", type=Path)
    ap.add_argument("--results-dir", default="results", type=Path)
    ap.add_argument("--readme", default="README.md", type=Path)
    ap.add_argument(
        "--extra",
        nargs="*",
        default=["SUBMISSION.md"],
        type=Path,
        help="other markdown files to scan (default: SUBMISSION.md)",
    )
    ap.add_argument(
        "--no-render-check",
        action="store_true",
        help="skip the stale-render comparison",
    )
    ap.add_argument(
        "--strict-legacy",
        action="store_true",
        help="legacy (non-generated) findings fail too",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)

    root = a.root
    docs_dir, templates_dir = root / a.docs, root / a.templates
    results_dir, readme = root / a.results_dir, root / a.readme
    findings: list[Finding] = []

    try:
        results, load_errors = load_results(results_dir)
    except DocsError as exc:
        results, load_errors = {}, [str(exc)]
    for e in load_errors:
        path, _, msg = e.partition(": ")
        findings.append(Finding(_rel(Path(path), root), 0, "CL009", "error", msg or e))

    files = collect_docs(docs_dir, templates_dir, [readme] + [root / x for x in a.extra])
    n_gen = n_legacy = n_skip = 0
    for p in files:
        text = read_text(p)
        gen = is_generated(text)
        if not gen and SKIP_RE.match(text):
            n_skip += 1
            continue
        n_gen += gen
        n_legacy += not gen
        findings += scan_text(_rel(p, root), text, results, gen)

    findings += template_findings(templates_dir, root)
    note = ""
    if n_gen == 0:
        note = "no generated docs yet"
    elif not a.no_render_check:
        findings += render_findings(templates_dir, results, docs_dir, readme, root)

    if a.strict_legacy:
        for f in findings:
            f.severity = "error"
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    summary = {
        "generated_docs": n_gen,
        "legacy_docs": n_legacy,
        "skipped_docs": n_skip,
        "results_files": len(results),
        "errors": len(errors),
        "warnings": len(warnings),
        "by_code": dict(sorted(Counter(f.code for f in findings).items())),
        "note": note,
    }
    if a.json:
        print(dumps_json({"findings": [asdict(f) for f in findings], "summary": summary}))
    else:
        for f in findings:
            print(f.render())
        if note:
            print(note)
        codes = ", ".join(f"{k}={v}" for k, v in summary["by_code"].items()) or "none"
        print(
            f"check_claims: {len(errors)} error(s), {len(warnings)} warning(s) "
            f"[{codes}] in {n_gen} generated / {n_legacy} legacy / {n_skip} skipped doc(s), "
            f"{len(results)} results file(s)"
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

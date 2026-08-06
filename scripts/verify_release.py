#!/usr/bin/env python3
"""Check released aggregates, expected files, and common anonymity leaks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q0 = ROOT / "results" / "ariac" / "ariac_q0_definition_audit.csv"
EXPECTED_FIGURES = (
    "fig2_cross_domain_decoupling.pdf",
    "fig3_mechanism.pdf",
    "fig4_necessity.pdf",
    "fig5_scope2x2.pdf",
)
BLOCKED_TEXT = (
    "lkmubihei",
    "/home/lk",
    "/home/pc",
    "sia_interface_zjy",
    "lk@todo.todo",
)


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or file_digest(path) != expected:
            raise AssertionError(f"Manifest mismatch: {relative}")


def check_q0() -> None:
    with Q0.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 50 or len({row["order_id"] for row in rows}) != 50:
        raise AssertionError("Q0 table must contain 50 unique trials")
    if any(float(row["eps"]) != 0.0 for row in rows):
        raise AssertionError("Q0 table contains a nonzero epsilon")
    subgoals = sum(int(row["n_subgoals"]) for row in rows)
    passed = sum(int(row["n_pass_static_first_check"]) for row in rows)
    macro = sum(float(row["q0_static"]) for row in rows) / len(rows)
    micro = passed / subgoals
    assert (subgoals, passed) == (195, 65)
    assert math.isclose(macro, 0.3453809523809524, abs_tol=1e-12)
    assert math.isclose(micro, 1 / 3, abs_tol=1e-12)


def check_outputs() -> None:
    audit = ROOT / "paper" / "figure_scripts" / "figure_data_audit.json"
    data = json.loads(audit.read_text(encoding="utf-8"))
    if not data.get("sources") or not data.get("fig2"):
        raise AssertionError("figure_data_audit.json is incomplete")
    for name in EXPECTED_FIGURES:
        path = ROOT / "paper" / name
        if not path.is_file() or path.stat().st_size < 10_000:
            raise AssertionError(f"Missing or empty figure: {path}")


def check_anonymity() -> None:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if path.resolve() == Path(__file__).resolve():
            continue
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() in {".png", ".pdf", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        for blocked in BLOCKED_TEXT:
            if blocked in text:
                violations.append(f"{path.relative_to(ROOT)}: {blocked}")
    if violations:
        raise AssertionError("Anonymity scan failed:\n" + "\n".join(violations))


def main() -> None:
    check_manifest()
    check_q0()
    check_outputs()
    check_anonymity()
    subprocess.run(
        [sys.executable, "-m", "py_compile", *[str(path) for path in ROOT.rglob("*.py")]],
        check=True,
        cwd=ROOT,
    )
    print("PASS: manifest, Q0, figure outputs, source syntax, and anonymity checks")


if __name__ == "__main__":
    main()

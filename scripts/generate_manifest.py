#!/usr/bin/env python3
"""Write SHA-256 checksums for all released, non-generated files."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "new_submission_pack", "previews"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
GENERATED_FILES = {
    "paper/fig2_cross_domain_decoupling.pdf",
    "paper/fig3_mechanism.pdf",
    "paper/fig4_necessity.pdf",
    "paper/fig5_scope2x2.pdf",
    "paper/figure_scripts/figure_data_audit.json",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    paths = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and path.relative_to(ROOT).as_posix() not in GENERATED_FILES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    ]
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(paths)]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST.relative_to(ROOT)} with {len(lines)} entries")


if __name__ == "__main__":
    main()

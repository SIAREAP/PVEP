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
    "paper/fig2_ariac.pdf",
    "paper/fig3_tv.pdf",
    "paper/fig4_rotor.pdf",
    "paper/fig5_coverage_scaling.pdf",
    "paper/fig6_scope_intervention.pdf",
    "paper/figure_scripts/fig2_ariac.pdf",
    "paper/figure_scripts/fig3_tv.pdf",
    "paper/figure_scripts/fig4_rotor.pdf",
    "paper/figure_scripts/fig5_coverage_scaling.pdf",
    "paper/figure_scripts/fig6_scope_intervention.pdf",
    "paper/figure_scripts/_preview_fig2_ariac.png",
    "paper/figure_scripts/_preview_fig3_tv.png",
    "paper/figure_scripts/_preview_fig4_rotor.png",
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

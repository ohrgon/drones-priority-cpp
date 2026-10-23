#!/usr/bin/env python3
"""Verify every file in the repository against FILES_SHA256SUMS.json.

Usage (from the repository root):
    python scripts/verify_bundle.py            # verify
    python scripts/verify_bundle.py --write    # regenerate FILES_SHA256SUMS.json

Exit status is 0 when all listed files are present and unchanged and no unlisted
files exist (apart from the record itself and the .git folder).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "FILES_SHA256SUMS.json"
SKIP_DIRS = {".git", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == RECORD:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        yield p


def write_record() -> None:
    files = {}
    for p in tracked_files():
        rel = p.relative_to(ROOT).as_posix()
        files[rel] = {"size_bytes": p.stat().st_size, "sha256": sha256(p)}
    RECORD.write_text(json.dumps({"self_excluded": True, "files": files}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RECORD.name}: {len(files)} files")


def verify() -> int:
    record = json.loads(RECORD.read_text(encoding="utf-8"))["files"]
    present = {p.relative_to(ROOT).as_posix(): p for p in tracked_files()}
    missing = sorted(set(record) - set(present))
    extra = sorted(set(present) - set(record))
    changed = []
    for rel, info in record.items():
        p = present.get(rel)
        if p is None:
            continue
        if p.stat().st_size != info["size_bytes"] or sha256(p) != info["sha256"]:
            changed.append(rel)
    for label, items in (("missing", missing), ("changed", changed), ("unlisted", extra)):
        for rel in items:
            print(f"{label}: {rel}")
    ok = not (missing or changed or extra)
    print(f"{len(record)} files listed, {len(changed)} changed, {len(missing)} missing, {len(extra)} unlisted -> "
          + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        write_record()
        sys.exit(0)
    sys.exit(verify())

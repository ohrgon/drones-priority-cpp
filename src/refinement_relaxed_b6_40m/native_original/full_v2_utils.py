"""Manifest, sampling, selection and serialization utilities (NumPy only)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from coverage_core import validate_priority_map


# Families listed in the revised Section 4.2. Unknown names remain explicit.
PATTERN_FAMILIES = {
    "gradient": ("band gradient", "directional gradient", "radial decay"),
    "corridor_and_separation": ("barrier split", "branching corridor", "linear corridor", "serpentine corridor"),
    "cluster_and_patch": ("single cluster", "multi cluster", "island chain", "perlin patch", "perlin ridge"),
    "loop_and_boundary": ("crescent open ring", "ring loop", "edge frame"),
    "uniform_and_alternating": ("checkerboard sparse", "uniform lattice", "uniform random"),
}


def pattern_family(pattern: str, source: str) -> str:
    if source == "osm":
        return "osm_derived"
    normalize = lambda text: "".join(c for c in text.lower() if c.isalnum())
    name = normalize(pattern)
    for family, members in PATTERN_FAMILIES.items():
        if name in {normalize(m) for m in members}:
            return family
    return "unknown"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    if os.name == "posix":
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path | str, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {k: json_safe(v) for k, v in row.items()}
            writer.writerow(clean)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    if os.name == "posix":
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class CSVLog:
    """Fixed schema, append per row; flush is automatic through reopening."""
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.fields = None
        if self.path.exists():
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                self.fields = next(csv.reader(f), None)

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = self.fields is None
        if new:
            self.fields = list(row)
        if set(row) != set(self.fields):
            raise ValueError(f"CSV schema changed: {self.path}")
        with self.path.open("a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                writer.writeheader()
            writer.writerow({k: json_safe(v) for k, v in row.items()})


def array_hash(a: np.ndarray) -> str:
    normalized = np.ascontiguousarray(a, dtype=np.uint8)
    payload = np.asarray(normalized.shape, dtype="<i8").tobytes() + normalized.tobytes()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    path: str
    source: str
    pattern: str
    tier: str
    split: str
    sha256: str


def load_manifest(path: Path | str, grid_size: int = 50, verify_files: bool = True) -> list[MapRecord]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"map_id", "path", "source", "pattern", "tier", "split", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty")
    records, ids, contents, paths = [], set(), {}, set()
    for row in rows:
        p = Path(row["path"])
        p = (path.parent/p).resolve() if not p.is_absolute() else p.resolve()
        source, split = row["source"].strip().lower(), row["split"].strip().lower()
        if source not in {"synthetic", "osm"}:
            raise ValueError(f"Unresolved source for {row['map_id']}: {source}; do not infer source from pixel values")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if source == "osm" and split != "test":
            raise ValueError("This revised synthetic-only protocol prohibits OSM training/validation")
        key = row["map_id"].strip()
        if not key or key in ids or str(p) in paths:
            raise ValueError(f"Repeated/empty map ID or path: {key}")
        ids.add(key); paths.add(str(p))
        h = row["sha256"].strip()
        if verify_files:
            a = validate_priority_map(np.load(p, allow_pickle=False), grid_size,
                                      min_active=2 if split in {"train", "validation"} else 1)
            actual = array_hash(a)
            if h != actual:
                raise ValueError(f"Map hash changed or missing: {p}; regenerate/review manifest")
        if len(h) != 64:
            raise ValueError(f"Invalid normalized-array SHA256 for {key}")
        if h in contents:
            raise ValueError(f"Duplicate map content: {key} and {contents[h]} (duplicates are not silently dropped)")
        contents[h] = key
        records.append(MapRecord(key, str(p), source, row["pattern"], row["tier"], split, h))
    return sorted(records, key=lambda r: (r.split, r.source, r.pattern, r.tier, r.map_id))


def require_train_splits(records: Sequence[MapRecord]) -> None:
    for split in ("train", "validation", "test"):
        if not any(r.split == split and r.source == "synthetic" for r in records):
            raise ValueError(f"No synthetic {split} maps; do not silently substitute another split")


class BalancedMapOrder:
    """Infinite reproducible map stream: balanced pattern-tier blocks, shuffled members.

    The same seed/rank sees the same nth-episode map across beta configurations.
    Different episode lengths still imply different maps at the same global timestep.
    """
    def __init__(self, records: Sequence[MapRecord], seed: int):
        if not records:
            raise ValueError("No training maps")
        self.records = list(records)
        self.rng = np.random.default_rng(seed)
        self.groups: dict[tuple[str, str], list[int]] = {}
        for i, r in enumerate(records):
            self.groups.setdefault((r.pattern, r.tier), []).append(i)
        self.keys = sorted(self.groups)
        self.group_deck: list[int] = []
        self.member_decks: dict[tuple[str, str], list[int]] = {}
        self.position = 0

    def state_dict(self) -> dict:
        from copy import deepcopy
        return {"record_order": [(r.map_id, r.sha256) for r in self.records],
                "rng": deepcopy(self.rng.bit_generator.state),
                "group_deck": list(self.group_deck),
                "member_decks": deepcopy(self.member_decks), "position": self.position}

    def load_state_dict(self, state) -> None:
        from copy import deepcopy
        if state["record_order"] != [(r.map_id, r.sha256) for r in self.records]:
            raise ValueError("Balanced map-order records changed during resume")
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.group_deck = list(state["group_deck"])
        self.member_decks = deepcopy(state["member_decks"])
        self.position = int(state["position"])

    def next_index(self) -> int:
        if not self.group_deck:
            self.group_deck = self.rng.permutation(len(self.keys)).tolist()
        key = self.keys[self.group_deck.pop()]
        if not self.member_decks.get(key):
            self.member_decks[key] = self.rng.permutation(self.groups[key]).tolist()
        self.position += 1
        return int(self.member_decks[key].pop())


def balanced_selection(candidates: Sequence[dict], minimum_success: float = 1.0) -> tuple[dict | None, list[dict]]:
    """One common min-max normalization over all eligible VALIDATION candidates.

    Every row must have success_rate, latency_sec, mission_time_sec. Ties: raw
    latency+time, then earlier actual timestep. Ineligible rows never win.
    """
    if not 0 <= minimum_success <= 1:
        raise ValueError("minimum_success must lie in [0,1]")
    ranked = [dict(r) for r in candidates]
    valid = [r for r in ranked if float(r["success_rate"]) >= minimum_success
             and all(r.get(k) is not None and np.isfinite(float(r[k])) for k in ("latency_sec", "mission_time_sec"))]
    for r in ranked:
        r["eligible"] = False
        r["J_bal"] = None
    if not valid:
        return None, ranked
    lo = {k: min(float(r[k]) for r in valid) for k in ("latency_sec", "mission_time_sec")}
    hi = {k: max(float(r[k]) for r in valid) for k in lo}
    for r in valid:
        r["eligible"] = True
        norms = []
        for k in lo:
            norm = (float(r[k])-lo[k])/(hi[k]-lo[k]) if hi[k] > lo[k] else 0.0
            r[f"normalized_{k}"] = norm
            norms.append(norm)
        r["J_bal"] = .5*sum(norms)
    best = min(valid, key=lambda r: (r["J_bal"], float(r["latency_sec"])+float(r["mission_time_sec"]), int(r.get("num_timesteps", 0))))
    return dict(best), ranked

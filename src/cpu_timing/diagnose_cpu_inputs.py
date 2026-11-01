#!/usr/bin/env python3
"""Read a small, fixed set of local CPU-benchmark inputs and write one report.

No model execution, network access, copying, input changes, or full-tree search.
The report identifies locations and limits; it does not certify compatibility.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

LIMITS = dict(file_bytes=2_000_000, total_read_bytes=20_000_000, files_read=80,
              directory_entries=200, source_depth=3, source_directories=80,
              source_packages=30, csv_rows=5000, csv_samples=8,
              json_depth=7, json_nodes=800, json_leaves=100, metadata_files=16)
READS = dict(files=0, bytes=0)
LIMIT_EVENTS = []
MARKERS = ("seed_evaluation.py", "ch5_common.py", "PACKAGE_SHA256.json")
PRUNE = {"data", "models", "weights", "checkpoints", "maps", "test", "train",
         "validation", "results", "evaluation", "logs", "runs", "stages", "__pycache__"}


def info(path):
    result = {"path": str(path)}
    try:
        item = path.stat()
        result.update(present=True, is_file=stat.S_ISREG(item.st_mode),
                      is_directory=stat.S_ISDIR(item.st_mode), size_bytes=item.st_size)
    except OSError as exc:
        result.update(present=False, error=f"{type(exc).__name__}: {exc}")
    return result


def read_small(path):
    meta = info(path)
    if not meta.get("is_file"):
        return None, meta
    reason = None
    if meta["size_bytes"] > LIMITS["file_bytes"]:
        reason = "file_bytes"
    elif READS["files"] >= LIMITS["files_read"]:
        reason = "files_read"
    elif READS["bytes"] + meta["size_bytes"] > LIMITS["total_read_bytes"]:
        reason = "total_read_bytes"
    if reason:
        meta.update(skipped=True, truncated=True, limit=reason)
        LIMIT_EVENTS.append({"path": str(path), "limit": reason})
        return None, meta
    try:
        with path.open("rb") as stream:
            data = stream.read(meta["size_bytes"])
            changed = os.fstat(stream.fileno()).st_size != meta["size_bytes"]
        READS["files"] += 1
        READS["bytes"] += len(data)
        if changed or len(data) != meta["size_bytes"]:
            meta.update(skipped=True, truncated=True, limit="file_changed_while_reading")
            LIMIT_EVENTS.append({"path": str(path), "limit": meta["limit"]})
            return None, meta
        meta["sha256"] = hashlib.sha256(data).hexdigest()
        return data, meta
    except OSError as exc:
        meta["read_error"] = f"{type(exc).__name__}: {exc}"
        return None, meta


def directory(path):
    result = dict(path=str(path), directories=[], files=[], truncated=False)
    try:
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= LIMITS["directory_entries"]:
                    result.update(truncated=True, limit="directory_entries")
                    break
                if entry.name.startswith("."):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    result["directories"].append(entry.name)
                elif entry.is_file(follow_symlinks=False):
                    result["files"].append(entry.name)
        result["directories"].sort()
        result["files"].sort()
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_csv(path):
    data, result = read_small(path)
    if data is None:
        return result
    counts, samples, sampled = Counter(), [], Counter()
    result.update(truncated=False, rows_counted=0)
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        result["columns"] = reader.fieldnames
        for index, row in enumerate(reader):
            if index >= LIMITS["csv_rows"]:
                result.update(truncated=True, limit="csv_rows")
                break
            result["rows_counted"] += 1
            group = (row.get("source", ""), row.get("split", ""))
            counts[group] += 1
            if sampled[group] < 2 and len(samples) < LIMITS["csv_samples"]:
                samples.append({key: str(row.get(key, ""))[:1200] for key in
                                ("map_id", "source", "split", "path", "sha256", "file_sha256", "status")})
                sampled[group] += 1
        result.update(source_split_counts=[dict(source=k[0], split=k[1], rows=v)
                                           for k, v in sorted(counts.items())], path_samples=samples,
                      map_files_opened=False)
    except (UnicodeError, csv.Error, TypeError) as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_json(path, candidate_root):
    data, result = read_small(path)
    if data is None:
        return result
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        return result
    leaves, model_paths, queue, nodes, reasons = [], set(), deque([(value, "$", 0)]), 0, set()
    while queue:
        if nodes >= LIMITS["json_nodes"] or len(leaves) >= LIMITS["json_leaves"]:
            reasons.add("json_nodes_or_leaves")
            break
        item, label, depth = queue.popleft()
        nodes += 1
        if isinstance(item, (dict, list)):
            if depth >= LIMITS["json_depth"]:
                reasons.add("json_depth")
                continue
            pairs = item.items() if isinstance(item, dict) else enumerate(item)
            for index, (key, child) in enumerate(pairs):
                if len(queue) + nodes >= LIMITS["json_nodes"]:
                    reasons.add("json_nodes")
                    break
                queue.append((child, f"{label}.{str(key)[:600]}", depth + 1))
        elif re.search(r"path|root|dir|snapshot|model|manifest|sha|missing|message|code|state|status|count|size|bytes|files|source|reason|warning|copied", label, re.I):
            leaves.append({"field": label, "value": item[:1600] if isinstance(item, str) else item})
            if isinstance(item, str) and item.lower().endswith((".zip", ".pth", ".pt")):
                model_paths.add(item)
    result.update(extracted_fields=leaves, nodes_examined=nodes, truncated=bool(reasons),
                  truncation_reasons=sorted(reasons), model_path_stats=[])
    for model_path in sorted(model_paths)[:12]:
        if "<" in model_path or ">" in model_path:
            result["model_path_stats"].append(dict(path=model_path, unresolved_placeholder=True))
        else:
            p = Path(model_path).expanduser()
            result["model_path_stats"].append(info(p if p.is_absolute() else candidate_root / p))
    result["model_weights_read"] = False
    return result


def inspect_sources(root):
    queue = deque([(root / "src", 0)])
    examined, packages, reasons = 0, [], set()
    direct = [root / p for p in ("src/MAIN_SEEDS_8_77", "src/main_seeds_8_77",
              "src/original/root0/MAIN_SEEDS_8_77", "MAIN_SEEDS_8_77", "main_seeds_8_77", ".")]
    seen = set()
    while direct or queue:
        if examined >= LIMITS["source_directories"] or len(packages) >= LIMITS["source_packages"]:
            reasons.add("source_directories_or_packages")
            break
        path, depth = (direct.pop(0), -1) if direct else queue.popleft()
        if str(path) in seen or not path.is_dir() or path.is_symlink():
            continue
        seen.add(str(path))
        examined += 1
        found = {}
        for name in MARKERS:
            p = path / name
            if p.is_file():
                _data, found[name] = read_small(p)
        if found:
            for name in ("ch5_vendor/test8_original.py", "ch5_vendor/segment_graph_original.py",
                         "templates/strict.json", "templates/relaxed.json"):
                p = path / name
                if p.is_file():
                    _data, found[name] = read_small(p)
            packages.append(dict(path=str(path), markers=found,
                                 runtime=info(path / "runtime"), templates=info(path / "templates")))
        if depth < 0:
            continue
        listing = directory(path)
        if listing["truncated"]:
            reasons.add("directory_entries")
        children = [name for name in listing["directories"] if name not in PRUNE]
        if depth >= LIMITS["source_depth"]:
            if children:
                reasons.add("source_depth")
            continue
        queue.extend((path / name, depth + 1) for name in children)
    return dict(search_base=str(root / "src"), directories_examined=examined, packages=packages,
                truncated=bool(reasons), truncation_reasons=sorted(reasons),
                src_listing=directory(root / "src"), src_original_listing=directory(root / "src/original"))


def inspect_candidate(root, is_suite=False):
    result = dict(root=info(root), immediate_listing=directory(root),
                  discovery_required={name: info(root / name) for name in
                                      ("data/manifest_portable.csv", "src/main_seeds_8_77/seed_evaluation.py")})
    if not result["root"].get("is_directory"):
        return result
    csv_names = ["data/manifest_portable.csv", "data/manifest_original.csv", "manifest.csv"]
    if is_suite:
        csv_names += ["protocols_5m_seed42/manifest.csv",
                      "chapter4_C3NT_COST11_STRICT_BETA_0_05_1_seed42/beta0p5/manifest.csv",
                      "chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11/manifest.csv"]
    result["manifests"] = [inspect_csv(root / name) for name in csv_names]
    metadata = {root / name for name in ("provenance/CHECKPOINTS.json", "provenance/PACKER_MISSING_ITEMS.json",
                "audit/CHECKPOINTS.json", "audit/MISSING.json", "audit/MISSING_ITEMS.json",
                "audit/SLIM_REPORT.json", "provenance/SLIM_REPORT.json", "SLIM_REPORT.json")}
    for folder in (root / "audit", root / "provenance"):
        for name in directory(folder)["files"]:
            if name.endswith(".json") and re.search(r"CHECKPOINT|MISSING|SLIM|PACK.*REPORT", name, re.I):
                metadata.add(folder / name)
    result["metadata"] = [inspect_json(path, root) for path in sorted(metadata)[:LIMITS["metadata_files"]]]
    result["metadata_truncated"] = len(metadata) > LIMITS["metadata_files"]
    result["source_search"] = inspect_sources(root)
    result["fixed_model_paths"] = [info(root / f"models/{method}/seed{seed}/final_40001536.zip")
                                   for method in ("strict", "relaxed") for seed in (8, 42, 77)]
    if is_suite:
        result["known_training_receipts"] = [inspect_json(root / parent / f"{method}_seed{seed}" /
                                                         "training_result.json", root)
                                             for parent in ("chapter4_MAIN_COST11_40M_seed8_77", "main_seeds_8_77")
                                             for method in ("strict", "relaxed") for seed in (8, 77)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", default="..", help="Original uav_chapter4_suite_v2 directory")
    parser.add_argument("--out", default="cpu_inputs_report.json", help="New JSON report; existing files are never overwritten")
    args = parser.parse_args()
    suite = Path(args.suite_root).expanduser().resolve()
    candidates = [suite] + [suite.parent / name for name in
                            ("drones-priority-cpp-1gb", "drones-priority-cpp2", "drones-priority-cpp")]
    report = dict(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), cwd=str(Path.cwd()),
                  suite_root=str(suite), limits=LIMITS,
                  scope="Bounded local input-location diagnostic; no complete validation or compatibility verdict.",
                  operations="Read small metadata/source files and stat checkpoint paths; no model execution/network/input changes.",
                  source_search_follows_symlink_subdirectories=False,
                  candidates=[inspect_candidate(root, root == suite) for root in candidates],
                  read_budget_used=READS, global_limit_events=LIMIT_EVENTS)
    out = Path(args.out).expanduser().resolve()
    if not out.parent.is_dir():
        parser.error(f"Output directory does not exist: {out.parent}")
    for suffix in range(10000):
        target = out if suffix == 0 else out.with_name(f"{out.stem}_{suffix:03d}{out.suffix or '.json'}")
        try:
            with target.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"Report saved: {target}")
            print("No inputs changed. This report is a bounded inventory, not a compatibility certification.")
            return
        except FileExistsError:
            continue
    parser.error("Cannot find an unused report filename.")


if __name__ == "__main__":
    main()

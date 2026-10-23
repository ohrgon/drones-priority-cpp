"""Transactional, trusted-local training recovery for Chapter 4.

A recovery generation is published only AFTER an optimizer/update boundary. Atomic
latest.json points to two complete generations; a half-written new generation never
replaces a good one. Evaluation checkpoints are separate from recovery bundles.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
import json
import os
import random
import shutil
import signal
import sys
import threading
import time
import uuid
import warnings

import numpy as np
from full_v2_utils import file_hash, read_json, write_json

FORMAT = "chapter4_resume_v2"


class TrainingPaused(RuntimeError):
    """A requested stop was honored at a durable training boundary."""
    def __init__(self, step: int, signum: int | None = None):
        self.step = int(step)
        self.signum = signum
        super().__init__(f"Training paused safely at {step:,} steps. Re-run the SAME command/root to resume.")


class StopController:
    """First Ctrl+C/SIGTERM requests safe save; second interrupt stops immediately."""
    def __init__(self):
        self.requested = False
        self.signum: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt("Immediate stop; recovery uses the last committed generation")
        self.requested = True
        self.signum = int(signum)
        print("\n[STOP REQUESTED] Finishing current rollout/update, then saving recovery state. "
              "A second Ctrl+C stops immediately and may lose unsaved work.", flush=True)

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
        return self

    def __exit__(self, *exc):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def raise_if_requested(self, step):
        if self.requested:
            raise TrainingPaused(step, self.signum)


@contextmanager
def execution_lock(path: Path | str):
    """OS-released exclusive writer lock; no stale PID deletion is necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another process is using this experiment/run: {path}") from exc
        elif os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0"); handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Another process holds {path}") from exc
        else:
            raise RuntimeError("Exclusive recovery locks require POSIX or Windows")
        yield
    finally:
        # close releases the advisory lock even after a hard process exit.
        handle.close()


def sync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def sync_directory(path):
    if os.name == "posix":
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def runtime_signature(device: str) -> dict:
    import importlib.metadata as md
    import torch
    result = {"device": str(device), "numpy": np.__version__, "torch": str(torch.__version__),
              "python": f"{sys.version_info.major}.{sys.version_info.minor}",
              "cuda_runtime": torch.version.cuda,
              "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for package in ("gymnasium", "stable-baselines3", "sb3-contrib"):
        try:
            result[package] = md.version(package)
        except md.PackageNotFoundError:
            result[package] = None
    return result


def check_runtime(saved, current):
    if saved != current:
        changes = {k: {"saved": saved.get(k), "current": current.get(k)}
                   for k in set(saved) | set(current) if saved.get(k) != current.get(k)}
        raise ValueError(f"Resume runtime differs. Restore the original software/device environment. "
                         f"Not silently treating a changed runtime as the same run: {changes}")


def capture_rng() -> dict:
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark}


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from recovery snapshot")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    torch.use_deterministic_algorithms(state["torch_deterministic"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def capture_log_offsets(run_dir, num_envs):
    names = ["training_episodes.csv", "training_updates.csv"]
    names += [f"map_order_rank{i}.csv" for i in range(num_envs)]
    offsets = {}
    for name in names:
        path = Path(run_dir) / name
        if path.exists():
            sync_file(path)
            offsets[name] = path.stat().st_size
        else:
            offsets[name] = 0
    return offsets


def restore_log_offsets(run_dir, offsets):
    """Discard only the suffix after a committed snapshot, never rewrite saved rows."""
    root = Path(run_dir)
    for name, size in offsets.items():
        p = root / name
        if p.parent != root:
            raise ValueError("Invalid recovery log path")
        if not p.exists():
            if size:
                raise ValueError(f"Recovery log missing: {p}")
            continue
        if p.stat().st_size < size:
            raise ValueError(f"Recovery log is shorter than the committed offset: {p}")
        if size == 0:
            p.unlink()
        elif p.stat().st_size != size:
            with p.open("r+b") as handle:
                handle.truncate(size); handle.flush(); os.fsync(handle.fileno())


def restore_checkpoint_index(run_dir, checkpoints):
    """Validation only sees committed optimizer-boundary model checkpoints."""
    root = Path(run_dir)
    for cp in checkpoints:
        if not Path(cp["path"]).exists() or file_hash(cp["path"]) != cp["sha256"]:
            raise ValueError(f"Committed validation checkpoint changed: {cp['path']}")
    write_json(root / "checkpoints.json", checkpoints)


class ResumeStore:
    def __init__(self, run_dir, identity, keep=2):
        self.root = Path(run_dir) / "resume"
        self.identity = str(identity)
        self.keep = max(2, int(keep))
        self._last_good = None

    def commit(self, step: int, writer: Callable[[Path], Any], *, reason="periodic") -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        identifier = f"step_{int(step):012d}_{uuid.uuid4().hex[:12]}"
        temporary = self.root / (".writing_" + identifier)
        target = self.root / identifier
        temporary.mkdir()
        try:
            writer(temporary)
            entries = {}
            for f in sorted(temporary.rglob("*")):
                if f.is_file():
                    sync_file(f)
                    entries[f.relative_to(temporary).as_posix()] = dict(sha256=file_hash(f), bytes=f.stat().st_size)
            if not entries:
                raise ValueError("Refusing to publish an empty recovery bundle")
            metadata = dict(format=FORMAT, identity=self.identity, step=int(step), reason=reason,
                            saved_unix_time=time.time(), files=entries)
            write_json(temporary / "generation.json", metadata)
            for directory in sorted((p for p in temporary.rglob('*') if p.is_dir()),reverse=True):
                sync_directory(directory)
            sync_directory(temporary)
            os.replace(temporary, target)
            sync_directory(self.root)
            pointer = self.root / "latest.json"
            old = read_json(pointer).get("generations", []) if pointer.exists() else []
            item = dict(directory=identifier, step=int(step), manifest_sha256=file_hash(target / "generation.json"))
            # After corruption fallback, the backup must be the generation which
            # actually loaded, not a known-corrupt newest entry.
            if self._last_good is not None and self._last_good in old:
                old = [self._last_good] + [x for x in old if x != self._last_good]
            kept = [item] + old[: self.keep - 1]
            write_json(pointer, dict(format=FORMAT, identity=self.identity, generations=kept))
            sync_directory(self.root)
            # Publication is complete. Extra generations and abandoned temporary writes
            # can be removed; a failed cleanup must not invalidate a good checkpoint.
            self._prune({x["directory"] for x in kept})
            self._last_good = item
            return item
        except BaseException:
            # May be large/partly written; last committed generation remains untouched.
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _prune(self, keep):
        for directory in self.root.iterdir():
            if directory.is_dir() and (directory.name.startswith("step_") or directory.name.startswith(".writing_")) and directory.name not in keep:
                try:
                    shutil.rmtree(directory)
                except OSError as exc:
                    warnings.warn(f"Could not remove old recovery generation {directory}: {exc}")

    def load(self):
        pointer = self.root / "latest.json"
        if not pointer.exists():
            if self.root.exists():
                self._prune(set())  # abandoned initial writes have no published state
            return None
        published = read_json(pointer)
        if published.get("format") != FORMAT or published.get("identity") != self.identity:
            raise ValueError("Resume identity/format mismatch; refusing incompatible recovery")
        failures = []
        for entry in published["generations"]:
            try:
                d = self.root / entry["directory"]
                if d.parent != self.root:
                    raise ValueError("Invalid generation path")
                if file_hash(d / "generation.json") != entry["manifest_sha256"]:
                    raise ValueError("Generation manifest hash mismatch")
                metadata = read_json(d / "generation.json")
                if metadata.get("format") != FORMAT or metadata.get("identity") != self.identity:
                    raise ValueError("Generation identity mismatch")
                for name, item in metadata["files"].items():
                    file = d / name
                    if not file.resolve().is_relative_to(d.resolve()):
                        raise ValueError("Invalid recovery payload path")
                    if not file.exists() or file.stat().st_size != item["bytes"] or file_hash(file) != item["sha256"]:
                        raise ValueError(f"Payload missing/corrupt: {name}")
                if failures:
                    print(f"[RESUME WARNING] Newest snapshot is invalid; using {entry['directory']}. {failures}", flush=True)
                self._last_good = entry
                self._prune({x['directory'] for x in published['generations']})
                return d, metadata
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                failures.append(f"{entry.get('directory')}: {exc}")
        raise ValueError(f"No valid committed recovery generation: {failures}. Original files preserved")

    def clear_after_completion(self):
        # Only call after final artifacts and COMPLETED status have been committed.
        if self.root.exists():
            shutil.rmtree(self.root)


def torch_load_local(path):
    """Never use on untrusted external checkpoints (Torch/pickle can execute code)."""
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)

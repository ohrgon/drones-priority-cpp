"""Stdlib-only tests for scheduler ownership, concurrency and shutdown.

Fake children test orchestration semantics, not machine performance.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_cpu.py"
if not MODULE_PATH.is_file():
    MODULE_PATH = Path(__file__).resolve().parents[1] / "cpu_timing_package/benchmark_cpu.py"
SPEC = importlib.util.spec_from_file_location("benchmark_parallel_under_test", MODULE_PATH)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def config():
    keys = [f"{method}_seed{seed}" for seed in (8, 42, 77) for method in ("strict", "relaxed")]
    slots = {key: (8, 42, 77).index(int(key.split("_seed")[1])) for key in keys}
    return dict(model_keys=keys, model_slots=slots, workers=3,
                cpu_groups=[[0, 1], [2, 3], [4, 5]])


class FakeProcess:
    def __init__(self, owner, model, stream, completion_poll, exit_code, ignore_term):
        self.owner, self.model, self.stdout = owner, model, stream
        self.slot = owner.config["model_slots"][model]
        self.pid = 1000 + len(owner.processes)
        self.completion_poll, self.exit_code = completion_poll, exit_code
        self.ignore_term = ignore_term
        self.polls = 0
        self.returncode = None
        self.reaped = False
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        self.polls += 1
        if self.returncode is None and self.polls >= self.completion_poll:
            self.returncode = self.exit_code
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.returncode is None:
            if timeout is not None:
                raise B.subprocess.TimeoutExpired(self.model, timeout)
            raise AssertionError("Scheduler waits on a live fake child without polling or terminating it")
        self.reaped = True
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.ignore_term:
            self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9


class FakeSpawner:
    def __init__(self, settings, *, failed=None, completion_poll=2, ignore_term=None):
        self.config = settings
        self.failed = failed or {}
        self.completion_poll = completion_poll
        self.ignore_term = set(ignore_term or ())
        self.processes = []
        self.max_live = 0
        self.start_order = []

    def __call__(self, command, *, stdout, stderr, env):
        model = command[-1]
        slot = self.config["model_slots"][model]
        live = [p for p in self.processes if not p.reaped]
        if any(p.slot == slot for p in live):
            raise AssertionError("Two live children were assigned to the same slot")
        if len(live) >= self.config["workers"]:
            raise AssertionError("Scheduler exceeded configured concurrency")
        if model in self.start_order:
            raise AssertionError("A model was scheduled more than once")
        child = FakeProcess(self, model, stdout, self.completion_poll,
                            self.failed.get(model, 0), model in self.ignore_term)
        self.processes.append(child)
        self.start_order.append(model)
        self.max_live = max(self.max_live, len(live) + 1)
        return child


class SchedulerTests(unittest.TestCase):
    def invoke(self, out, settings, spawner, *, interrupt=False):
        (out / "logs").mkdir(exist_ok=True)
        captured = io.StringIO()
        with patch.object(B.subprocess, "Popen", spawner), \
             patch.object(B.time, "sleep", side_effect=KeyboardInterrupt if interrupt else None), \
             contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            return B.run_model_workers(settings, out)

    def test_bound_concurrency_and_stable_per_seed_slots(self):
        settings = config()
        spawner = FakeSpawner(settings)
        with TemporaryDirectory() as temp:
            out = Path(temp)
            self.assertEqual(self.invoke(out, settings, spawner), [])
            events = json.loads((out / "worker_events.json").read_text())
        self.assertEqual(spawner.max_live, 3)
        self.assertEqual(set(spawner.start_order), set(settings["model_keys"]))
        self.assertEqual(len(spawner.processes), 6)
        for seed in (8, 42, 77):
            strict, relaxed = f"strict_seed{seed}", f"relaxed_seed{seed}"
            self.assertLess(spawner.start_order.index(strict), spawner.start_order.index(relaxed))
        self.assertTrue(all(p.reaped and p.wait_calls == 1 and p.stdout.closed for p in spawner.processes))
        self.assertTrue(all(not p.terminated and not p.killed for p in spawner.processes))
        for event in events:
            slot = settings["model_slots"][event["model_key"]]
            self.assertEqual(event["worker_slot"], slot)
            self.assertEqual(event["cpu_ids"], settings["cpu_groups"][slot])
            self.assertEqual(event["return_code"], 0)

    def test_failed_job_does_not_prevent_other_models_finishing(self):
        settings = config()
        spawner = FakeSpawner(settings, failed={"strict_seed42": 17})
        with TemporaryDirectory() as temp:
            out = Path(temp)
            errors = self.invoke(out, settings, spawner)
            events = json.loads((out / "worker_events.json").read_text())
        self.assertEqual([(e["model_key"], e["return_code"]) for e in errors], [("strict_seed42", 17)])
        self.assertEqual(set(spawner.start_order), set(settings["model_keys"]))
        self.assertEqual(len(events), 6)
        self.assertTrue(all(p.reaped and p.stdout.closed for p in spawner.processes))
        self.assertTrue(all(not p.terminated for p in spawner.processes))

    def test_interrupt_terminates_reaps_and_closes_every_active_child(self):
        settings = config()
        # Keep children alive until the simulated interrupt. One ignores TERM,
        # exercising the kill fallback without a real sleep.
        spawner = FakeSpawner(settings, completion_poll=999, ignore_term={"strict_seed42"})
        with TemporaryDirectory() as temp:
            out = Path(temp)
            with self.assertRaises(KeyboardInterrupt):
                self.invoke(out, settings, spawner, interrupt=True)
            events = json.loads((out / "worker_events.json").read_text())
        self.assertEqual(len(spawner.processes), 3)
        self.assertTrue(all(p.terminated and p.reaped and p.stdout.closed for p in spawner.processes))
        self.assertEqual([p.model for p in spawner.processes if p.killed], ["strict_seed42"])
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e["interrupted"] for e in events))
        self.assertTrue(all(e["return_code"] in (-15, -9) for e in events))


class ParallelTrialIdentityTests(unittest.TestCase):
    def test_resume_rejects_wrong_worker_slot_or_cpu_group(self):
        settings = dict(identity="fixed", model_keys=["strict_seed8"], repeats=1,
                        model_slots={"strict_seed8": 0}, model_cpu_ids={"strict_seed8": [2, 3]})
        records = {"map-a": dict(sha256="map-sha", source="synthetic")}
        row = dict(run_identity="fixed", model_key="strict_seed8", method="strict", train_seed=8,
                   map_id="map-a", map_sha256="map-sha", source="synthetic", repeat=1,
                   worker_slot=0, cpu_ids=[2, 3], raw_success=False, refined_success=None,
                   raw_wall_sec=1.0, attempt_wall_sec=1.0,
                   refine_validate_wall_sec=None, total_plan_wall_sec=None)
        B.validate_trial(row, "strict_seed8", settings, records)
        for bad in ({"worker_slot": 1}, {"cpu_ids": [4, 5]}):
            with self.subTest(bad=bad), self.assertRaisesRegex(RuntimeError, "CPU allocation/worker slot"):
                B.validate_trial(dict(row, **bad), "strict_seed8", settings, records)


if __name__ == "__main__":
    unittest.main(verbosity=2)

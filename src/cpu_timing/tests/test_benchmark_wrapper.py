"""Focused wrapper tests. Fixed fake clocks test accounting, not performance."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmark_cpu.py"
SPEC = importlib.util.spec_from_file_location("benchmark_cpu_under_test", MODULE_PATH)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class FakeActions:
    def __init__(self, data):
        self.data = bytes(data)

    def tobytes(self):
        return self.data


def bundle_for(success):
    raw = dict(is_success=success, steps=6 if success else 15, active_cells=3,
               coverage_pct=100 if success else 2 / 3 * 100,
               mission_time_sec=10.0 if success else 25.0,
               Lw_sec=4.0 if success else None, policy_compute_time_sec=0.25)
    trace = {"actions": FakeActions([0, 1, 2])}
    refined = dict(is_success=True, steps=5, mission_time_sec=9.0, Lw_sec=3.5)
    refined_trace = {"actions": FakeActions([0, 1])}
    evaluate = Mock(return_value=(raw, trace))
    refine = Mock(return_value=(refined, refined_trace, {"order_preserved": True}, []))
    v = SimpleNamespace(evaluate_map=evaluate, truth=bool, T="mission_time_sec", LW="Lw_sec")
    c = SimpleNamespace(common_refine=refine)
    bundle = SimpleNamespace(V=v, C=c, rt=object(), native=object(), model=object())
    return bundle, raw, trace


class MeasureCaseTests(unittest.TestCase):
    def setUp(self):
        self.rec = {"map_id": "a"}
        self.array = object()
        self.ic = dict(pair_id="pair-a", start=[2, 3], heading=1)

    def test_success_times_full_calls_without_extra_replay_or_disk_io(self):
        bundle, raw, trace = bundle_for(True)
        # Gap between Raw end and refinement start is excluded; sum of calls is reported.
        with patch.object(B.time, "perf_counter_ns", side_effect=[0, 2_000_000_000, 5_000_000_000, 8_000_000_000]) as clock:
            row = B.measure_case(bundle, self.rec, self.array, self.ic)
        bundle.V.evaluate_map.assert_called_once_with(bundle.rt, bundle.native, bundle.model,
                                                     self.rec, self.array, self.ic, old=None)
        bundle.C.common_refine.assert_called_once_with(bundle.rt, raw, trace)
        self.assertEqual(clock.call_count, 4)
        self.assertEqual(row["raw_wall_sec"], 2.0)
        self.assertEqual(row["refine_validate_wall_sec"], 3.0)
        self.assertEqual(row["total_plan_wall_sec"], 5.0)
        self.assertEqual(row["attempt_wall_sec"], 5.0)
        self.assertEqual(row["policy_action_only_wall_sec"], 0.25)
        self.assertNotEqual(row["raw_wall_sec"], row["raw_model_T_sec"])
        self.assertTrue(row["refined_success"])
        self.assertEqual(row["initial_pair_id"], "pair-a")
        self.assertEqual(len(row["raw_actions_sha256"]), 64)

    def test_failed_raw_is_measured_but_refinement_is_never_called(self):
        bundle, _, _ = bundle_for(False)
        bundle.C.common_refine.side_effect = AssertionError("Failed Raw must not reach refinement")
        with patch.object(B.time, "perf_counter_ns", side_effect=[0, 7_000_000_000]) as clock:
            row = B.measure_case(bundle, self.rec, self.array, self.ic)
        bundle.C.common_refine.assert_not_called()
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(row["raw_wall_sec"], 7.0)
        self.assertEqual(row["attempt_wall_sec"], 7.0)
        for field in ("refined_success", "refined_steps", "refine_validate_wall_sec",
                      "total_plan_wall_sec", "refined_actions_sha256", "raw_model_Lw_sec"):
            self.assertIsNone(row[field], field)
        self.assertEqual(row["refinement_status"], "skipped_raw_incomplete")

    def test_refinement_losing_completion_is_rejected(self):
        bundle, _, _ = bundle_for(True)
        bundle.C.common_refine.return_value = ({"is_success": False}, {}, {}, [])
        with patch.object(B.time, "perf_counter_ns", side_effect=[0, 1, 2, 3]):
            with self.assertRaisesRegex(RuntimeError, "lost completion"):
                B.measure_case(bundle, self.rec, self.array, self.ic)


def make_trials():
    plans = {
        ("strict_seed8", "a"): (True, [1, 3, 5], [3, 1, 9]),
        ("strict_seed8", "b"): (True, [10, 12, 14], [2, 2, 2]),
        ("strict_seed8", "c"): (False, [20, 22, 24], None),
        ("relaxed_seed8", "a"): (True, [2, 4, 6], [2, 2, 2]),
        ("relaxed_seed8", "b"): (False, [30, 32, 34], None),
        ("relaxed_seed8", "c"): (True, [7, 9, 11], [1, 1, 1]),
    }
    rows = []
    for (model, map_id), (success, raw, refine) in plans.items():
        for i in range(3):
            total = float(raw[i] + refine[i]) if success else None
            rows.append(dict(
                model_key=model, method=model.split("_seed")[0], train_seed=8,
                map_id=map_id, source="osm" if map_id == "c" else "synthetic",
                pattern="fixed", tier="1", repeat=i + 1, run_identity="test-identity",
                map_sha256="sha-" + map_id, initial_pair_id="pair-" + map_id,
                raw_success=success, refined_success=True if success else None,
                raw_steps=10, refined_steps=8 if success else None,
                raw_actions_sha256="raw-" + model + map_id,
                refined_actions_sha256="refined-" + model + map_id if success else None,
                raw_model_T_sec=100.0, raw_model_Lw_sec=50.0 if success else None,
                refined_model_T_sec=90.0 if success else None,
                refined_model_Lw_sec=45.0 if success else None,
                raw_wall_sec=float(raw[i]), refine_validate_wall_sec=float(refine[i]) if success else None,
                total_plan_wall_sec=total, attempt_wall_sec=total if success else float(raw[i]),
            ))
    return rows


def write_fixture(out, rows):
    out = Path(out)
    (out / "trials").mkdir()
    config = dict(identity="test-identity", model_keys=["strict_seed8", "relaxed_seed8"],
                  repeats=3, sources=["synthetic", "osm"], cpu_ids=[0], threads=1,
                  records=[dict(map_id=x, sha256="sha-" + x,
                                source="osm" if x == "c" else "synthetic") for x in ("a", "b", "c")])
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    for model in config["model_keys"]:
        content = "".join(json.dumps(row) + "\n" for row in rows if row["model_key"] == model)
        (out / "trials" / (model + ".jsonl")).write_text(content, encoding="utf-8")
    return config


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows, **criteria):
    found = [row for row in rows if all(row[k] == str(v) for k, v in criteria.items())]
    if len(found) != 1:
        raise AssertionError((criteria, found))
    return found[0]


class SummaryTests(unittest.TestCase):
    def test_summary_uses_map_medians_failures_and_common_success_correctly(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, make_trials())
            result = B.summarize(out)
            self.assertEqual(result, dict(complete=True, n_trials=18, expected_trials=18))
            maps = read_csv(out / "per_map.csv")
            map_a = find_row(maps, model_key="strict_seed8", map_id="a")
            # Median(raw+refine) must be computed directly; medians are not additive.
            self.assertEqual(float(map_a["raw_wall_sec_median"]), 3.0)
            self.assertEqual(float(map_a["refine_validate_wall_sec_median"]), 3.0)
            self.assertEqual(float(map_a["total_plan_wall_sec_median"]), 4.0)
            summary = read_csv(out / "summary.csv")
            success = find_row(summary, model_key="strict_seed8", source="all",
                               population="successful", metric="total_plan_wall_sec")
            self.assertEqual(int(success["n_maps"]), 2)
            self.assertEqual(float(success["mean"]), 9.0)
            self.assertAlmostEqual(float(success["p95"]), 13.5)
            self.assertEqual(float(success["max_of_map_medians"]), 14.0)
            self.assertEqual(float(success["max_single_trial"]), 16.0)
            failed = find_row(summary, model_key="strict_seed8", source="all",
                              population="failed", metric="raw_wall_sec")
            self.assertEqual(float(failed["mean"]), 22.0)
            self.assertFalse(any(row["population"] == "failed" and row["metric"] in
                                 ("total_plan_wall_sec", "refine_validate_wall_sec") for row in summary))
            common = find_row(summary, model_key="strict_seed8", source="all",
                              population="common_success", metric="total_plan_wall_sec")
            self.assertEqual(int(common["n_maps"]), 1)
            self.assertEqual(float(common["mean"]), 4.0)
            other_common = find_row(summary, model_key="relaxed_seed8", source="all",
                                    population="common_success", metric="total_plan_wall_sec")
            self.assertEqual(float(other_common["mean"]), 6.0)
            paper = find_row(read_csv(out / "paper_table.csv"), model_key="strict_seed8",
                             source="all", population="successful")
            self.assertEqual((paper["n_evaluated_maps"], paper["n_success"], paper["n_failed"]), ("3", "2", "1"))
            self.assertAlmostEqual(float(paper["success_pct"]), 200 / 3)

    def test_incomplete_repetitions_are_excluded_from_reported_timing_populations(self):
        rows = [r for r in make_trials() if not
                (r["model_key"] == "strict_seed8" and r["map_id"] == "a" and r["repeat"] == 3)]
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, rows)
            result = B.summarize(out)
            self.assertFalse(result["complete"])
            summary = read_csv(out / "summary.csv")
            strict = find_row(summary, model_key="strict_seed8", source="all",
                              population="successful", metric="total_plan_wall_sec")
            self.assertEqual(int(strict["n_maps"]), 1)
            self.assertEqual(float(strict["mean"]), 14.0)
            self.assertFalse(any(row["population"] == "common_success" for row in summary))

    def test_changed_repeated_path_is_rejected(self):
        rows = make_trials()
        rows[0]["raw_actions_sha256"] = "different-path"
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, rows)
            with self.assertRaisesRegex(RuntimeError, "Repeated paths differ"):
                B.summarize(out)

    def test_different_initial_conditions_across_models_are_rejected(self):
        rows = make_trials()
        for row in rows:
            if row["model_key"] == "relaxed_seed8" and row["map_id"] == "a":
                row["initial_pair_id"] = "wrong-initial-condition"
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, rows)
            with self.assertRaisesRegex(RuntimeError, "Initial conditions differ"):
                B.summarize(out)

    def test_out_of_range_repeat_is_rejected_even_when_row_count_is_complete(self):
        rows = make_trials()
        rows[0]["repeat"] = 0
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, rows)
            with self.assertRaisesRegex(RuntimeError, "repeat|trial|configuration|membership"):
                B.summarize(out)

    def test_unknown_map_is_rejected_even_when_row_count_is_complete(self):
        rows = make_trials()
        rows[0]["map_id"] = "unconfigured-map"
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_fixture(out, rows)
            with self.assertRaisesRegex(RuntimeError, "map.*configuration"):
                B.summarize(out)

    def test_wrong_model_result_file_is_rejected(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            rows = make_trials()
            write_fixture(out, rows)
            path = out / "trials/strict_seed8.jsonl"
            stored = B.read_trials(path)
            stored[0]["model_key"] = "relaxed_seed8"
            stored[0]["method"] = "relaxed"
            path.write_text("".join(json.dumps(row) + "\n" for row in stored), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "model membership/file"):
                B.summarize(out)

    def test_duplicate_copied_across_model_files_is_rejected(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            rows = make_trials()
            write_fixture(out, rows)
            with (out / "trials/relaxed_seed8.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(RuntimeError, "membership|Duplicate"):
                B.summarize(out)

    def test_inconsistent_total_and_failed_zero_refinement_are_rejected(self):
        for kind in ("wrong_total", "failed_zero_refinement"):
            with self.subTest(kind=kind), TemporaryDirectory() as temp:
                rows = make_trials()
                if kind == "wrong_total":
                    rows[0]["total_plan_wall_sec"] += 1
                else:
                    next(row for row in rows if not row["raw_success"])["refine_validate_wall_sec"] = 0.0
                out = Path(temp)
                write_fixture(out, rows)
                with self.assertRaisesRegex(RuntimeError, "total|Failed Raw"):
                    B.summarize(out)


class TrialFileTests(unittest.TestCase):
    def test_interrupted_tail_is_repaired_without_changing_completed_rows(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "trials.jsonl"
            row = dict(model_key="strict_seed8", map_id="a", repeat=1)
            prefix = (json.dumps(row) + "\n").encode()
            path.write_bytes(prefix + b'{"model_key":')
            with self.assertRaisesRegex(RuntimeError, "Incomplete final"):
                B.read_trials(path)
            self.assertEqual(B.read_trials(path, repair_tail=True), [row])
            self.assertEqual(path.read_bytes(), prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)

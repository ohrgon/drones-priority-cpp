"""Focused allocation checks against artificial Linux CPU topologies."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpu_bench_parallel import allocate_cpu_groups, assign_models_to_slots


def details(rows):
    return {
        "available_cpu_ids": [cpu for cpu, _socket, _core, _node in rows],
        "topology": {
            str(cpu): dict(physical_package_id=socket, core_id=core,
                           numa_node=node, topology_verified=True)
            for cpu, socket, core, node in rows
        },
    }


class AllocationTests(unittest.TestCase):
    def assert_disjoint(self, data, groups, threads):
        keys = []
        for group in groups:
            self.assertEqual(len(group), threads)
            for cpu in group:
                topo = data["topology"][str(cpu)]
                keys.append((topo["physical_package_id"], topo["core_id"]))
        self.assertEqual(len(keys), len(set(keys)))

    def test_smt_siblings_are_never_separate_cores(self):
        data = details([(cpu, 0, cpu % 12, 0) for cpu in range(24)])
        groups = allocate_cpu_groups(data, 4, 3)
        self.assertEqual(len(groups), 3)
        self.assert_disjoint(data, groups, 4)
        self.assertEqual({cpu for group in groups for cpu in group}, set(range(12)))

    def test_single_bucket_preferred_for_all_groups(self):
        data = details([(cpu, 0, cpu, 0 if cpu < 4 else 1) for cpu in range(16)])
        groups = allocate_cpu_groups(data, 4, 3)
        self.assertEqual({data["topology"][str(cpu)]["numa_node"]
                          for group in groups for cpu in group}, {1})
        self.assert_disjoint(data, groups, 4)

    def test_multiple_numa_nodes_keep_each_group_local(self):
        data = details([(cpu, 0, cpu, cpu // 8) for cpu in range(16)])
        groups = allocate_cpu_groups(data, 4, 3)
        self.assert_disjoint(data, groups, 4)
        for group in groups:
            self.assertEqual(len({data["topology"][str(cpu)]["numa_node"] for cpu in group}), 1)

    def test_fragmented_nodes_preserve_disjoint_cores(self):
        data = details([(cpu, 0, cpu, cpu // 3) for cpu in range(12)])
        groups = allocate_cpu_groups(data, 4, 3)
        self.assert_disjoint(data, groups, 4)

    def test_insufficient_physical_capacity_does_not_use_smt(self):
        data = details([(cpu, 0, cpu % 4, 0) for cpu in range(8)])
        with self.assertRaisesRegex(RuntimeError, "only 4"):
            allocate_cpu_groups(data, 2, 3)

    def test_explicit_pool_is_a_pool_not_shared_allocation(self):
        data = details([(cpu, 0, cpu % 12, 0) for cpu in range(24)])
        groups = allocate_cpu_groups(data, 2, 3, "12-17")
        self.assertEqual({cpu for group in groups for cpu in group}, set(range(12, 18)))
        self.assert_disjoint(data, groups, 2)

    def test_explicit_pool_outside_affinity_or_malformed_is_rejected(self):
        data = details([(cpu, 0, cpu, 0) for cpu in range(12)])
        for pool in ("0,1,99", "0,,1", "4-2", "", "-1,2", "0-999999999"):
            with self.subTest(pool=pool), self.assertRaises(RuntimeError):
                allocate_cpu_groups(data, 1, 1, pool)

    def test_unverified_topology_is_excluded(self):
        data = details([(cpu, 0, cpu, 0) for cpu in range(12)])
        for cpu in range(8, 12):
            data["topology"][str(cpu)]["topology_verified"] = False
        with self.assertRaisesRegex(RuntimeError, "only 8"):
            allocate_cpu_groups(data, 4, 3)
        self.assert_disjoint(data, allocate_cpu_groups(data, 4, 2), 4)

    def test_topology_numa_disagreement_is_rejected(self):
        data = details([(0, 0, 0, 0), (8, 0, 0, 1)])
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            allocate_cpu_groups(data, 1, 1)

    def test_invalid_worker_thread_arguments(self):
        data = details([(0, 0, 0, None)])
        for threads, workers in ((0, 1), (1, 0), (True, 1), (1, True)):
            with self.subTest(threads=threads, workers=workers), self.assertRaises(RuntimeError):
                allocate_cpu_groups(data, threads, workers)


class AssignmentTests(unittest.TestCase):
    keys = [f"{method}_seed{seed}" for seed in (8, 42, 77) for method in ("strict", "relaxed")]

    def test_standard_three_slots_keep_seed_pairs_together(self):
        assigned = assign_models_to_slots(self.keys, 3)
        for slot, seed in enumerate((8, 42, 77)):
            self.assertEqual(assigned[f"strict_seed{seed}"], slot)
            self.assertEqual(assigned[f"relaxed_seed{seed}"], slot)

    def test_two_slots_assign_seeds_roundrobin(self):
        assigned = assign_models_to_slots(self.keys, 2)
        self.assertEqual([assigned[key] for key in self.keys], [0, 0, 1, 1, 0, 0])

    def test_extra_slots_distribute_individual_models(self):
        assigned = assign_models_to_slots(self.keys, 4)
        self.assertEqual([assigned[key] for key in self.keys], [0, 1, 2, 3, 0, 1])

    def test_each_job_has_exactly_one_slot(self):
        for workers in (1, 2, 3, 4, 6):
            assigned = assign_models_to_slots(self.keys, workers)
            self.assertEqual(set(assigned), set(self.keys))
            self.assertTrue(all(0 <= slot < workers for slot in assigned.values()))

    def test_duplicate_and_invalid_model_keys_are_rejected(self):
        for keys in ([], ["strict_seed8", "strict_seed8"], ["strict"], [None]):
            with self.subTest(keys=keys), self.assertRaises(RuntimeError):
                assign_models_to_slots(keys, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

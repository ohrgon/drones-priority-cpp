"""Deterministic CPU allocation and model-slot assignment for parallel timing.

Only standard-library imports are used.  Allocation never uses two SMT siblings
or invents topology when the operating system did not identify a physical core.
"""
from __future__ import annotations

from collections import defaultdict
import re


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _positive_integer(value: int, name: str) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{name} must be a positive integer.")


def _eligible_cpus(text: str, available: set[int]) -> list[int]:
    _require(isinstance(text, str), "cpu_ids_text must be 'auto' or a numeric CPU pool.")
    if text == "auto":
        return sorted(available)
    _require(bool(text.strip()), "An explicit CPU pool cannot be empty.")
    selected: set[int] = set()
    for part in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", part)
        _require(match is not None,
                 "CPU pool must use numeric IDs/ranges such as 0,2,4,6 or 0-11.")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        _require(end >= start, f"CPU range is reversed: {part.strip()}")
        # Avoid expanding an arbitrarily large invalid range into memory.
        _require(end - start + 1 <= len(available),
                 f"CPU range contains IDs outside the permitted CPU pool: {part.strip()}")
        selected.update(range(start, end + 1))
    unavailable = sorted(selected - available)
    _require(not unavailable,
             f"Explicit CPU pool includes IDs outside process affinity: {unavailable}")
    _require(bool(selected), "No eligible CPUs were selected.")
    return sorted(selected)


def _bucket_order(bucket: tuple[int, int | None]) -> tuple:
    socket, node = bucket
    return (socket, node is None, node if node is not None else -1)


def allocate_cpu_groups(
    details: dict,
    threads: int,
    workers: int,
    cpu_ids_text: str = "auto",
) -> list[list[int]]:
    """Allocate ``workers`` disjoint groups of ``threads`` physical cores.

    Explicit CPU IDs describe an *eligible pool*, not a group to be reused by
    every worker.  For each verified (socket, core) pair, one allowed logical ID
    is selected.  A single (socket, NUMA node) hosts every group when possible;
    otherwise complete groups are placed within separate NUMA buckets.  Only
    when bucket fragmentation prevents that layout are remaining cores joined,
    preferring one socket before spanning sockets.  Unknown NUMA nodes are
    recorded as None; verified physical core identity is still mandatory.
    """
    _positive_integer(threads, "threads")
    _positive_integer(workers, "workers")
    _require(isinstance(details, dict), "CPU details must be a dictionary.")
    available_values = details.get("available_cpu_ids", [])
    _require(isinstance(available_values, (list, tuple, set)) and available_values,
             "No allowed CPU IDs are available.")
    _require(all(isinstance(cpu, int) and not isinstance(cpu, bool) and cpu >= 0
                 for cpu in available_values), "Allowed CPU IDs must be nonnegative integers.")
    available = set(available_values)
    eligible = _eligible_cpus(cpu_ids_text, available)
    topology = details.get("topology", {})
    _require(isinstance(topology, dict), "CPU topology must be a dictionary.")

    physical: dict[tuple[int, int], dict] = {}
    unverified: list[int] = []
    for cpu in eligible:
        entry = topology.get(str(cpu), {})
        if not isinstance(entry, dict) or entry.get("topology_verified") is not True:
            unverified.append(cpu)
            continue
        socket, core = entry.get("physical_package_id"), entry.get("core_id")
        _require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                     for value in (socket, core)), f"Invalid verified topology for CPU {cpu}.")
        node = entry.get("numa_node")
        _require(node is None or (isinstance(node, int) and not isinstance(node, bool) and node >= 0),
                 f"Invalid NUMA node for CPU {cpu}.")
        key = (socket, core)
        if key not in physical:
            physical[key] = {"cpu": cpu, "node": node}
        else:
            previous_node = physical[key]["node"]
            _require(previous_node is None or node is None or previous_node == node,
                     f"SMT siblings disagree about NUMA node for physical core {key}.")
            physical[key]["cpu"] = min(physical[key]["cpu"], cpu)
            if previous_node is None:
                physical[key]["node"] = node

    needed = threads * workers
    _require(
        len(physical) >= needed,
        f"Need {needed} distinct verified physical cores for {workers} workers x {threads} threads, "
        f"but the eligible pool contains only {len(physical)}. "
        f"Unverified logical CPUs excluded: {len(unverified)}. "
        "Reduce --workers or --threads, or select a larger permitted CPU pool; "
        "SMT siblings and overlapping groups are not used as substitutes.",
    )
    buckets: dict[tuple[int, int | None], list[int]] = defaultdict(list)
    for (socket, core), entry in sorted(physical.items()):
        buckets[(socket, entry["node"])].append(entry["cpu"])

    # First try one locality bucket for all groups, using the smallest fitting
    # bucket to avoid consuming unrelated sockets when alternatives are equal.
    fitting = [bucket for bucket, cpus in buckets.items() if len(cpus) >= needed]
    if fitting:
        bucket = min(fitting, key=lambda key: (len(buckets[key]), _bucket_order(key)))
        selected = buckets[bucket][:needed]
        return [sorted(selected[start:start + threads]) for start in range(0, needed, threads)]

    groups: list[list[int]] = []
    # Preserve NUMA locality wherever a complete group fits.
    for bucket in sorted(buckets, key=lambda key: (-(len(buckets[key]) // threads),
                                                  -len(buckets[key]), _bucket_order(key))):
        while len(buckets[bucket]) >= threads and len(groups) < workers:
            groups.append(sorted(buckets[bucket][:threads]))
            del buckets[bucket][:threads]
    # If every bucket has a small remainder, prefer combining nodes from one
    # socket.  Physical identity remains disjoint even if locality is imperfect.
    while len(groups) < workers:
        socket_counts: dict[int, int] = defaultdict(int)
        for (socket, _node), cpus in buckets.items():
            socket_counts[socket] += len(cpus)
        fitting_sockets = [socket for socket, count in socket_counts.items() if count >= threads]
        if fitting_sockets:
            socket = min(fitting_sockets, key=lambda key: (-socket_counts[key], key))
            candidates = [key for key in buckets if key[0] == socket and buckets[key]]
        else:
            candidates = [key for key in buckets if buckets[key]]
        candidates.sort(key=lambda key: (-len(buckets[key]), _bucket_order(key)))
        group: list[int] = []
        for bucket in candidates:
            count = min(threads - len(group), len(buckets[bucket]))
            group.extend(buckets[bucket][:count])
            del buckets[bucket][:count]
            if len(group) == threads:
                break
        _require(len(group) == threads, "Internal allocation error: remaining cores are insufficient.")
        groups.append(sorted(group))
    flat = [cpu for group in groups for cpu in group]
    _require(len(flat) == len(set(flat)) == needed, "Internal allocation error: repeated CPU ID.")
    chosen_physical = {(topology[str(cpu)]["physical_package_id"], topology[str(cpu)]["core_id"])
                       for cpu in flat}
    _require(len(chosen_physical) == needed, "Internal allocation error: SMT siblings overlap.")
    return groups


def assign_models_to_slots(model_keys: list[str], workers: int) -> dict[str, int]:
    """Keep each seed's pair together unless there are more slots than seeds.

    Seed order follows first appearance in ``model_keys``.  When workers exceed
    the number of selected seeds, individual model jobs are assigned roundrobin
    in input order so the extra slots can run useful work.
    """
    _positive_integer(workers, "workers")
    _require(isinstance(model_keys, list) and model_keys,
             "model_keys must be a nonempty list.")
    _require(all(isinstance(key, str) for key in model_keys), "Model keys must be strings.")
    _require(len(model_keys) == len(set(model_keys)), "Each model key must appear exactly once.")
    seed_by_model: dict[str, int] = {}
    seeds: list[int] = []
    for key in model_keys:
        match = re.fullmatch(r"(?:strict|relaxed)_seed(\d+)", key)
        _require(match is not None, f"Unrecognized model key: {key}")
        seed = int(match.group(1))
        seed_by_model[key] = seed
        if seed not in seeds:
            seeds.append(seed)
    if workers > len(seeds):
        return {key: index % workers for index, key in enumerate(model_keys)}
    slot_by_seed = {seed: index % workers for index, seed in enumerate(seeds)}
    return {key: slot_by_seed[seed_by_model[key]] for key in model_keys}

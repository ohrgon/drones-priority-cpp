#!/usr/bin/env python3
"""Reproduce the measured columns of Table 23 from the flight records and the planned routes.

For every route flown for Table 23 (data/flight_logs/route_binding.csv, used_in_table23 = yes):

* T_exec  = sum of the OSD.flyTime [s] interval spans of its record sections (Eq. 19);
* RMS     = root mean square of the minimum distance from each recorded position sample of the
            route (data/flight_logs/routes/*.csv) to the segments of the planned waypoint route
            (data/flight_logs/planned_routes/*.csv), Eq. 21.  Both are in the same local
            east-north frame, so the distance is the plain 2-D Euclidean distance.

Run from the repository root:

    python scripts/table23_reproduce.py
"""
from __future__ import annotations

import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data" / "flight_logs"

# Table 23 of the manuscript (T_exec in s, RMS in m)
TABLE23 = {
    ("synthetic", "Strict", "refined"):  (434.7, 0.313),
    ("synthetic", "Relaxed", "refined"): (502.0, 0.276),
    ("synthetic", "BCD", "refined"):     (473.3, 0.413),
    ("synthetic", "WLG", "refined"):     (520.0, 0.342),
    ("synthetic", "Relaxed", "raw"):     (694.6, 0.185),
    ("osm", "Strict", "refined"):        (1977.5, 0.270),
    ("osm", "Relaxed", "refined"):       (1634.6, 0.155),
    ("osm", "BCD", "refined"):           (1864.3, 0.249),
    ("osm", "WLG", "refined"):           (2233.2, 0.223),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def point_segment_distance(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def path_rms(samples, waypoints) -> float:
    segments = list(zip(waypoints, waypoints[1:]))
    acc = 0.0
    for p in samples:
        acc += min(point_segment_distance(p, a, b) for a, b in segments) ** 2
    return math.sqrt(acc / len(samples))


def main() -> int:
    binding = [r for r in read_csv(LOGS / "route_binding.csv") if r["used_in_table23"] == "yes"]
    routes: "OrderedDict[tuple, list]" = OrderedDict()
    for r in binding:
        routes.setdefault((r["domain"], r["method"], r["variant"]), []).append(r)
    ok = True
    print(f"{'domain':10s} {'method':8s} {'variant':8s} {'segments':14s} {'T_exec':>8s} {'tab.':>7s}   {'RMS':>6s} {'tab.':>6s}")
    for key, segs in routes.items():
        dom, me, va = key
        t_exec = sum(float(s["fly_time_end_s"]) - float(s["fly_time_start_s"]) for s in segs)
        runs = "-".join(s["run_id"] for s in segs)
        route_file = LOGS / "routes" / f"{dom}_{me.lower()}_{va}_{runs}.csv"
        plan_file = LOGS / "planned_routes" / f"{dom}_{me.lower()}_{va}.csv"
        samples = [(float(r["east_m"]), float(r["north_m"])) for r in read_csv(route_file)]
        waypoints = [(float(r["east_m"]), float(r["north_m"])) for r in read_csv(plan_file)]
        rms = path_rms(samples, waypoints)
        t_ref, rms_ref = TABLE23[key]
        good = abs(t_exec - t_ref) < 0.05 and abs(rms - rms_ref) < 0.0005
        ok &= good
        print(f"{dom:10s} {me:8s} {va:8s} {'+'.join(s['run_id'] for s in segs):14s} {t_exec:8.1f} {t_ref:7.1f}   {rms:6.3f} {rms_ref:6.3f}"
              + ("" if good else "   <-- mismatch"))
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

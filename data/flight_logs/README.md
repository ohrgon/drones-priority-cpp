# Waypoint flight records (Section 4.6, Figure 6, Table 23)

Twelve DJI Matrice 4E flight records from 6 October 2026, the day on which the nine routes of
Table 23 were flown: the four final methods (Strict, Relaxed, BCD, WLG; refined paths) on one
selected Synthetic map (`priority_map2`) and one selected OSM-derived map (`map_02165`), plus the
Synthetic Relaxed-Raw auxiliary route. All waypoint flights used a start-point altitude of 140 m, a
waypoint speed command of 14.2 m/s and a 16 m grid spacing. The test site is in Seoul, Republic of
Korea, and is not the area from which the OSM-derived maps were built.

## Coordinate frame

Positions are given in a **local east–north frame in metres**, not as latitude/longitude. The origin
is the home point recorded in flight `2026-10-06_12-48-35` (the Figure 6a flight); the home points
of the other flights lie within 8 m of it (`flights_index.csv`, `home_east_m` / `home_north_m`).
The transform is the equirectangular approximation about the origin latitude φ₀:

```
east_m  = (lon − lon₀) · (111412.84·cos φ₀ − 93.5·cos 3φ₀ + 0.118·cos 5φ₀)
north_m = (lat − lat₀) · (111132.92 − 559.82·cos 2φ₀ + 1.175·cos 4φ₀)
```

Over the ≤ 1 km extent of the flights the distortion is below 1 cm, so distances, headings and the
path-correspondence RMS of Eq. (21) can be computed directly in this frame. The origin coordinates
are not published; they can be obtained from the corresponding author for a re-flight.

## Files

| Path | Description |
|---|---|
| `records_local/<date>_<HH-MM-SS>_aircraft_local.csv` | The complete DJI flight-record export (62 columns, 10 Hz) with the four coordinate columns replaced: `OSD.latitude/longitude` → `OSD.east_m/north_m`, `HOME.latitude/longitude` → `HOME.east_m/north_m`. All other cells are exactly as exported; units are imperial as stated in the headers (ft, mph, °F). |
| `trajectories_si_local/<date>_<HH-MM-SS>.csv` | SI subset for analysis: `fly_time_s`, UTC date/time, `east_m`, `north_m`, `height_m` (above take-off), `altitude_m`, horizontal and x/y/z speeds in m/s, yaw, flight-controller state, flight action, GNSS satellite count. |
| `route_binding.csv` | Binding of every flown route section (including repeats and the one incomplete flight) to a record and an `OSD.flyTime [s]` interval, with status, whether it enters Table 23, and the Table 23 duration. |
| `routes/<domain>_<method>_<variant>_<runs>.csv` | The recorded samples of the nine Table 23 routes, cut from the records by `route_binding.csv`. For the OSM routes, which span two or three records because of battery changes, the segments are concatenated with a `segment` number and a cumulative `route_time_s`. The repeat and incomplete flights listed in `route_binding.csv` (S02, S03, S04, S06, S09) remain inside the full records but are not cut out separately. |
| `planned_routes/<domain>_<method>_<variant>.csv` | The planned waypoint routes of the nine Table 23 rows, as uploaded to the aircraft (DJI WPML missions) and converted to the same local frame: waypoint index, `east_m`, `north_m`, execute height, waypoint speed, turn mode and heading mode. `planned_routes_index.csv` lists per route the number of waypoints, the WPML distance and duration estimates, the mission speed (14.213 m/s), height mode (relative to start point, 140 m), finish action and the take-off reference point in the local frame. The KMZ files themselves are not published because they carry absolute coordinates. For Strict, BCD and WLG the raw and refined waypoint lists are identical (connection refinement changes nothing on these paths), so only the refined routes are kept; for Relaxed the raw and refined routes differ and both were flown on the synthetic map. |
| `flights_index.csv` | One row per record: start time (KST from the file name; the export's own time stamps are UTC), samples and interval, total fly time, time in `Waypoints` mode, maximum height, home-point offset, and the run IDs it contains. |

File names carry the local start time (KST, UTC+9). The `CUSTOM.date/updateTime [local]` columns
of the export are UTC despite their label.

## Reproducing Table 23

Measured duration *T*exec (Eq. 19) is the span of the `OSD.flyTime [s]` interval of each route: it
starts at the first sample at which the horizontal speed reaches 1 m/s from rest and ends at the
first sample at which the aircraft is within 1 m of the final waypoint and the horizontal speed drops
to or below 1 m/s; turns and decelerations are included, take-off, climb, approach, waiting, return
and landing are excluded. For the OSM routes the segment spans are summed.

Path-correspondence RMS (Eq. 21) is the root mean square, over all recorded position samples of the
route, of the minimum 2-D distance from the sample to the segments of the planned waypoint polyline.
Because records and planned routes share one local frame, no coordinate transform is needed.

```bash
python scripts/table23_reproduce.py
```

prints both columns for the nine routes next to the values of Table 23; all eighteen values agree
(durations to 0.1 s, RMS to 0.001 m).

## Privacy and licensing

The export contains no serial numbers, account identifiers or personal data; `APP.tip` and
`APP.warning` hold the flight app's operational messages (partly in Korean). Absolute positions were
removed as described above. The records are released under CC BY 4.0 (see `../../LICENSE-DATA`).

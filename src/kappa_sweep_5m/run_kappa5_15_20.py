#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add kappa=5,15,20 to Table 5.14(b): fresh 5M, Validation180, common refinement.

Existing kappa=1,3,7,9,10,11,12,13,14 are NEVER trained/inferred/refined here.
The uploaded native trainer/CNN/physics and common refinement solver are reused.
Default --gpus 0 queues the three NEW cases on one GPU. All evaluation is CPU.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import kappa_extension as X
E, F = X.E, X.F
HERE = Path(__file__).resolve().parent
DEFAULT_OUT = 'chapter5_TABLE514_KAPPA5_15_20_FRESH5M_v1'
MAX_CSV = 16 * 1024 * 1024


def resolve(root: Path, value: Path) -> Path:
    value = value.expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def csv_bytes(data: bytes) -> list[dict[str, str]]:
    X.need(len(data) <= MAX_CSV, 'Existing result CSV exceeds allowed size')
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    X.need(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'Invalid CSV header')
    rows = list(reader)
    X.need(rows and all(None not in r for r in rows), 'Empty/malformed existing result CSV')
    return rows


def existing_payload(path: Path) -> tuple[list[dict[str, str]], dict]:
    """Read small saved result tables only; no checkpoint loading, no extraction."""
    original = path
    if path.is_dir():
        options = [path / 'report/per_map_nine_costs.csv', path / 'per_map_nine_costs.csv',
                   path / 'report/cost1_fresh_per_map.csv', path / 'cost1_fresh_per_map.csv',
                   path / 'TABLE514_KAPPA1_FRESH5M_REVIEW.zip']
        found = [p for p in options if p.is_file()]
        X.need(found, 'No saved kappa1/nine-condition result found under: ' + str(path))
        path = found[0]
    X.need(path.is_file(), 'Existing result not found: ' + str(path))
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            options = [(0 if i.filename.endswith('per_map_nine_costs.csv') else 1, i.filename, i)
                       for i in z.infolist() if not i.is_dir() and
                       (i.filename.endswith('/per_map_nine_costs.csv') or i.filename == 'per_map_nine_costs.csv' or
                        i.filename.endswith('/cost1_fresh_per_map.csv') or i.filename == 'cost1_fresh_per_map.csv')]
            X.need(options, 'Review ZIP lacks saved per-map kappa1/nine-condition records')
            best = min(i[0] for i in options); found = [i for i in options if i[0] == best]
            X.need(len(found) == 1, 'Ambiguous existing result members; supply a specific CSV')
            member = found[0][2]; p = PurePosixPath(member.filename)
            X.need(not p.is_absolute() and '..' not in p.parts and member.file_size <= MAX_CSV, 'Unsafe/oversized CSV member')
            data = z.read(member)
        source = dict(path=str(original), resolved_file=str(path), zip_member=member.filename,
                      sha256=E.sha(path), representation='saved_review_csv')
    else:
        X.need(path.suffix.lower() == '.csv' and path.stat().st_size <= MAX_CSV, 'Use a CSV, review ZIP or result folder')
        data = path.read_bytes()
        source = dict(path=str(original), resolved_file=str(path), sha256=E.sha(path), representation='saved_permap_csv')
    return csv_bytes(data), source


def validated_existing(rows: list[dict]) -> list[dict]:
    costs = tuple(sorted({int(float(r['cost'])) for r in rows}))
    X.need(costs in ((1,), X.EXISTING), 'Need kappa1 alone or the completed nine-condition per-map table; not an aggregate')
    X.check_rows(rows, costs)
    original_eight = F.refs()[0]
    if costs == X.EXISTING:
        lookup = {(int(float(r['cost'])), r['map_id']): r for r in original_eight}
        for r in rows:
            c = int(float(r['cost']))
            if c == 1:
                continue
            b = lookup[(c, r['map_id'])]
            X.need(E.flag(r['is_success']) == E.flag(b['is_success']), 'Existing eight-cost success changed')
            for key in ('steps', 'active_cells', 'coverage_pct', E.T, 'original_Lw_sec',
                        'witness_steps', 'witness_time_sec', 'witness_Lw_sec'):
                E.near(r[key], b[key], 'Existing eight-cost result differs: ' + key)
    # Wide nine-cost CSVs add blank columns that are absent from the kappa1-only
    # CSV. Remove only redundant blank columns; preserve undefined Lw fields.
    keep_blank = {'original_Lw_sec', 'witness_Lw_sec'}
    own = [{key: value for key, value in r.items() if value not in (None, '') or key in keep_blank}
           for r in rows if int(float(r['cost'])) == 1]
    for r in own:
        X.need(r.get('experiment_origin') == 'fresh_rerun_not_recovered_historical_checkpoint',
               'Expected the completed uploaded-package kappa1 fresh rerun, not historical aggregates')
    X.need(len({r['checkpoint_sha256'] for r in own}) == 1, 'Multiple kappa1 checkpoints mixed')
    return sorted(own, key=lambda r: r['map_id'])


def freeze_existing(root: Path, out: Path, explicit: Path | None) -> tuple[list[dict], dict]:
    """Known local paths only. Missing kappa1 never triggers retraining or fabricated rows."""
    ref = out / 'existing_reference'; frozen = ref / 'kappa1_per_map.csv'; meta = ref / 'source.json'
    if frozen.is_file() or meta.is_file():
        X.need(frozen.is_file() and meta.is_file(), 'Incomplete saved reference snapshot')
        saved = E.read(meta); X.need(E.sha(frozen) == saved['frozen_csv_sha256'], 'Saved kappa1 reference changed')
        own = validated_existing(E.rows(frozen))
        if explicit is not None:
            rows, supplied = existing_payload(resolve(root, explicit))
            newer = validated_existing(rows)
            X.need(E.digest(newer) == saved['canonical_rows_sha256'], 'A different kappa1 result cannot replace the frozen source')
        return own, saved
    candidates = [resolve(root, explicit)] if explicit is not None else [
        root / 'chapter5_TABLE514_KAPPA1_FRESH5M_v1',
        root / 'TABLE514_KAPPA1_FRESH5M_REVIEW.zip',
        root / 'TABLE514_KAPPA1_FRESH5M_REVIEW(1).zip',
    ]
    found = []
    for p in candidates:
        if not p.exists():
            if explicit is not None:
                raise RuntimeError('Explicit --existing-nine not found: ' + str(p))
            continue
        if p.is_dir() and not any((p / n).is_file() for n in
                ('report/per_map_nine_costs.csv', 'per_map_nine_costs.csv', 'report/cost1_fresh_per_map.csv',
                 'cost1_fresh_per_map.csv', 'TABLE514_KAPPA1_FRESH5M_REVIEW.zip')):
            continue
        rows, source = existing_payload(p); own = validated_existing(rows)
        found.append((own, source))
    if not found:
        return [], dict(status='KAPPA1_REFERENCE_MISSING', known_paths=[str(p) for p in candidates],
                        action='Only new kappa5/15/20 may run; the full twelve-condition table remains pending')
    signatures = {E.digest(own) for own, source in found}
    X.need(len(signatures) == 1, 'Different kappa1 results found; specify --existing-nine explicitly')
    own, source = found[0]
    ref.mkdir(exist_ok=True); E.csvwrite(frozen, own)
    source.update(status='FROZEN_SAVED_RESULT', canonical_rows_sha256=E.digest(own),
                  frozen_csv_sha256=E.sha(frozen), rows=180, inference_executed=False, refinement_executed=False)
    E.write(meta, source)
    return own, source


def make_plan(args) -> dict:
    p = args.out / 'plan.json'
    packages = X.verify_package()
    if p.is_file():
        plan = E.read(p)
        X.need(plan['version'] == X.VERSION and plan['package_sha256'] == packages, 'Run/package version changed')
        X.need(plan['root'] == str(args.root) and plan['out'] == str(args.out), 'Use original --root/--out')
        if args.phase in ('all', 'train', 'check'):
            X.need(plan['gpus'] == args.gpus and plan['device'] == args.device, 'Resume with the same GPU/device assignment')
            expected_dataset = str(resolve(args.root, args.dataset_root)) if args.dataset_root else None
            X.need(plan['dataset_root'] == expected_dataset, 'Dataset option changed on resume')
        return plan
    X.need(args.phase in ('all', 'train', 'check'), 'No plan exists. Run --phase all or train first')
    allowed = {'.launcher.lock'}
    X.need(not any(p.name not in allowed for p in args.out.iterdir()), 'Use a new empty output directory')
    devices = ['cpu'] if args.device == 'cpu' else [f'cuda:{g}' for g in args.gpus]
    plan = dict(version=X.VERSION, root=str(args.root), out=str(args.out),
                dataset_root=str(resolve(args.root, args.dataset_root)) if args.dataset_root else None,
                gpus=args.gpus, device=args.device, package_sha256=packages,
                fresh_costs=list(X.ADDED), reused_costs=list(X.EXISTING), final_costs=list(X.ALL_COSTS),
                target_steps=X.TARGET, final_optimizer_boundary=X.FINAL_STEP, train_maps=900,
                validation_maps=180, test_maps=0, osm_maps=0,
                jobs={str(c): dict(kappa=c, device=devices[i % len(devices)], case_dir=str(args.out / f'cost{c}'))
                      for i, c in enumerate(X.ADDED)})
    E.write(p, plan); return plan


def worker(plan: dict, kappa: int, phase: str, workers: int) -> None:
    X.verify_package()
    X.need(plan['package_sha256'] == E.read(HERE / 'PACKAGE_SHA256.json'), 'Worker package/plan mismatch')
    X.configure(kappa)
    job = plan['jobs'][str(kappa)]; out = Path(job['case_dir']); root = Path(plan['root'])
    out.mkdir(parents=True, exist_ok=True)
    with E.lock(out / '.launcher.lock'):
        if phase in ('train', 'check'):
            pr = F.prepare(out, root, job['device'], Path(plan['dataset_root']) if plan['dataset_root'] else None)
            status = out / 'runs' / X.run_id(kappa) / 'status.json'
            if phase == 'train' and status.is_file() and E.read(status).get('state') == 'completed':
                X.source_context(out)
                print(f'[TRAIN SKIP COST{kappa}] completed final {X.FINAL_STEP:,}; no validation or retraining.', flush=True)
                return
            F.train(out, pr, check_only=(phase == 'check'))
            if phase == 'train' and not (out / 'report/complete.json').exists():
                F.report(out)
        elif phase == 'refine':
            # --phase refine includes inference of ONLY missing validation Raw paths.
            # It never sees any test/OSM record and never loads existing kappa weights.
            F.evaluate(out, workers)
        else:
            raise RuntimeError('Unsupported internal phase')


def launch(plan: dict, phase: str, workers: int) -> None:
    pending = list(X.ADDED); active = {}; failed = []
    logs = Path(plan['out']) / 'logs'; logs.mkdir(exist_ok=True)
    def pump(proc, logfile, kappa):
        with logfile.open('a', encoding='utf-8') as log:
            for line in proc.stdout:
                log.write(line); log.flush()
                print(f'[κ={kappa}] ' + line, end='', flush=True)
        proc.stdout.close()
    try:
        while pending or active:
            busy = {info['slot'] for info in active.values()}
            for c in list(pending):
                job = plan['jobs'][str(c)]
                slot = job['device'] if phase in ('train', 'check') else 'CPU_REFINE'
                if slot in busy:
                    continue
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
                if phase == 'refine':
                    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--_worker', phase,
                           '--_plan', str(Path(plan['out']) / 'plan.json'), '--_kappa', str(c), '--workers', str(workers)]
                proc = subprocess.Popen(command, cwd=plan['root'], env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                        bufsize=1, start_new_session=True)
                logfile = logs / f'kappa{c}_{phase}.log'
                thread = threading.Thread(target=pump, args=(proc, logfile, c), daemon=True); thread.start()
                active[proc] = dict(kappa=c, slot=slot, thread=thread); pending.remove(c); busy.add(slot)
                print(f'[LAUNCH] COST{c} {phase} {slot} pid={proc.pid}; log={logfile}', flush=True)
            for proc, info in list(active.items()):
                code = proc.poll()
                if code is not None:
                    info['thread'].join()
                    print(f'[EXIT] COST{info["kappa"]} {phase}: {code}', flush=True)
                    if code:
                        failed.append((info['kappa'], code))
                    del active[proc]
            if failed:
                raise RuntimeError('Worker stopped: ' + str(failed) + '. Completed checkpoints and map caches are retained.')
            if active:
                time.sleep(0.25)
    except BaseException:
        for proc in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
        for proc, info in active.items():
            proc.wait(); info['thread'].join()
        raise


def complete_case(plan: dict, kappa: int) -> list[dict]:
    out = Path(plan['jobs'][str(kappa)]['case_dir']); report = out / 'report'; marker = report / 'complete.json'
    if not marker.is_file():
        return []
    cm = E.read(marker)
    X.need(cm['version'] == X.VERSION and int(cm['kappa']) == kappa, 'Wrong completed case')
    for name, digest in cm['files'].items():
        X.need(E.sha(E.member(report, name)) == digest, 'Completed case table changed')
    job = X.source_context(out)
    identity = E.digest(dict(version=X.VERSION, checkpoint=job['checkpoint_sha256'], step=job['checkpoint_step'],
                  spec=job['spec'], map_members=[(r['map_id'], r['sha256']) for r in job['records']],
                  initial_seed=job['evaluation']['initial_seed'], engine_sha256=E.sha(X.UP/'engine/ch5_common.py'),
                  solver_sha256=E.sha(X.UP/'engine/ch5_vendor/segment_graph_original.py'), native=job['hashes']))
    X.need(cm['identity'] == identity, 'Completed case evaluation identity differs from final source')
    rr = E.rows(report / 'per_map.csv'); X.check_rows(rr, (kappa,), strict_new=True)
    X.need(all(r['checkpoint_sha256'] == job['checkpoint_sha256'] for r in rr), 'Another checkpoint in case summary')
    for row in rr:
        d = F.mapdir(out, row)
        m = F.committed(d, 'complete.json', cm['identity'])
        X.need(m is not None and m['map_id'] == row['map_id'], 'Per-map completed cache missing')
        summary = E.read(d / 'summary.json')
        for key in ('map_sha256', 'initial_pair_id', 'checkpoint_sha256'):
            X.need(row[key] == summary[key], 'Reported map provenance differs: ' + key)
        for key in (E.T, 'original_Lw_sec', 'witness_time_sec', 'witness_Lw_sec'):
            E.near(row[key], summary[key], 'Reported/committed value mismatch')
    return rr


def markdown_table(title: str, rows: list[dict]) -> str:
    text = f'# {title}\n\n| κ | 성공/180 | 공통 n | Raw T (s) | Raw Lw (s) | Refined T (s) | Refined Lw (s) | 출처 |\n'
    text += '|---|---|---|---|---|---|---|---|\n'
    def f(v):
        return '—' if v is None else f'{v:,.2f}'
    for r in rows:
        text += (f'| {r["kappa"]} | {r["success_n"]}/180 | {r["common_n"]} | {f(r["Raw_T_sec"])} | '
                 f'{f(r["Raw_Lw_sec"])} | {f(r["Refined_T_sec"])} | {f(r["Refined_Lw_sec"])} | {r["origin"]} |\n')
    text += '\n성공률은 전체 Validation 180개, 시간 평균은 표에 포함한 모든 κ가 성공한 동일 지도 교집합 기준이다. '
    text += '기존 163/167을 고정하지 않으며, 공통 성공이 없으면 시간 평균은 미정의(—)이다.\n'
    return text


def build_report(plan: dict, existing: list[dict], source: dict) -> dict:
    out = Path(plan['out']); rep = out / 'report'; rep.mkdir(exist_ok=True)
    eight, hashes, old_table, old_members = F.refs()
    E.csvwrite(rep / 'existing_eight_RAW_REFINED.csv', old_table)
    E.atomic(rep / 'table_5_14a_historical_RAW.csv',
             (X.UP / 'reference/summary_costs_matched_extracted.csv').read_text(encoding='utf-8-sig'))
    complete = {c: complete_case(plan, c) for c in X.ADDED}
    own = [r for c in X.ADDED for r in complete[c]]
    E.csvwrite(rep / 'per_map_added_costs.csv', own, fields=None if own else ['cost', 'map_id'])
    if all(complete.values()):
        tab, members = X.table(own, X.ADDED)
        E.csvwrite(rep / 'added_three_common_RAW_REFINED.csv', tab)
        E.csvwrite(rep / 'added_three_common_success_maps.csv', members, fields=['map_id', 'map_sha256', 'initial_pair_id'])
        E.atomic(rep / 'Added_Three_KO.md', markdown_table('추가 κ=5·15·20만의 3조건 공통집합', tab))
    status = 'COMPLETE' if existing and all(complete.values()) else 'PARTIAL'
    audit = dict(version=X.VERSION, status=status, planned_fresh_kappas=list(X.ADDED),
                 completed_added_maps={str(c): len(r) for c, r in complete.items()},
                 reused_kappas=list(X.EXISTING if existing else X.EXISTING[1:]),
                 reused_maps=len(eight) + len(existing), existing_source=source,
                 existing_reference_hashes=hashes, training_M=5, validation_maps_per_kappa=180,
                 new_test_rollouts=0, new_osm_rollouts=0, existing_kappas_training=False,
                 existing_kappas_inference=False, existing_kappas_refinement=False)
    if status == 'COMPLETE':
        combined = existing + eight + own
        tab, members = X.table(combined, X.ALL_COSTS)
        E.csvwrite(rep / 'table_5_14b_twelve_costs_RAW_REFINED.csv', tab)
        E.csvwrite(rep / 'twelve_common_success_maps.csv', members, fields=['map_id', 'map_sha256', 'initial_pair_id'])
        E.csvwrite(rep / 'per_map_twelve_costs.csv', combined)
        E.atomic(rep / 'Table_5_14b_KO.md', markdown_table('Table 5.14(b): κ≤20의 12조건 Raw/Refined 비교', tab))
        audit.update(common_n=len(members), total_per_map_rows=len(combined),
                     successful_maps={str(r['kappa']): r['success_n'] for r in tab})
    else:
        E.atomic(rep / 'STATUS_KO.md', '# 처리 상태: PARTIAL\n\n' +
                 '\n'.join(f'- κ={c}: {len(r)}/180 완료' for c, r in complete.items()) +
                 f'\n- 기존 κ=1: {len(existing)}/180\n\n12조건이 모두 확보되기 전에는 최종 Table 5.14(b)를 확정하지 않는다. '
                 '누락은 성공 0이 아니며, 기존 조건을 새로 학습하지 않는다.\n')
    E.write(rep / 'audit.json', audit)
    make_review(plan, own)
    print(f'[REPORT] {status}; new Raw/Refined maps={len(own)}/540; existing={len(eight)+len(existing)}/1620', flush=True)
    print('[REVIEW ZIP]', out / 'TABLE514_KAPPA5_15_20_REVIEW.zip', flush=True)
    return audit


def make_review(plan: dict, own: list[dict]) -> None:
    out = Path(plan['out']); path = out / 'TABLE514_KAPPA5_15_20_REVIEW.zip'; temp = path.with_suffix('.writing.zip')
    included = set()
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        def add(p, arc):
            if p.is_file() and arc not in included:
                z.write(p, arc); included.add(arc)
        add(out / 'plan.json', 'plan.json')
        for directory in ('report', 'existing_reference'):
            for p in sorted((out / directory).glob('*')):
                add(p, p.relative_to(out).as_posix())
        for c in X.ADDED:
            case = out / f'cost{c}'; run = case / 'runs' / X.run_id(c)
            for name in ('protocol.json', 'manifest.csv'):
                add(case / name, f'cost{c}/' + name)
            for name in ('run_spec.json', 'training_settings.json', 'status.json', 'checkpoints.json',
                         'training_updates.csv', 'software.json', 'network_observation_audit.json'):
                add(run / name, (run / name).relative_to(out).as_posix())
            for directory in ('audit', 'report'):
                for p in sorted((case / directory).glob('*')):
                    add(p, p.relative_to(out).as_posix())
            chosen = []
            for ok in (True, False):
                subset = [r for r in own if int(float(r['cost'])) == c and E.flag(r['is_success']) == ok]
                subset.sort(key=lambda r: (not E.flag(r.get('path_changed', False)), r['map_id']))
                chosen += subset[:2]
            for row in chosen:
                d = F.mapdir(case, row)
                for p in sorted(d.iterdir()):
                    add(p, f'samples/cost{c}/' + d.name + '/' + p.name)
            for kind in ('train', 'refine', 'check'):
                log = out / 'logs' / f'kappa{c}_{kind}.log'
                if log.is_file():
                    with log.open('rb') as f:
                        f.seek(max(0, log.stat().st_size - 65536)); tail = f.read()
                    z.writestr(f'logs/kappa{c}_{kind}_tail.txt', tail)
        for name in ('run_kappa5_15_20.py', 'kappa_extension.py', 'README_KO.md', 'PACKAGE_SHA256.json'):
            add(HERE / name, 'program/' + name)
    with zipfile.ZipFile(temp) as z:
        X.need(z.testzip() is None, 'Result ZIP integrity failed')
    os.replace(temp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('.'), help='Existing uav_chapter4_suite_v2 directory')
    ap.add_argument('--out', type=Path, help='New output directory (never the original kappa1 output)')
    ap.add_argument('--dataset-root', type=Path, help='Only if .npy maps moved; directory containing original map IDs')
    ap.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU IDs. One ID queues the three new models sequentially')
    ap.add_argument('--device', choices=['cpu'], help='Optional CPU training')
    ap.add_argument('--workers', type=int, default=3, help='CPU validation/refinement workers; one kappa at a time')
    ap.add_argument('--phase', choices=['all', 'check', 'train', 'refine', 'report'], default='all')
    ap.add_argument('--existing-nine', type=Path, help='Saved kappa1/nine-cost CSV, original output directory, or review ZIP')
    ap.add_argument('--_worker', choices=['train', 'check', 'refine'], help=argparse.SUPPRESS)
    ap.add_argument('--_plan', type=Path, help=argparse.SUPPRESS)
    ap.add_argument('--_kappa', type=int, choices=X.ADDED, help=argparse.SUPPRESS)
    args = ap.parse_args()
    X.need(args.workers >= 1 and args.gpus and min(args.gpus) >= 0 and len(args.gpus) == len(set(args.gpus)), 'Invalid worker/GPU count')
    if args._worker:
        X.need(args._plan is not None and args._kappa is not None, 'Internal worker arguments missing')
        worker(E.read(args._plan), args._kappa, args._worker, args.workers); return 0
    args.root = args.root.expanduser().resolve()
    args.out = resolve(args.root, args.out or Path(DEFAULT_OUT))
    X.need(args.root.is_dir() and args.out != args.root and args.out not in args.root.parents, 'Invalid suite/output path')
    X.need(not overlaps(args.out, HERE), 'Do not write results over/inside the supplied code package')
    old_default = args.root / 'chapter5_TABLE514_KAPPA1_FRESH5M_v1'
    X.need(not overlaps(args.out, old_default), 'Do not overwrite the original kappa1 output')
    if args.existing_nine:
        source_path = resolve(args.root, args.existing_nine)
        X.need(not overlaps(args.out, source_path), 'Existing results must be read-only and separate')
    args.out.mkdir(parents=True, exist_ok=True)
    with E.lock(args.out / '.launcher.lock'):
        plan = make_plan(args)
        existing, source = freeze_existing(args.root, args.out, args.existing_nine)
        for c in X.ADDED:
            print(f'[PLAN] κ={c}: fresh 0 -> 5M; {plan["jobs"][str(c)]["device"]}; Validation180 -> common refinement', flush=True)
        print('[REUSE] Existing κ=1,3,7,9,10,11,12,13,14: no training/inference/refinement.', flush=True)
        print('[SCOPE] Train900 + Validation180 only; Test=0; OSM=0; final fixed 5M evaluation.', flush=True)
        if not existing:
            print('[REFERENCE PENDING] Saved κ=1 records not found at the known paths. New 5/15/20 will still run. '
                  'Use --existing-nine later for the twelve-condition table; κ=1 will NOT be retrained.', flush=True)
        try:
            if args.phase in ('all', 'train', 'check'):
                launch(plan, 'check' if args.phase == 'check' else 'train', args.workers)
            if args.phase in ('all', 'refine'):
                launch(plan, 'refine', args.workers)
        finally:
            result = build_report(plan, existing, source)
        # Partial after train/check is expected. Final/report must clearly signal missing data.
        return 0 if args.phase in ('train', 'check') or result['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('[STOP] Checkpoints/recovery and completed per-map caches are retained. Rerun the same command.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print('[ERROR]', type(exc).__name__, str(exc), file=sys.stderr)
        traceback.print_exc(); raise SystemExit(1)

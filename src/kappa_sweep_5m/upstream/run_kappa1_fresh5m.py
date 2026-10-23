#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fresh COST1 only: train 5M -> Validation180 -> common refinement -> nine-cost table.

No search for old COST1 checkpoints; no other cost is trained/inferred/refined.
The original COST1 core/CNN and original trainer transformation are bundled.
The new kappa=1 run is always marked fresh_rerun, not the recovered historical run.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib, io, json, math, os, shutil, sys, tempfile, time, traceback, zipfile
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE/'engine'))
import run_table514_kappa1 as E

VERSION='kappa1_fresh5m_only_v1_20261005'
OUT_NAME='chapter5_TABLE514_KAPPA1_FRESH5M_v1'
EXPERIMENT_ORIGIN='fresh_rerun_not_recovered_historical_checkpoint'
RUN_ID=E.RUN_ID
COSTS=E.COSTS
NATIVE_FILES=tuple(sorted(p.name for p in (HERE/'native').glob('*.py')))
RT=JOB=None


def need(ok,message):
    if not ok:raise RuntimeError(message)

def verify_package():
    expected=E.read(HERE/'PACKAGE_SHA256.json')
    for name,h in expected.items():
        p=E.member(HERE,name)
        need(p.is_file() and E.sha(p)==h,'Package file missing/changed: '+name)
    return expected

def refs():
    rr,hashes=E.load_other_eight(HERE,[HERE/'reference/cost_3_7_9_11.csv',HERE/'reference/cost_10_12_13_14.csv'])
    table,members=E.paired_table(rr,required=COSTS[1:])
    need(len(rr)==1440 and len(members)==167,'Bundled eight-cost input differs')
    return rr,hashes,table,members

def make_request(device):
    spec=E.read(HERE/'reference/fresh_spec.json')
    settings=E.read(HERE/'reference/training_settings.json');settings['device']=device
    evaluation=E.read(HERE/'reference/evaluation.json')
    need(spec['reward_definition']['cost_multiplier']==1,'Not COST1')
    return dict(version=E.NATIVE_VERSION,launcher_version=VERSION,experiment_origin=EXPERIMENT_ORIGIN,
                spec=spec,settings=settings,evaluation=evaluation,eval_device='cpu',
                warm_start=False,old_checkpoint_required=False,test_enabled=False,
                other_costs_training=False,other_costs_refinement=False,
                settings_source='uploaded original 5M cost sweep; original COST1 core/feature extractor',
                reference_settings_sha256=E.sha(HERE/'reference/training_settings.json'),
                original_manifest_array_membership='reference/train_validation_manifest.csv',
                original_165_success_is_not_an_acceptance_target=True)

def resolve_maps(root,dataset_root=None):
    rows=E.rows(HERE/'reference/train_validation_manifest.csv')
    need(Counter(r['split'] for r in rows)=={'train':900,'validation':180},'Wrong frozen split counts')
    sys.path.insert(0,str(HERE/'native'))
    from coverage_core import validate_priority_map
    from full_v2_utils import array_hash
    missing=[];resolved=[]
    for row in rows:
        r=dict(row);mid=Path(r['map_id'])
        need(not mid.is_absolute() and '..' not in mid.parts,'Unsafe map_id')
        candidates=([dataset_root/mid] if dataset_root else [])+[Path(r['path']),root.parent/mid,root/mid]
        path=next((p.resolve() for p in candidates if p.is_file()),None)
        if path is None:
            missing.append(dict(map_id=r['map_id'],split=r['split'],expected_original_path=r['path']));continue
        arr=validate_priority_map(np.load(path,allow_pickle=False),50,min_active=2)
        need(array_hash(arr)==r['sha256'],'Map content differs; not replaced: '+str(path))
        r['path']=str(path);resolved.append(r)
    need(not missing,'Dataset .npy missing (%d). First: %s. Use --dataset-root (directory containing train/ and test/); no model search is needed.'%(len(missing),missing[:1]))
    need(len({r['map_id'] for r in resolved})==1080 and len({r['sha256'] for r in resolved})==1080,'Duplicate split IDs/content')
    return resolved

def install_snapshot(out):
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    hashes={}
    for name in NATIVE_FILES:
        source=HERE/'native'/name;dest=snap/name
        if dest.exists():need(E.sha(dest)==E.sha(source),'Existing snapshot changed: '+name)
        else:shutil.copy2(source,dest)
        hashes[name]=E.sha(dest)
    return hashes

def import_native(snapshot):
    snapshot=Path(snapshot).resolve();sys.path.insert(0,str(snapshot));importlib.invalidate_caches()
    mods={}
    for name in ['coverage_core','core_ch4','core_ch4_bounded_b6','full_v2_utils','resume_ch4',
                 'nrg1_c3nt_core','nrg1_c3nt_features','trainers_ch4_bounded_b6_nrg1_c3nt',
                 'run_nrg1_c3nt','run_nrg1_c3nt_logfix']:
        old=sys.modules.get(name)
        if old is not None and Path(old.__file__).resolve().parent!=snapshot:del sys.modules[name]
        m=importlib.import_module(name)
        need(Path(m.__file__).resolve().parent==snapshot,'Native import escaped snapshot: '+name)
        mods[name]=m
    t=mods['trainers_ch4_bounded_b6_nrg1_c3nt'];u=mods['full_v2_utils'];fix=mods['run_nrg1_c3nt_logfix']
    if not getattr(t,'_fresh_logging_patch',False):
        native=t.CSVLog
        need(native is u.CSVLog,'Unexpected pre-existing CSV logger patch')
        def logger(path,*a,**kw):
            obj=native(path,*a,**kw)
            return fix.EpisodeCSVCompatibility(obj,Path(path)) if Path(path).name=='training_episodes.csv' else obj
        t.CSVLog=logger;t._fresh_logging_patch=True
    return mods

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,s):
        for stream in self.streams:stream.write(s);stream.flush()
        return len(s)
    def flush(self):
        for stream in self.streams:stream.flush()
    def isatty(self):return False


def prepare(out,root,device,dataset_root):
    request=make_request(device);pp=out/'protocol.json'
    if pp.exists():
        p=E.read(pp)
        need(p.get('request')==request,'Frozen run settings changed. Resume with the same device/options, or use a new --out.')
        need(E.sha(out/'manifest.csv')==p['manifest_sha256'],'Frozen manifest changed')
        for name,h in p['snapshot_hash'].items():need(E.sha(out/'source_snapshot'/name)==h,'Native snapshot changed')
        rows=E.rows(out/'manifest.csv')
    else:
        need(not (out/'runs').exists(),'Unrecognized run output; do not overwrite')
        rows=resolve_maps(root,dataset_root)
        hashes=install_snapshot(out);E.csvwrite(out/'manifest.csv',rows)
        p=dict(request=request,manifest_sha256=E.sha(out/'manifest.csv'),snapshot_hash=hashes,
               counts={'train':900,'validation':180,'test':0})
        E.write(pp,p)
    va=[r for r in rows if r['split']=='validation']
    other=refs()[0];old={r['map_id']:r for r in other if int(float(r['cost']))==3}
    need({r['map_id']:r['sha256'] for r in va}=={k:v['map_sha256'] for k,v in old.items()},'Validation membership differs from existing eight costs')
    # Same deterministic starting conditions as the original cost diagnostics.
    job=dict(snapshot=str(out/'source_snapshot'),spec=request['spec'],evaluation=request['evaluation'],
             native_version=E.NATIVE_VERSION,hashes=p['snapshot_hash'])
    rt=E.load_native(job)
    for r in va:
        path=Path(r['path']);need(path.is_file(),'Validation map disappeared: '+str(path))
        arr=np.load(path,allow_pickle=False)
        need(rt.utils.array_hash(arr)==r['sha256'],'Validation map changed: '+r['map_id'])
        ic=E.initial(rt,r,arr,request['evaluation']['initial_seed'])
        need(ic['pair_id']==old[r['map_id']]['initial_pair_id'],'Initial condition differs: '+r['map_id'])
    E.write(out/'audit/validation_membership.json',dict(validation_maps=180,matched_other_eight=True,initial_pairs_checked=180))
    return p


def train(out,protocol,check_only=False):
    snap=out/'source_snapshot';mods=import_native(snap)
    t=mods['trainers_ch4_bounded_b6_nrg1_c3nt'];u=mods['full_v2_utils'];runner=mods['run_nrg1_c3nt']
    request=protocol['request'];spec=deepcopy(request['spec']);spec['nrg1_protocol_sha256']=E.sha(out/'protocol.json')
    st=request['settings'];dev=st['device']
    runner.software_check()
    import torch
    if dev.startswith('cuda:'):
        idx=int(dev.split(':')[1]);need(torch.cuda.is_available() and idx<torch.cuda.device_count(),'Requested GPU unavailable: '+dev)
    # Native load_manifest verifies every train/validation map; it creates no split.
    records=u.load_manifest(out/'manifest.csv',50)
    rt=(t,u,None,mods['nrg1_c3nt_core'],mods['nrg1_c3nt_features'],None)
    with tempfile.TemporaryDirectory(prefix='kappa1_fresh_check_') as td:
        checks=runner.preflight(rt,spec,st,records,td)
        checks['episode_csv']=mods['run_nrg1_c3nt_logfix'].verify_csv_roundtrip(
            u.CSVLog,mods['run_nrg1_c3nt_logfix'].episode_rows_for_check(t,spec),Path(td)/'csv_probe')
    E.write(out/'audit/preflight.json',checks)
    if check_only:
        print('[CHECK PASS] Original COST1 core/CNN/PPO and exact 900/180 split; zero optimizer updates.');return
    print('[TRAIN] COST1 only; fresh 0 -> 5M (or resume this NEW run); no COST11 weights loaded.',flush=True)
    run=out/'runs'/RUN_ID
    status=t.train_run(spec,st,[r for r in records if r.split=='train'],run)
    need(status.get('state')=='completed','Fresh COST1 training did not complete')
    need(int(status['actual_timesteps'])==5_005_312,'Unexpected final 5M update boundary')
    E.write(out/'audit/fresh_training_complete.json',dict(origin=EXPERIMENT_ORIGIN,checkpoint_step=status['actual_timesteps'],
            checkpoint_sha256=E.sha(run/'last_model.zip'),other_costs_retrained=False,old_cost1_success_not_enforced=True))
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    print('[TRAIN COMPLETE] final fixed 5M; next: Validation180 and refinement only.',flush=True)


def eval_job(out):
    job=E.source_context(out,explicit=out)
    # E's NATIVE_VERSION check is an engine identity check. This is a NEW run.
    pr=E.read(out/'protocol.json')
    need(pr['request']['experiment_origin']==EXPERIMENT_ORIGIN,'Expected newly trained COST1 experiment')
    cp=Path(job['checkpoint_path'])
    need(cp.is_file() and E.sha(cp)==job['checkpoint_sha256'],'Fresh checkpoint missing/changed')
    job['experiment_origin']=EXPERIMENT_ORIGIN
    identity=E.digest(dict(version=VERSION,checkpoint=job['checkpoint_sha256'],step=job['checkpoint_step'],
                 spec=job['spec'],map_members=[(r['map_id'],r['sha256']) for r in job['records']],
                 initial_seed=job['evaluation']['initial_seed'],engine_sha256=E.sha(HERE/'engine/ch5_common.py'),
                 solver_sha256=E.sha(HERE/'engine/ch5_vendor/segment_graph_original.py'),
                 native=job['hashes']))
    return job,identity

def worker_init(job):
    global JOB,RT
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    JOB=job;RT=E.load_native(job);E.MODEL=None

def mapdir(out,rec):return Path(out)/'validation_refinement/maps'/hashlib.sha256(rec['map_id'].encode()).hexdigest()[:24]

def committed(d,name,ident):
    p=d/name
    if not p.exists():return None
    c=E.read(p);need(c['identity']==ident,'Stale cache identity: '+str(d))
    for fname,h in c['files'].items():
        f=E.member(d,fname);need(f.is_file() and E.sha(f)==h,'Corrupt cache: '+str(f))
    return c

def save_npz(path,trace):
    temp=path.with_name(path.stem+'.writing.npz');np.savez_compressed(temp,**trace);os.replace(temp,path)

def process(task):
    rec,out,ident=task;d=mapdir(out,rec)
    if committed(d,'complete.json',ident):return E.read(d/'summary.json')
    d.mkdir(parents=True,exist_ok=True)
    job=JOB;rt=RT
    arr=np.load(rec['path'],allow_pickle=False);need(rt.utils.array_hash(arr)==rec['sha256'],'Map changed')
    ic=E.initial(rt,rec,arr,job['evaluation']['initial_seed'])
    if committed(d,'raw_complete.json',ident):
        raw=E.read(d/'raw_metrics.json')
        with np.load(d/'raw_trace.npz',allow_pickle=False) as z:tr={k:z[k].copy() for k in z.files}
        need(raw['initial_pair_id']==ic['pair_id'],'Cached initial condition differs')
    else:
        raw,tr=E.run_episode(rt,job,rec,arr,ic)
        E.write(d/'raw_metrics.json',raw);save_npz(d/'raw_trace.npz',tr)
        E.write(d/'raw_complete.json',dict(identity=ident,files={n:E.sha(d/n) for n in ('raw_metrics.json','raw_trace.npz')}))
    refined,rtrace,audit,segs=E.C.common_refine(rt,raw,tr)
    summary=dict(cost=1,map_id=rec['map_id'],map_sha256=rec['sha256'],initial_pair_id=ic['pair_id'],source='synthetic',split='validation',
        tier=rec['tier'],pattern=rec['pattern'],is_success=E.flag(raw['is_success']),active_cells=int(raw['active_cells']),
        steps=int(raw['steps']),coverage_pct=raw['coverage_pct'],mission_time_sec=raw[E.T],original_Lw_sec=raw[E.LW],
        witness_time_sec=refined[E.T],witness_Lw_sec=refined[E.LW],witness_steps=int(refined['steps']),
        actual_source_replay_verified=True,native_witness_replay_verified=True,
        witness_respects_original_execution_cap=int(refined['steps'])<=int(raw['steps']),
        checkpoint_step=job['checkpoint_step'],checkpoint_sha256=job['checkpoint_sha256'],experiment_origin=EXPERIMENT_ORIGIN,
        historical_kappa1_reproduced=False,path_changed=bool(audit.get('path_changed',False)))
    need(E.flag(raw['is_success'])==E.flag(refined['is_success']),'Refinement changed success')
    E.write(d/'summary.json',summary);E.write(d/'refined_metrics.json',refined);E.write(d/'refinement_audit.json',audit)
    save_npz(d/'refined_trace.npz',rtrace);E.csvwrite(d/'segments.csv',segs,fields=None if segs else ['segment'])
    names=('summary.json','raw_metrics.json','raw_trace.npz','raw_complete.json','refined_metrics.json','refined_trace.npz','refinement_audit.json','segments.csv')
    E.write(d/'complete.json',dict(identity=ident,map_id=rec['map_id'],files={n:E.sha(d/n) for n in names}))
    return summary


def nine_table(own,other):
    rr=own+other
    # Use the already-tested map-hash/initial-pair checks and matched means.
    groups={c:{r['map_id']:r for r in rr if int(float(r['cost']))==c} for c in COSTS}
    need(len(rr)==1620 and all(len(g)==180 for g in groups.values()),'Duplicate/missing rows in nine-condition input')
    maps=set(groups[1]);need(all(set(g)==maps for g in groups.values()),'Different map IDs')
    common=set(maps)
    for g in groups.values():
        for mid,r in g.items():
            b=groups[1][mid]
            need(r['map_sha256']==b['map_sha256'] and r['initial_pair_id']==b['initial_pair_id'],'Map/initial condition mismatch')
        common &={mid for mid,r in g.items() if E.flag(r['is_success'])}
    if common:table,members=E.paired_table(rr)
    else:
        members=[];table=[dict(kappa=c,N_validation=180,success_n=sum(E.flag(r['is_success']) for r in g.values()),common_n=0,
                 Raw_T_sec=None,Raw_Lw_sec=None,Refined_T_sec=None,Refined_Lw_sec=None) for c,g in groups.items()]
    for r in table:
        r['training_M']=5;r['origin']='fresh_rerun' if r['kappa']==1 else 'existing_saved_result'
        for k in ['T','Lw']:
            a,b=r[f'Raw_{k}_sec'],r[f'Refined_{k}_sec']
            r[f'{k}_reduction_pct']=None if a is None or a==0 else 100*(a-b)/a
    return table,members


def report(out,job=None,ident=None):
    other,hashes,oldtable,oldmaps=refs();rp=out/'report';rp.mkdir(exist_ok=True)
    E.csvwrite(rp/'existing_eight_RAW_REFINED_n167.csv',oldtable)
    E.csvwrite(rp/'existing_eight_common_success_maps.csv',oldmaps)
    # Historical rows remain archival; new kappa1 never overwrites 165/180.
    shutil.copy2(HERE/'reference/summary_costs_matched_extracted.csv',rp/'table_5_14a_historical_RAW_n163.csv')
    own=[]
    if job is not None:
        for rec in job['records']:
            d=mapdir(out,rec)
            if committed(d,'complete.json',ident):own.append(E.read(d/'summary.json'))
    E.csvwrite(rp/'cost1_fresh_per_map.csv',own,fields=None if own else ['cost','map_id'])
    text='# Table 5.14(b): κ=1 재학습 + 기존 여덟 조건 재사용\n\n'
    text+='κ=1은 원래와 같은 COST1·B6·β=0.5·FULL·CNN3/no-clock·seed42·5M의 **새 재학습 실행**이다. 과거 모델 복구나 과거 성공 165개 재현으로 표기하지 않는다. 나머지 κ=3·7·9·10·11·12·13·14는 기존 map별 Raw/Refined 수치를 재사용하였다.\n\n'
    status='COMPLETE' if len(own)==180 else 'PARTIAL'
    train_status_path=out/'runs'/RUN_ID/'status.json'
    train_done=train_status_path.is_file() and E.read(train_status_path).get('state')=='completed'
    audit=dict(version=VERSION,status=status,kappa1_maps=len(own),target_validation_maps=180,
               planned_fresh_kappas=[1],fresh_retrained_kappas=([1] if train_done else []),
               other_retrained_kappas=[],other_inferred_kappas=[],other_refined_kappas=[],
               test_rollouts=0,osm_rollouts=0,experiment_origin=EXPERIMENT_ORIGIN,
               original_kappa1_success_not_required=165,existing_eight_rows=len(other),existing_eight_common_n=167,
               reference_sha256=hashes,validation_only=True)
    if len(own)==180:
        table,members=nine_table(own,other)
        E.csvwrite(rp/'table_5_14b_nine_costs_RAW_REFINED.csv',table)
        E.csvwrite(rp/'nine_common_success_maps.csv',members,fields=['map_id','map_sha256','initial_pair_id'])
        E.csvwrite(rp/'per_map_nine_costs.csv',own+other)
        audit.update(common_n=len(members),fresh_kappa1_success=sum(E.flag(r['is_success']) for r in own),
                     success_preserved=True,checkpoint_step=job['checkpoint_step'],checkpoint_sha256=job['checkpoint_sha256'])
        text+='| κ | 성공/180 | 공통 n | Raw T (초) | Raw Lw (초) | Refined T (초) | Refined Lw (초) | 출처 |\n|---|---|---|---|---|---|---|---|\n'
        f=lambda v:'—' if v is None else f'{v:,.2f}'
        for r in table:text+=f"| {r['kappa']} | {r['success_n']}/180 | {r['common_n']} | {f(r['Raw_T_sec'])} | {f(r['Raw_Lw_sec'])} | {f(r['Refined_T_sec'])} | {f(r['Refined_Lw_sec'])} | {r['origin']} |\n"
        text+='\n성공률은 전체 Validation 180개, 시간 평균은 아홉 조건 모두 성공한 **새 교집합**을 사용한다. 나머지 여덟 경로를 재실행한 것이 아니라 저장된 1,440행을 새 교집합으로 다시 평균한 것이다. 기존 163/167을 고정 분모로 대입하지 않는다. 공통 성공이 0이면 시간은 미정의로 남긴다.\n'
    else:text+=f'현재 κ=1 Raw/Refined 완료 {len(own)}/180. 기존 여덟 조건 표는 보존되며 아홉 조건 표는 아직 확정하지 않는다.\n'
    E.atomic(rp/'Table_5_14b_KO.md',text);E.write(rp/'audit.json',audit)
    make_review(out,own)
    print(f'[REPORT] {status}; COST1={len(own)}/180. Other eight: reused CSV only.',flush=True)
    print('[REVIEW ZIP]',out/'TABLE514_KAPPA1_FRESH5M_REVIEW.zip',flush=True)
    return audit


def make_review(out,own):
    dest=out/'TABLE514_KAPPA1_FRESH5M_REVIEW.zip';tmp=out/'review.writing.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        files=[out/'protocol.json',out/'manifest.csv']
        files+=list((out/'report').glob('*'))+list((out/'audit').glob('*'))
        run=out/'runs'/RUN_ID
        for name in ['run_spec.json','training_settings.json','status.json','checkpoints.json','training_updates.csv','software.json','network_observation_audit.json']:
            files.append(run/name)
        for p in files:
            if p.is_file():z.write(p,p.relative_to(out).as_posix())
        for p in sorted((HERE/'reference').glob('*')):z.write(p,'reference/'+p.name)
        for name in ['run_kappa1_fresh5m.py','PACKAGE_SHA256.json']:
            z.write(HERE/name,'program/'+name)
        chosen=[]
        for ok in (True,False):
            subset=[r for r in own if E.flag(r['is_success'])==ok]
            subset.sort(key=lambda r:(not bool(r.get('path_changed')),r['map_id']))
            chosen+=subset[:2]
        for r in chosen:
            d=mapdir(out,r)
            for p in d.iterdir():
                if p.is_file():z.write(p,'samples/'+d.name+'/'+p.name)
        log=out/'logs/kappa1_train.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read()
            z.writestr('logs/kappa1_train_tail.txt',tail)
    with zipfile.ZipFile(tmp) as z:need(z.testzip() is None,'Review ZIP checksum error')
    os.replace(tmp,dest)


def evaluate(out,workers):
    job,ident=eval_job(out)
    own=[];pending=[]
    for r in job['records']:
        d=mapdir(out,r)
        if committed(d,'complete.json',ident):own.append(E.read(d/'summary.json'))
        else:pending.append(r)
    print(f'[VALIDATION+REFINE] fresh COST1 5M; cached={len(own)}, pending={len(pending)}; CPU={workers}',flush=True)
    try:
        if workers==1:
            worker_init(job)
            for r in pending:
                own.append(process((r,str(out),ident)))
                print(f'[COST1 {len(own)}/180] success={own[-1]["is_success"]}',flush=True)
        elif pending:
            import multiprocessing
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(job,)) as pool:
                futures=[pool.submit(process,(r,str(out),ident)) for r in pending]
                for f in as_completed(futures):
                    own.append(f.result());print(f'[COST1 {len(own)}/180] success={own[-1]["is_success"]}',flush=True)
    finally:report(out,job,ident)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--out',type=Path)
    ap.add_argument('--dataset-root',type=Path,help='Only when maps moved: directory containing train/ and test/. No recursive search.')
    ap.add_argument('--gpu',type=int,default=0,help='Physical GPU index for training only; default 0 is not a claim that it is free.')
    ap.add_argument('--device',choices=['cpu'],help='CPU training instead of the requested GPU; optional and much slower.')
    ap.add_argument('--workers',type=int,default=3,help='CPU Validation/refinement workers')
    ap.add_argument('--phase',choices=['check','all','train','refine','report'],default='all')
    a=ap.parse_args();need(a.workers>0 and a.gpu>=0,'Invalid worker/GPU count');root=a.root.expanduser().resolve()
    out=a.out.expanduser().resolve() if a.out else root/OUT_NAME
    need(root.is_dir() and out!=root and out!=HERE and out not in root.parents and HERE not in out.parents,'Unsafe output path')
    package=verify_package()
    if out.exists() and not (out/'protocol.json').exists():
        allowed={'.launcher.lock','report','audit','logs','source_snapshot','manifest.csv','TABLE514_KAPPA1_FRESH5M_REVIEW.zip'}
        need(not {p.name for p in out.iterdir()}-allowed,'Existing unrelated output. Use the new default folder.')
    out.mkdir(parents=True,exist_ok=True)
    with E.lock(out/'.launcher.lock'):
        if a.phase=='report':
            status=out/'runs'/RUN_ID/'status.json'
            if status.exists() and E.read(status).get('state')=='completed':job,ident=eval_job(out);report(out,job,ident)
            else:report(out)
            return
        if a.phase=='refine':
            need((out/'protocol.json').is_file(),'Train new COST1 first; no old checkpoint search.')
            evaluate(out,a.workers);return
        device=a.device or f'cuda:{a.gpu}'
        protocol=prepare(out,root,device,a.dataset_root.expanduser().resolve() if a.dataset_root else None)
        E.write(out/'audit/package_identity.json',package)
        print('[PLAN] κ=1 only, B6 beta=.5 FULL CNN3/no-clock, seed42, fresh 5M.',flush=True)
        print('[REUSE] κ=3,7,9,10,11,12,13,14: 1,440 existing Raw/Refined rows; NO retraining.',flush=True)
        print('[DATA] train900 + validation180; NO Test/OSM; NO old COST1 model search.',flush=True)
        if a.phase=='check':train(out,protocol,True);report(out);return
        (out/'logs').mkdir(exist_ok=True)
        with (out/'logs/kappa1_train.log').open('a',encoding='utf-8') as f:
            with redirect_stdout(Tee(sys.stdout,f)),redirect_stderr(Tee(sys.stderr,f)):train(out,protocol)
        if a.phase=='all':evaluate(out,a.workers)
        else:report(out)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('[STOP] New-run training recovery and completed map caches retained. Rerun the same command.',file=sys.stderr);sys.exit(130)
    except Exception as exc:
        print('[ERROR]',type(exc).__name__,str(exc),file=sys.stderr);traceback.print_exc();sys.exit(1)

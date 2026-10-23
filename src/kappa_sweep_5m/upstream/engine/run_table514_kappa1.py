#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete Table 5.14(b): native COST1 validation and paired 9-cost refinement.

No training. No test/OSM evaluation. Original experiment files are read-only.
Reuses authenticated saved COST1 paths; if absent, infers ONLY missing paths from
its original final 5M checkpoint on CPU. The uploaded common A* is unchanged.
The other eight cost paths are not recomputed: their existing diagnostic rows
are joined by map hash AND initial-condition ID; ALL nine means are recomputed.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib, io, json, math, os, sys, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import ch5_common as C

VERSION = 'table514_kappa1_native_completion_v1_20261003'
T, LW = C.T, C.LW
COSTS = (1, 3, 7, 9, 10, 11, 12, 13, 14)
RUN_ID = 'ppo_relaxedB6_NRG1_C3NT_b0.5_g1_l0.95_budgetobs_seed42'
NATIVE_VERSION = 'nrg1_cnn3_noclock_externalcap_v1_20260925'
NATIVE_HASH = '6a85e28a83dc989f0546bf8ec391c1a85ffbc5182b1daf9c9865e0da55729239'
FILES = ('coverage_core.py','core_ch4.py','core_ch4_bounded_b6.py','full_v2_utils.py',
         'nrg1_c3nt_core.py','nrg1_c3nt_features.py')
DIAGS = ('chapter4_C3NT_COST_3_7_9_11_connection_diagnostic_validation',
         'chapter4_C3NT_COST_10_12_13_14_connection_diagnostic_validation')
RT = MODEL = JOB = None

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [safe(v) for v in x]
    if isinstance(x,Path):return str(x)
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def digest(x):return hashlib.sha256(json.dumps(safe(x),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def flag(x):
    s=str(x).lower().strip();need(s in ('true','false','1','0','1.0','0.0'),'Invalid success flag: '+s)
    return s in ('true','1','1.0')
def num(x):
    if x is None or x=='':return None
    try:v=float(x)
    except (ValueError,TypeError):return None
    return v if math.isfinite(v) else None

def near(a,b,msg):
    a,b=num(a),num(b)
    need((a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=2e-9,abs_tol=2e-6)),f'{msg}: {a} != {b}')
def atomic(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing.'+str(os.getpid()))
    try:
        with tmp.open('w',encoding='utf-8',newline='') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()
def write(p,x):atomic(p,json.dumps(safe(x),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f);need(r.fieldnames and len(set(r.fieldnames))==len(r.fieldnames),'Bad CSV header: '+str(p));rr=list(r)
    need(all(None not in r for r in rr),'Malformed CSV: '+str(p));return rr

def csvwrite(p,rr,fields=None):
    rr=[safe(r) for r in rr];fields=fields or list(dict.fromkeys(k for r in rr for k in r))
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields);w.writeheader();w.writerows(rr);atomic(p,'\ufeff'+s.getvalue())

def member(parent,name):
    parent=Path(parent).resolve();p=(parent/name).resolve()
    need(not Path(name).is_absolute() and '..' not in Path(name).parts and parent in p.parents,'Unsafe member path')
    return p

@contextmanager
def lock(p):
    import fcntl
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another run owns this output')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def source_context(root,explicit=None):
    root=Path(root).resolve()
    candidates=[Path(explicit).expanduser().resolve()] if explicit else [root/'chapter4_relaxedB6_NRG1_C3NT_seed42']
    hits=[p for p in candidates if (p/'protocol.json').is_file()]
    need(len(hits)==1,'Original COST1 C3NT folder not found. Use --cost1-root /actual/chapter4_relaxedB6_NRG1_C3NT_seed42. No COST5/11 fallback.')
    base=hits[0];pr=read(base/'protocol.json');req=pr['request'];sp=req['spec'];st=req['settings'];env=sp['environment'];rd=sp['reward_definition']
    need(req['version']==NATIVE_VERSION and sp['run_id']==RUN_ID and sp['reward_case']=='NRG1_C3NT','Not original COST1 C3NT')
    need(sp['seed']==42 and sp['algorithm']=='ppo' and sp['beta']==.5,'Not requested seed/beta/PPO')
    need(sp['encoder_ablation']['merged_feature_dim']==261 and sp['encoder_ablation']['name']=='CNN3','Wrong network')
    need(sp.get('remaining_step_observation') is False and not sp.get('diagnostic_smoke_only',False),'Clock/smoke run not allowed')
    need(env['cell_size_m']==16 and env['v_straight']==11.462 and env['v_turn']==5.67 and env['initialization']=='priority_mass','Time model/initialization differs from existing cost sweep')
    need(env['mask_mode']=='relaxed' and env['detour_budget_max']==6 and env['step_budget_factor']==5,'Not B6/H5')
    need(rd['version']==NATIVE_VERSION and rd['success_terminal']==2 and rd['failure_terminal']==0,'Not FULL reward')
    need(rd['cost']=='-((1-beta)+beta*W_pre/W_total)*dt/(ceil(5*N)*time_turn_sec)','Not coefficient-one reward')
    need(st['total_timesteps']==5_000_000 and st['num_envs']==8 and st['ppo']['gamma']==1 and st['ppo']['gae_lambda']==.95,'Wrong training budget/settings')
    run=base/'runs'/RUN_ID;status=read(run/'status.json');need(status['state']=='completed','COST1 training is not completed; this script never trains')
    step=int(status['actual_timesteps']);need(5_000_000<=step<5_000_000+st['num_envs']*st['ppo']['n_steps'],'Not final 5M checkpoint')
    actual=read(run/'run_spec.json');need(actual.get('nrg1_protocol_sha256')==sha(base/'protocol.json'),'Run protocol differs')
    expected=dict(actual);expected.pop('nrg1_protocol_sha256',None);need(expected==sp,'Requested and actual specs differ')
    regs=read(run/'checkpoints.json');finals=[r for r in regs if r.get('final') and int(r['num_timesteps'])==step]
    need(len(finals)==1,'Missing/ambiguous final model registration');cp=finals[0]
    protected={str(base/'protocol.json'):sha(base/'protocol.json'),str(run/'run_spec.json'):sha(run/'run_spec.json'),str(run/'checkpoints.json'):sha(run/'checkpoints.json')}
    hashes={}
    for n in FILES:
        p=base/'source_snapshot'/n;need(p.is_file() and n in pr['snapshot_hash'] and sha(p)==pr['snapshot_hash'][n],'Native source missing/changed: '+str(p));hashes[n]=sha(p);protected[str(p)]=sha(p)
    need(hashes['nrg1_c3nt_core.py']==NATIVE_HASH,'Original COST1 reward adapter changed')
    mf=base/'manifest.csv';need(sha(mf)==pr['manifest_sha256'],'Manifest changed');protected[str(mf)]=sha(mf)
    allrows=rows(mf);va=[r for r in allrows if r['source']=='synthetic' and r['split'].lower() in ('validation','val','valid')]
    need(len(va)==180 and len({r['map_id'] for r in va})==180,'Require 180 unique synthetic validation maps')
    for r in va:r['split']='validation'
    va.sort(key=lambda r:(r['source'],str(r['tier']),r['pattern'],r['map_id']))
    return dict(root=str(root),base=str(base),run=str(run),spec=actual,settings=st,checkpoint_sha256=cp['sha256'],checkpoint_step=step,
       checkpoint_path=str(run/'last_model.zip'),snapshot=str(base/'source_snapshot'),hashes=hashes,protected=protected,records=va,
       evaluation=req['evaluation'],native_version=NATIVE_VERSION)

def load_native(job):
    sys.dont_write_bytecode=True;snap=Path(job['snapshot']).resolve();mm={}
    names=[Path(n).stem for n in FILES if n!='nrg1_c3nt_features.py']
    sys.path.insert(0,str(snap));importlib.invalidate_caches()
    try:
        for n in names:
            p=snap/(n+'.py');need(sha(p)==job['hashes'][n+'.py'],'Native source changed: '+n)
            old=sys.modules.get(n)
            if old is not None and Path(old.__file__).resolve().parent!=snap:del sys.modules[n]
            m=importlib.import_module(n);need(Path(m.__file__).resolve().parent==snap,'Wrong native module: '+n);mm[n]=m
    finally:sys.path.remove(str(snap))
    core=mm['nrg1_c3nt_core'];need(core.VERSION==NATIVE_VERSION and core.NORMALIZATION_STEP_FACTOR==5,'Wrong native adapter')
    bounded=mm['core_ch4_bounded_b6'];cfg=bounded.ExperimentEnvConfig(**job['spec']['environment'])
    return SimpleNamespace(cfg=cfg,factory=core.wrap_factory(bounded.make_core),core=core,bounded=bounded,initial=mm['core_ch4'],utils=mm['full_v2_utils'],modules=mm,moves=C.V.MOVE_TUPLES,snapshot=snap)

def worker_init(job):
    global RT,MODEL,JOB
    JOB=job;RT=load_native(job);MODEL=None

def initial(rt,rec,array,seed):
    native=rt.initial.ExperimentCore(replace(rt.cfg,initialization='priority_mass'))
    seed_hash=int(hashlib.sha256(json.dumps(('initial',rec['sha256'],'priority_mass',0,seed),sort_keys=True,default=str).encode()).hexdigest()[:8],16)
    native.reset(array,seed=seed_hash)
    pair=hashlib.sha256(json.dumps([rec['sha256'],'priority_mass',0,seed,native.start_pos,native.start_heading],sort_keys=True).encode()).hexdigest()
    return dict(start=list(native.start_pos),heading=int(native.start_heading),pair_id=pair)

def normalized(row,rec,ic,job):
    r=deepcopy(row);r.update(method='relaxed_kappa1',label='Relaxed COST1 beta0.5 5M',family='ppo',beta=.5,effective_B=6,
        nominal_m=5,checkpoint_step=job['checkpoint_step'],checkpoint_sha256=job['checkpoint_sha256'],map_id=rec['map_id'],map_sha256=rec['sha256'],source='synthetic',split='validation',
        pattern=rec['pattern'],tier=rec['tier'],initial_pair_id=ic['pair_id'],start_r=ic['start'][0],start_c=ic['start'][1],start_heading=ic['heading'],evaluation_seed=job['evaluation']['seed'],initial_seed=job['evaluation']['initial_seed'],native_reward_beta=.5)
    return safe(r)

def old_cache(job,rec):
    """Return only a fully committed, checksum-verified original trajectory."""
    folder=Path(job['base'])/'validation_final';md=folder/'maps'/hashlib.sha256(rec['map_id'].encode()).hexdigest()[:24]
    commit=md/'map_complete.json'
    if not commit.exists():return None
    need((folder/'evaluation_identity.json').is_file(),'Stored map without evaluation identity')
    ident=read(folder/'evaluation_identity.json');par=ident['parameters'];cm=read(commit)
    need(digest(par)==ident['sha256']==cm['identity_sha256'],'Old cache identity mismatch')
    need(par['checkpoint_sha256']==job['checkpoint_sha256'] and par['spec']==job['spec'] and par['phase']=='validation','Wrong old checkpoint/split/spec')
    need(par['evaluation']['deterministic'] is True and par['evaluation']['seed']==job['evaluation']['seed'] and par['evaluation']['initial_seed']==job['evaluation']['initial_seed'],'Old evaluation settings differ')
    reg={r[0]:(r[1],r[2]) for r in par['maps']};need(reg.get(rec['map_id'])==(rec['sha256'],'validation'),'Old cache contains a different map')
    for name in ('trajectory.npz','metrics.json','initial_condition.json','first_visits.csv'):
        p=member(md,name);need(name in cm['files'] and p.is_file() and sha(p)==cm['files'][name],'Corrupt committed artifact: '+str(p))
    with np.load(md/'trajectory.npz',allow_pickle=False) as z:tr={k:z[k].copy() for k in z.files}
    m=read(md/'metrics.json');ic=read(md/'initial_condition.json')
    need(m['map_id']==rec['map_id'] and m['map_sha256']==rec['sha256'] and m['initial_pair_id']==ic['pair_id'] and m['run_id']==RUN_ID,'Old cache belongs to another run/map')
    need(m['nrg_version']==NATIVE_VERSION,'Old reward adapter differs')
    return m,tr,ic,dict(origin='original_validation_cache',map_complete_sha256=sha(commit),trajectory_sha256=sha(md/'trajectory.npz'))

def load_model(job,rt):
    global MODEL
    if MODEL is not None:return MODEL
    p=Path(job['checkpoint_path']);need(p.is_file() and sha(p)==job['checkpoint_sha256'],'Missing raw trace and original final COST1 checkpoint unavailable')
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1);sys.path.insert(0,str(rt.snapshot))
    try:
        f=importlib.import_module('nrg1_c3nt_features');need(Path(f.__file__).resolve().parent==rt.snapshot and sha(f.__file__)==job['hashes']['nrg1_c3nt_features.py'],'Wrong original CNN')
        MODEL=MaskablePPO.load(p,device='cpu');need(int(MODEL.num_timesteps)==job['checkpoint_step'],'Wrong timestep')
        rt.core.audit_model(MODEL);f.audit_cnn3_model(MODEL);MODEL.policy.set_training_mode(False)
    finally:sys.path.remove(str(rt.snapshot))
    return MODEL

def run_episode(rt,job,rec,array,ic,decision=None):
    c=rt.factory(rt.cfg);ob,_=c.reset(array,start=tuple(ic['start']),heading=ic['heading'],seed=job['evaluation']['seed'],map_name=rec['map_id'])
    actions=[];positions=[list(c.drone_pos)];times=[0.];masks=[];budgets=[]
    if c.complete_at_reset:c.step(0)
    while not c._done:
        mask=np.asarray(c.action_masks(),bool).copy()
        if decision is None:
            model=load_model(job,rt)
            import torch
            with torch.no_grad():
                obs,_=model.policy.obs_to_tensor(ob);d=model.policy.get_distribution(obs,action_masks=mask.reshape(1,-1));p=d.distribution.probs[0].detach().cpu().numpy()
            need(np.isfinite(p).all() and np.all(p[~mask]<1e-6) and abs(float(p.sum())-1)<1e-5,'Invalid policy probabilities');a=int(np.argmax(p))
        else:a=int(decision(c,ob,mask))
        need(0<=a<4 and mask[a],'Action outside original mask');actions.append(a);masks.append(mask);budgets.append(int(c.detour_budget))
        ob,_,term,trunc,_=c.step(a);need(not(term and trunc),'Native stopping contradiction');positions.append(list(c.drone_pos));times.append(c.elapsed_time_sec)
    c.assert_consistent();row=normalized(c.info(final=True),rec,ic,job)
    trace=dict(priority_map=np.asarray(array),actions=np.asarray(actions,np.int8),positions=np.asarray(positions,np.int32),time_sec=np.asarray(times,float),
       pre_action_masks=np.asarray(masks,bool).reshape(-1,4),budget_before=np.asarray(budgets,np.int16),first_visit_step=c.first_visit_step.copy(),first_visit_time_sec=c.first_visit_time.copy())
    return row,trace

def frozen(p,x):
    if p.exists():need(read(p)==safe(x),'Changed input/code; use a new --out: '+str(p))
    else:write(p,x)

def read_new_cache(out,rec,ident):
    d=Path(out)/'cost1/maps'/hashlib.sha256(rec['map_id'].encode()).hexdigest()[:24];p=d/'complete.json'
    if not p.exists():return None
    co=read(p);need(co['identity']==ident and co['map_id']==rec['map_id'],'New output identity mismatch')
    for name,h in co['files'].items():need(sha(member(d,name))==h,'Corrupt completed output '+str(d))
    return read(d/'summary.json')

def process(task):
    rec,out,ident,cached_only=task;job=JOB;rt=RT
    previous=read_new_cache(out,rec,ident)
    if previous is not None:return previous
    prior=old_cache(job,rec)
    if prior is not None:
        m,tr,ic,provenance=prior;array=tr['priority_map'];row=normalized(m,rec,ic,job)
    else:
        need(not cached_only,'Missing saved COST1 trajectory for '+rec['map_id']+'; omit --cached-only to infer from the original checkpoint')
        p=Path(rec['path']);need(p.is_file(),'Original validation array not found: '+str(p));array=np.load(p,allow_pickle=False)
        ic=initial(rt,rec,array,job['evaluation']['initial_seed']);row,tr=run_episode(rt,job,rec,array,ic);provenance=dict(origin='original_checkpoint_cpu_inference_missing_path_only')
    need(rt.utils.array_hash(array)==rec['sha256'],'Map array differs');ic2=initial(rt,rec,array,job['evaluation']['initial_seed']);need(ic['pair_id']==ic2['pair_id'] and list(ic['start'])==ic2['start'] and int(ic['heading'])==ic2['heading'],'Original initial condition differs')
    # This entry point does NOT call the COST11-only load_native in ch5_common.
    # It receives the authenticated COST1 native factory, preserving original reward/physics.
    raw=row
    refined,rtrace,audit,segs=C.common_refine(rt,row,tr)
    result=dict(cost=1,map_id=rec['map_id'],map_sha256=rec['sha256'],initial_pair_id=ic['pair_id'],source='synthetic',split='validation',tier=rec['tier'],pattern=rec['pattern'],
       is_success=flag(raw['is_success']),steps=int(raw['steps']),active_cells=int(raw['active_cells']),coverage_pct=float(raw['coverage_pct']),
       mission_time_sec=raw[T],original_Lw_sec=raw[LW],witness_time_sec=refined[T],witness_Lw_sec=refined[LW],
       witness_steps=int(refined['steps']),actual_source_replay_verified=True,native_witness_replay_verified=True,
       witness_respects_original_execution_cap=int(refined['steps'])<=int(raw['steps']),
       checkpoint_step=job['checkpoint_step'],checkpoint_sha256=job['checkpoint_sha256'],**provenance)
    d=Path(out)/'cost1/maps'/hashlib.sha256(rec['map_id'].encode()).hexdigest()[:24];d.mkdir(parents=True,exist_ok=True)
    write(d/'summary.json',result);write(d/'audit.json',audit);write(d/'raw_metrics.json',raw);write(d/'refined_metrics.json',refined)
    for name,trace in [('raw.npz',tr),('refined.npz',rtrace)]:
        tmp=d/(name+'.writing.npz');np.savez_compressed(tmp,**trace);os.replace(tmp,d/name)
    csvwrite(d/'segments.csv',segs,fields=None if segs else ['segment'])
    write(d/'complete.json',dict(identity=ident,map_id=rec['map_id'],files={n:sha(d/n) for n in ['summary.json','audit.json','raw_metrics.json','refined_metrics.json','raw.npz','refined.npz','segments.csv']}))
    return result

def load_other_eight(root,overrides=None):
    paths=[Path(p).resolve() for p in overrides] if overrides else [Path(root)/d/'per_map_all_costs.csv' for d in DIAGS]
    rr=[];protected={}
    for p in paths:
        need(p.is_file(),'Existing eight-cost diagnostic CSV missing: '+str(p));protected[str(p)]=sha(p);part=rows(p)
        for r in part:
            cost=int(float(r['cost']));need(cost in COSTS[1:],'Unexpected cost in diagnostic')
            need(flag(r['actual_source_replay_verified']) and flag(r['native_witness_replay_verified']) and flag(r['witness_respects_original_execution_cap']),'Diagnostic lacks verified replay/cap witness')
            rr.append(r)
    return rr,protected

def paired_table(rr,required=COSTS,expected_n=180):
    """Never append COST1 to the old n=167 means. Rebuild intersection and all means."""
    grouped={c:{} for c in required}
    for r in rr:
        c=int(float(r['cost']));need(c in grouped,'Unknown cost');mid=r['map_id'];need(mid not in grouped[c],'Duplicate cost/map')
        grouped[c][mid]=r
    need(all(len(g)==expected_n for g in grouped.values()),'Every cost must have all validation rows, including failures')
    maps=set(grouped[required[0]])
    need(all(set(g)==maps for g in grouped.values()),'Validation map sets differ')
    common=set(maps)
    for g in grouped.values():
        common &= {m for m,r in g.items() if flag(r['is_success'])}
    for mid in maps:
        first=grouped[required[0]][mid]
        for g in grouped.values():
            need(g[mid]['map_sha256']==first['map_sha256'] and g[mid]['initial_pair_id']==first['initial_pair_id'],'Map hash/initial pair differs: '+mid)
    need(common,'No common successful map; report success only, no artificial means')
    ordered=sorted(common);out=[]
    for c,g in grouped.items():
        row=dict(kappa=c,N_validation=expected_n,success_n=sum(flag(r['is_success']) for r in g.values()),common_n=len(ordered))
        for source,target in [(T,'Raw_T_sec'),('original_Lw_sec','Raw_Lw_sec'),('witness_time_sec','Refined_T_sec'),('witness_Lw_sec','Refined_Lw_sec')]:
            v=[num(g[mid][source]) for mid in ordered];need(all(x is not None for x in v),'Missing complete-mission metric');row[target]=float(np.mean(v))
        need(row['Refined_T_sec']<=row['Raw_T_sec']+2e-6 and row['Refined_Lw_sec']<=row['Raw_Lw_sec']+2e-6,'Refinement worsened success-mean metric')
        out.append(row)
    return out,[dict(map_id=m,map_sha256=grouped[required[0]][m]['map_sha256'],initial_pair_id=grouped[required[0]][m]['initial_pair_id']) for m in ordered]

def report(root,out,job,ident,other,protected):
    own=[read_new_cache(out,r,ident) for r in job['records']];own=[r for r in own if r is not None]
    csvwrite(out/'cost1/per_map.csv',own,fields=None if own else ['cost','map_id'])
    status=dict(version=VERSION,phase='validation_only',training_updates=0,test_rollouts=0,cost1_n=len(own),target_n=180,status='COMPLETE' if len(own)==180 else 'PARTIAL',source_20m='reference/Table_5_16_Strict20M_Raw_verified.csv contains existing Raw, not a new refinement run')
    text='# Table 5.14(b) — κ=1 포함 동일 validation 비교\n\n'
    if len(own)==180:
        table,members=paired_table(own+other);csvwrite(out/'report/table_5_14b_kappa1_included.csv',table);csvwrite(out/'report/common_success_maps.csv',members);csvwrite(out/'report/per_map_nine_costs.csv',own+other)
        status.update(common_n=len(members),kappas=list(COSTS),source_hashes=protected)
        text+='| κ | 성공/180 | 공통 n | Raw T (초) | Raw Lw (초) | Refined T (초) | Refined Lw (초) |\n|---|---|---|---|---|---|---|\n'
        for r in table:text+=f"| {r['kappa']} | {r['success_n']}/180 | {r['common_n']} | {r['Raw_T_sec']:.2f} | {r['Raw_Lw_sec']:.2f} | {r['Refined_T_sec']:.2f} | {r['Refined_Lw_sec']:.2f} |\n"
        text+='\n모든 시간 평균은 κ=1을 포함한 아홉 조건이 모두 성공한 같은 지도에서 다시 계산했다. 이전 167개 평균은 재사용하지 않는다. 실패한 지도의 완주 Lw를 채우지 않는다.\n'
    else:text+=f'κ=1 처리 {len(own)}/180. 아홉 조건의 공통 성공집합과 전체 표는 아직 확정하지 않았습니다.\n'
    atomic(out/'report/Table_5_14b_KO.md',text);write(out/'report/audit.json',status)
    for p,h in protected.items():need(sha(p)==h,'Read-only source changed during work: '+p)
    archive=out/'TABLE514_KAPPA1_COMPLETION_REVIEW.zip';tmp=out/'review.writing.zip'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.csv','.json','.md') and '/maps/' not in str(p) and p.name not in ('.lock',):z.write(p,p.relative_to(out))
        for p in sorted((Path(__file__).parent/'reference').glob('*')):z.write(p,'reference/'+p.name)
        for r in sorted(own,key=lambda x:x['map_id'])[:2]:
            d=out/'cost1/maps'/hashlib.sha256(r['map_id'].encode()).hexdigest()[:24]
            for p in sorted(d.iterdir()):
                if p.is_file():z.write(p,Path('samples')/d.name/p.name)
    os.replace(tmp,archive)
    with zipfile.ZipFile(archive) as z:need(z.testzip() is None,'Review ZIP integrity failed')
    print('[REPORT]',status['status'],'cost1',len(own),'/180',flush=True);print('[ZIP]',archive,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--cost1-root',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--phase',choices=['check','all','report'],default='all');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cached-only',action='store_true',help='Never load model; fail if a saved COST1 trajectory is missing')
    p.add_argument('--diagnostic-csv',type=Path,nargs=2,help='Two existing eight-cost per_map_all_costs.csv files, if moved')
    a=p.parse_args();need(a.workers>=1,'workers must be positive');root=a.root.expanduser().resolve();out=a.out.expanduser().resolve() if a.out else root/'chapter5_TABLE514_KAPPA1_VALIDATION_v1'
    job=source_context(root,a.cost1_root);base=Path(job['base']);need(out!=base and base not in out.parents and out not in base.parents,'Output overlaps input experiment')
    other,ph=load_other_eight(root,a.diagnostic_csv);ph.update(job['protected'])
    # Verify the same 180 validation map IDs and initial-map hashes before lengthy work.
    oldmap={r['map_id']:r['map_sha256'] for r in other};need(oldmap=={r['map_id']:r['sha256'] for r in job['records']},'Eight-cost and COST1 validation sets differ')
    codehash={str(q.relative_to(Path(__file__).parent)):sha(q) for q in Path(__file__).parent.rglob('*.py') if 'tests' not in q.parts}
    identity=dict(version=VERSION,cost1=job['checkpoint_sha256'],step=job['checkpoint_step'],spec=job['spec'],source_hashes=ph,code_hashes=codehash,scope='validation180_only',solver='same_uploaded_common_SegmentGraph_nonexpanding_steps')
    ident=digest(identity)
    if a.phase=='check':
        rt=load_native(job);print('[CHECK] Original COST1 metadata/source PASS; other eight raw/refined files found.');print('[COST1]',base);print('[MAPS]',len(job['records']));return
    out.mkdir(parents=True,exist_ok=True)
    with lock(out/'.lock'):
        frozen(out/'protocol.json',dict(identity=ident,parameters=identity));write(out/'source_audit.json',job)
        if a.phase=='all':
            pending=[r for r in job['records'] if read_new_cache(out,r,ident) is None];done=180-len(pending);print(f'[PLAN] COST1 {done}/180 cached; {len(pending)} pending. No training/Test/OSM.',flush=True)
            if a.workers==1:
                worker_init(job)
                for r in pending:
                    result=process((r,str(out),ident,a.cached_only));done+=1;print(f'[COST1 {done}/180] success={result["is_success"]}',flush=True)
            elif pending:
                import multiprocessing
                with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(job,)) as pool:
                    futures=[pool.submit(process,(r,str(out),ident,a.cached_only)) for r in pending]
                    for f in as_completed(futures):result=f.result();done+=1;print(f'[COST1 {done}/180] success={result["is_success"]}',flush=True)
        report(root,out,job,ident,other,ph)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('[STOP] Completed map caches are preserved. Rerun the same command.',file=sys.stderr);sys.exit(130)
    except Exception as e:print('[ERROR]',type(e).__name__,str(e),file=sys.stderr);raise

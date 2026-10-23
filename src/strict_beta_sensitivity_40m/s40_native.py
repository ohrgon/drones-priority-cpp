"""Native policy trace reconstruction and common refinement; no optimizer calls."""
from __future__ import annotations
import importlib, importlib.metadata
import math, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import ch5_common as C
from native_reference import strict50m_original as N
import s40_io as I

sys.dont_write_bytecode=True
T,LW=C.T,C.LW


def normalized(metric,job,rec,ic):
    r=dict(metric)
    r.update(method=job['key'],model=job['key'],label=f"Strict beta={job['beta']:g} 40M",family='ppo',beta=job['beta'],
        native_reward_beta=job['beta'],effective_B=0,nominal_m=40,checkpoint_step=I.STEP,
        checkpoint_sha256=job['checkpoint']['sha256'],map_id=rec['map_id'],map_sha256=rec['sha256'],source='synthetic',split='validation',
        pattern=rec['pattern'],tier=rec['tier'],initial_pair_id=ic['pair_id'],start_r=ic['start'][0],start_c=ic['start'][1],start_heading=ic['heading'],
        initial_seed=20260907,evaluation_seed=42,deterministic=True,stage='raw_stage1_same_40M_checkpoint')
    return I.safe(r)


def verify_software(job):
    current={}
    for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib'):
        got=importlib.metadata.version(name);want=job['software'].get(name);current[name]=got
        I.need(want is not None and got==want,f'Original dqn_env required: {name}: expected {want}, actual {got}. No automatic upgrade.')
    return current


def load_model(job,rt):
    # Only trusted, locally produced checkpoints are used. No unpickling in check/report.
    verify_software(job)
    path=Path(job['checkpoint']['path']);I.need(I.sha(path)==job['checkpoint']['sha256'],'40M weight changed')
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(int(job['settings'].get('torch_threads',4)))
    snapshot=Path(job['snapshot']).resolve()
    name=Path(I.FEATURE).stem;f=snapshot/I.FEATURE
    I.need(I.sha(f)==job['feature_sha256'],'Original CNN feature module changed')
    old=sys.modules.get(name)
    if old is not None and Path(old.__file__).resolve().parent!=snapshot:sys.modules.pop(name)
    sys.path.insert(0,str(snapshot));importlib.invalidate_caches()
    try:
        features=importlib.import_module(name)
        I.need(Path(features.__file__).resolve()==f,'Wrong feature module imported')
        model=MaskablePPO.load(path,device='cpu')
        I.need(int(model.num_timesteps)==I.STEP,'Loaded checkpoint is not 40M')
        rt.core.audit_model(model);features.audit_cnn3_model(model);model.policy.set_training_mode(False)
    finally:sys.path.remove(str(snapshot))
    return model


def infer_trace(rt,job,rec,array,ic,model=None,decision=None):
    """Same native masked argmax as run_strict_50m.evaluate_one, plus actions.
    decision is only used by isolated tests and is not exposed by the CLI.
    """
    core=rt.factory(rt.cfg)
    ob,_=core.reset(array,start=tuple(ic['start']),heading=int(ic['heading']),seed=42,map_name=rec['map_id'])
    positions=[list(core.drone_pos)];times=[0.];actions=[];masks=[];budgets=[];ent=[];norm=[]
    if core.complete_at_reset:core.step(0)
    while not core._done:
        mask=np.asarray(core.action_masks(),bool).copy();I.need(mask.shape==(4,) and mask.any(),'Invalid native mask')
        if decision is None:
            I.need(model is not None,'No model for missing raw trace')
            import torch
            with torch.no_grad():
                obs,_=model.policy.obs_to_tensor(ob);dist=model.policy.get_distribution(obs,action_masks=mask.reshape(1,-1))
                p=dist.distribution.probs[0].detach().cpu().numpy().astype(float)
            I.need(np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and np.all(p[~mask]<1e-6),'Invalid masked policy distribution')
            action=int(np.argmax(p));h=-float(np.sum(p[p>0]*np.log(p[p>0])));ent.append(h)
            if mask.sum()>1:norm.append(h/math.log(int(mask.sum())))
        else:action=int(decision(core,ob,mask))
        I.need(0<=action<4 and mask[action],'Policy action violates Strict mask')
        actions.append(action);masks.append(mask);budgets.append(int(core.detour_budget))
        ob,_,term,trunc,_=core.step(action)
        I.need(not(term and trunc) and len(actions)<=core.max_steps,'Native stop inconsistency')
        positions.append(list(core.drone_pos));times.append(float(core.elapsed_time_sec))
    core.assert_consistent();r=normalized(core.info(final=True),job,rec,ic)
    r.update(masked_entropy_mean=N.mean(ent),multi_action_normalized_entropy_mean=N.mean(norm))
    N.validate_row(r,job)
    trace={'priority_map':np.asarray(array).copy(),'actions':np.asarray(actions,np.int8),
           'positions':np.asarray(positions,np.int16).reshape(-1,2),'time_sec':np.asarray(times,float),
           'pre_action_masks':np.asarray(masks,bool).reshape(-1,4),'budget_before':np.asarray(budgets,np.int16),
           'first_visit_step':core.first_visit_step.copy(),'first_visit_time_sec':core.first_visit_time.copy()}
    return I.safe(r),trace


def compare_historical(row,reference):
    if reference is None:return {'status':'no_per_map_metric_reference','compared':False}
    for k in ('map_id','map_sha256','initial_pair_id','checkpoint_sha256'):
        I.need(row[k]==reference[k],'Historical row identity changed: '+k)
    for k in (T,LW,'steps','max_steps','active_cells','coverage_pct','turn_count','revisit_steps','zero_transit_steps','path_length_m',
              'time_straight_sec','time_turn_sec','start_r','start_c','start_heading','c3nt_cost_multiplier','nrg_detour_ratio'):
        I.near(row.get(k),reference.get(k),'Reconstructed raw vs original '+k)
    I.need(I.flag(row['is_success'])==I.flag(reference['is_success']),'Historical success changed')
    return {'status':'same_checkpoint_raw_matches_historical_metrics','compared':True,
            'absolute_T_difference_sec':abs(float(row[T])-float(reference[T])),
            'absolute_Lw_difference_sec':abs(float(row[LW])-float(reference[LW])) if I.flag(row['is_success']) else None}


def verify_result(directory,identity,rec):
    co=I.receipt_read(directory,'complete.json',identity)
    if co is None:return None
    I.need(co.get('map_id')==rec['map_id'] and co.get('map_sha256')==rec['sha256'],'Completed map identity differs')
    required={'raw_metrics.json','refined_metrics.json','raw_trace.npz','refined_trace.npz','refinement_audit.json','segments.csv','raw_complete.json'}
    I.need(required<=set(co.get('files',{})),'Incomplete completed-map receipt')
    raw=I.read(Path(directory)/'raw_metrics.json');ref=I.read(Path(directory)/'refined_metrics.json');audit=I.read(Path(directory)/'refinement_audit.json')
    for r in (raw,ref):I.need(r.get('map_id')==rec['map_id'] and r.get('map_sha256')==rec['sha256'],'Changed result map')
    I.need(I.flag(raw['is_success'])==I.flag(ref['is_success']),'Changed success in result')
    return raw,ref,audit


def process_map(rt,job,rec,out,identity,model_getter,cached_only=False,decision=None):
    d=Path(out)/'methods'/job['key']/'maps'/I.map_key(rec)
    previous=verify_result(d,identity,rec)
    if previous is not None:return (*previous,'completed_refinement_reused')
    oldraw=I.receipt_read(d,'raw_complete.json',identity)
    if oldraw is not None:
        I.need({'raw_metrics.json','raw_trace.npz','raw_provenance.json'}<=set(oldraw['files']),'Incomplete raw receipt')
        raw=I.read(d/'raw_metrics.json');trace=I.load_npz(d/'raw_trace.npz');origin='saved_raw_trace_reused'
        I.need(raw['checkpoint_sha256']==job['checkpoint']['sha256'] and int(raw['checkpoint_step'])==I.STEP,'Wrong saved raw checkpoint')
    else:
        I.need(not cached_only,'Only metric CSV/JSON is available; action trace missing for '+job['key']+' '+rec['map_id']+'. Omit --cached-only to reconstruct from the same 40M checkpoint.')
        array=np.load(rec['path'],allow_pickle=False);I.need(rt.utils.array_hash(array)==rec['sha256'],'Validation map bytes changed')
        ic=N.initial_condition(rt,SimpleNamespace(**rec),array)
        model=None if decision is not None else model_getter()
        raw,trace=infer_trace(rt,job,rec,array,ic,model,decision)
        check=compare_historical(raw,job.get('historical_rows',{}).get(rec['map_id']))
        I.save_npz(d/'raw_trace.npz',trace);I.write(d/'raw_metrics.json',raw)
        I.write(d/'raw_provenance.json',{'origin':'same_registered_40M_checkpoint_cpu_argmax','original_metrics_check':check,
            'checkpoint_path':job['checkpoint']['path'],'checkpoint_sha256':job['checkpoint']['sha256'],
            'optimizer_updates':0,'refinement_applied_in_this_stage':False})
        I.write(d/'raw_complete.json',{'identity':identity,'map_id':rec['map_id'],'map_sha256':rec['sha256'],
            'files':{n:I.sha(d/n) for n in ('raw_metrics.json','raw_trace.npz','raw_provenance.json')}})
        origin='same_40M_checkpoint_raw_trace_reconstructed'
    # Full validation of source map and initialization also on resumed Raw caches.
    array=trace['priority_map'];I.need(rt.utils.array_hash(array)==rec['sha256'],'Stored raw map hash changed')
    ic=N.initial_condition(rt,SimpleNamespace(**rec),array)
    I.need(raw['initial_pair_id']==ic['pair_id'] and [int(raw['start_r']),int(raw['start_c'])]==list(ic['start']) and int(raw['start_heading'])==int(ic['heading']),'Native starting condition differs')
    reference_check=compare_historical(raw,job.get('historical_rows',{}).get(rec['map_id']))
    ref,arr,audit,segments=C.common_refine(rt,raw,trace)
    ref.update(model=job['key'],deterministic=True)
    audit.update(beta=job['beta'],model=job['key'],map_id=rec['map_id'],map_sha256=rec['sha256'],checkpoint_step=I.STEP,
        checkpoint_sha256=job['checkpoint']['sha256'],initial_pair_id=raw['initial_pair_id'],source='synthetic',split='validation',
        raw_provenance=origin,historical_raw_check=reference_check)
    I.save_npz(d/'refined_trace.npz',arr);I.write(d/'refined_metrics.json',ref);I.write(d/'refinement_audit.json',audit)
    I.csvwrite(d/'segments.csv',segments)
    files=['raw_metrics.json','refined_metrics.json','raw_trace.npz','refined_trace.npz','refinement_audit.json','segments.csv','raw_complete.json','raw_provenance.json']
    I.write(d/'complete.json',{'identity':identity,'map_id':rec['map_id'],'map_sha256':rec['sha256'],'files':{n:I.sha(d/n) for n in files}})
    return raw,ref,audit,origin


def worker(context,key,identity,cached_only=False,decision=None):
    I.need(context['code_sha256']==I.code_hashes(),'Packaged source changed after preflight')
    job=context['jobs'][key]; rt=C.load_native(job);model=None;rr=[];ff=[];aa=[];origin_counts={}
    base=Path(context['out'])/'methods'/key
    # Separate process per beta: no cross-beta native module or RNG leakage.
    def getter():
        nonlocal model
        if model is None:model=load_model(job,rt)
        return model
    start=time.time()
    try:
        for i,rec in enumerate(job['records'],1):
            raw,ref,audit,origin=process_map(rt,job,rec,context['out'],identity,getter,cached_only,decision)
            rr.append(raw);ff.append(ref);aa.append(audit);origin_counts[origin]=origin_counts.get(origin,0)+1
            if i==1 or i%10==0 or i==len(job['records']):
                I.csvwrite(base/'raw_per_map.csv',rr);I.csvwrite(base/'refined_per_map.csv',ff);I.csvwrite(base/'audit_per_map.csv',aa)
                print(f"[VAL-REFINE] {key} {i}/{len(job['records'])}; success={sum(I.flag(r['is_success']) for r in rr)}; changed={sum(a['path_changed'] for a in aa)}; elapsed={time.time()-start:.1f}s",flush=True)
        files=['raw_per_map.csv','refined_per_map.csv','audit_per_map.csv']
        I.write(base/'complete.json',{'identity':identity,'n_maps':len(rr),'key':key,'checkpoint_sha256':job['checkpoint']['sha256'],
            'checkpoint_step':I.STEP,'origin_counts':origin_counts,'files':{n:I.sha(base/n) for n in files}})
    finally:
        model=None
    return rr,ff,aa

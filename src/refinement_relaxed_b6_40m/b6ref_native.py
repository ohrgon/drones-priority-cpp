"""Original masked CPU argmax -> committed Raw trace -> common connection refinement.
No optimizer step, policy update, checkpoint writing or baseline replanning exists here.
"""
from __future__ import annotations
import importlib, importlib.metadata, math, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import b6ref_io as I
import ch5_common as C
import b124_native_support as U
from b124_core_guard import install_budget_guard
sys.dont_write_bytecode=True


def load_runtime(job):
    rt=C.load_native(job)
    # Exactly the guard used in the source B6 training, only applied in memory.
    record=install_budget_guard(rt.core,int(job['B']),job['sources']['c3nt_cost_sweep_generated_core.py']['sha256'])
    rt.guard=record
    I.need(rt.cfg.mask_mode=='relaxed' and rt.cfg.detour_budget_max==job['B'],'B runtime mismatch')
    return rt


def verify_software(job):
    seen={}
    for name in ('numpy','torch','gymnasium','stable-baselines3','sb3-contrib'):
        got=importlib.metadata.version(name);wanted=job['software'].get(name);seen[name]=got
        I.need(got==wanted,f'Use original dqn_env: {name} expected {wanted}, actual {got}; no automatic upgrade.')
    return seen


def load_model(job,rt):
    verify_software(job)
    path=Path(job['checkpoint']['path'])
    I.need(I.sha(path)==job['checkpoint']['sha256'],'Exact 40M checkpoint changed')
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(int(job['settings'].get('torch_threads',4)))
    snap=Path(job['snapshot']).resolve();f=snap/I.FEATURE
    I.need(I.sha(f)==job['feature_sha256'],'Original CNN module changed')
    name=Path(I.FEATURE).stem
    old=sys.modules.get(name)
    if old is not None and Path(old.__file__).resolve().parent!=snap:sys.modules.pop(name)
    sys.path.insert(0,str(snap));importlib.invalidate_caches()
    try:
        features=importlib.import_module(name)
        I.need(Path(features.__file__).resolve()==f,'Wrong CNN module imported')
        model=MaskablePPO.load(path,device='cpu')
        I.need(int(model.num_timesteps)==I.STEP,'Loaded weight is not requested 40M checkpoint')
        features.audit_cnn3_model(model);rt.core.audit_model(model)
        if job.get('parameter_sha256'):
            I.need(U.tensor_digest(model.get_parameters())==job['parameter_sha256'],'Original model parameter digest differs')
        model.policy.set_training_mode(False)
    finally:sys.path.remove(str(snap))
    print('[MODEL LOADED]',job['key'],'step='+str(model.num_timesteps),'CPU; no optimizer updates',flush=True)
    return model


def annotate(metric,job,rec,ic):
    r=dict(metric)
    r.update(method=job['key'],model=job['key'],label=f"Relaxed B={job['B']} beta=0.5 40M",family='ppo',beta=.5,
        native_reward_beta=.5,effective_B=job['B'],nominal_m=40,checkpoint_step=I.STEP,
        checkpoint_sha256=job['checkpoint']['sha256'],map_id=rec['map_id'],map_sha256=rec['sha256'],
        source=rec['source'],split=rec['split'],pattern=rec['pattern'],tier=rec['tier'],
        initial_pair_id=ic['pair_id'],start_r=ic['start'][0],start_c=ic['start'][1],start_heading=ic['heading'],
        initial_seed=20260907,evaluation_seed=42,deterministic=True,raw_only=True,connection_refinement=False,
        stage='raw_stage1_same_B6_40M_checkpoint',evaluation_phase='validation' if rec['split']=='validation' else 'test')
    return I.safe(r)


def infer_trace(rt,job,rec,array,ic,model=None,decision=None):
    """Same masked probability argmax as b124_evaluation.evaluate_one, saving actions.
    decision is only an internal test hook; there is no CLI for a substitute policy.
    """
    core=rt.factory(rt.cfg)
    ob,_=core.reset(array,start=tuple(ic['start']),heading=int(ic['heading']),seed=42,map_name=rec['map_id'])
    positions=[list(core.drone_pos)];times=[0.];actions=[];masks=[];budgets=[];ent=[];norm=[];counts=[0,0,0,0]
    strict_steps=budget_open=0;B=int(job['B'])
    if core.complete_at_reset:core.step(0)
    while not core._done:
        mask=np.asarray(core.action_masks(),bool).copy();I.need(mask.shape==(4,) and mask.any(),'Empty action mask')
        strict,_=core.strict_mask_details_no_cache();before=int(core.detour_budget)
        I.need(np.array_equal(mask,core.feasible_mask() if before>0 else strict),'Source B mask changed')
        I.need(set(ob)=={'map','heading','detour_budget'},'Wrong input keys/clock')
        I.need(abs(float(np.asarray(ob['detour_budget'])[0])-before/B)<1e-6,'Budget observation mismatch')
        if decision is None:
            I.need(model is not None,'No model for missing Raw action trace')
            import torch
            with torch.no_grad():
                tensor,_=model.policy.obs_to_tensor(ob)
                dist=model.policy.get_distribution(tensor,action_masks=mask.reshape(1,-1))
                probabilities=dist.distribution.probs[0].detach().cpu().numpy().astype(float)
            I.need(np.isfinite(probabilities).all() and abs(probabilities.sum()-1)<1e-5 and np.all(probabilities[~mask]<1e-6),'Bad masked probabilities')
            action=int(np.argmax(probabilities));positive=probabilities[probabilities>0]
            entropy=-float(np.sum(positive*np.log(positive)));ent.append(entropy)
            if mask.sum()>1:norm.append(entropy/math.log(int(mask.sum())))
        else:action=int(decision(core,ob,mask))
        I.need(0<=action<4 and mask[action],'Action violates current B mask')
        K=int(mask.sum());counts[K-1]+=1;strict_steps+=int(strict[action]);budget_open+=int(before>0)
        actions.append(action);masks.append(mask);budgets.append(before)
        ob,_,term,trunc,info=core.step(action)
        expected=B if bool(info.get('is_new_active_cell',0)) else before if strict[action] else max(0,before-1)
        I.need(int(core.detour_budget)==expected,'B decrement/replenishment changed')
        I.need(not(term and trunc) and len(actions)<=core.max_steps,'Native stop inconsistency')
        positions.append(list(core.drone_pos));times.append(float(core.elapsed_time_sec))
    core.assert_consistent();r=annotate(core.info(final=True),job,rec,ic);steps=len(actions)
    r.update(decision_steps=steps,masked_entropy_mean=U.mean(ent),multiaction_normalized_entropy_mean=U.mean(norm),
        allowed_actions_sum=sum((k+1)*v for k,v in enumerate(counts)),
        allowed_actions_mean=sum((k+1)*v for k,v in enumerate(counts))/steps if steps else None,
        chosen_strict_ratio=strict_steps/steps if steps else None,budget_open_ratio=budget_open/steps if steps else None)
    for k,v in enumerate(counts,1):r[f'allowed_actions_k{k}_count']=v
    trace=dict(priority_map=np.asarray(array).copy(),actions=np.asarray(actions,np.int8),positions=np.asarray(positions,np.int16).reshape(-1,2),
        time_sec=np.asarray(times,float),pre_action_masks=np.asarray(masks,bool).reshape(-1,4),budget_before=np.asarray(budgets,np.int16),
        first_visit_step=core.first_visit_step.copy(),first_visit_time_sec=core.first_visit_time.copy())
    return I.safe(r),trace


def verify_result(directory,identity,rec):
    co=I.receipt_read(directory,'complete.json',identity)
    if co is None:return None
    I.need(co.get('map_id')==rec['map_id'] and co.get('map_sha256')==rec['sha256'],'Completed map identity changed')
    required={'raw_metrics.json','refined_metrics.json','raw_trace.npz','refined_trace.npz','refinement_audit.json','segments.csv','raw_complete.json','raw_provenance.json','result.json'}
    I.need(required<=set(co.get('files',{})),'Incomplete final receipt')
    raw=I.read(Path(directory)/'raw_metrics.json');ref=I.read(Path(directory)/'refined_metrics.json');audit=I.read(Path(directory)/'refinement_audit.json')
    for r in (raw,ref):I.need(r.get('map_id')==rec['map_id'] and r.get('map_sha256')==rec['sha256'],'Cached result map changed')
    I.need(I.flag(raw['is_success'])==I.flag(ref['is_success']),'Success changed across refinement')
    I.need(audit.get('order_preserved') and audit.get('arrival_heading_preserved') and audit.get('no_time_increase'),'Missing refinement validation')
    return raw,ref,audit


def process_map(rt,job,rec,out,identity,model_getter,cached_only=False,decision=None):
    d=Path(out)/'methods'/job['key']/'maps'/I.map_key(rec)
    previous=verify_result(d,identity,rec)
    if previous is not None:return (*previous,'completed_refinement_reused')
    old=I.receipt_read(d,'raw_complete.json',identity)
    if old is not None:
        I.need({'raw_metrics.json','raw_trace.npz','raw_provenance.json'}<=set(old['files']),'Raw receipt incomplete')
        raw=I.read(d/'raw_metrics.json');trace=I.load_npz(d/'raw_trace.npz');origin='saved_raw_trace_reused'
        I.need(raw['checkpoint_sha256']==job['checkpoint']['sha256'] and int(raw['checkpoint_step'])==I.STEP and int(raw['effective_B'])==job['B'],'Wrong saved checkpoint/B')
    else:
        I.need(not cached_only,'Raw action trace missing; original B6 has metrics only. Same 40M CPU inference is required for '+job['key']+' '+rec['map_id'])
        array=np.load(rec['path'],allow_pickle=False)
        I.need(rt.utils.array_hash(array)==rec['sha256'],'Original map content changed: '+rec['map_id'])
        ic=U.initial_condition(rt,SimpleNamespace(**rec),array)
        model=None if decision is not None else model_getter()
        raw,trace=infer_trace(rt,job,rec,array,ic,model,decision)
        check=I.compare_metrics(raw,job['historical_rows'][rec['map_id']])
        I.save_npz(d/'raw_trace.npz',trace);I.write(d/'raw_metrics.json',raw)
        I.write(d/'raw_provenance.json',dict(origin='same_registered_B6_40M_checkpoint_cpu_argmax',
            checkpoint_sha256=job['checkpoint']['sha256'],raw_comparison=check,optimizer_updates=0,raw_trace_has_refinement=False))
        I.write(d/'raw_complete.json',dict(identity=identity,map_id=rec['map_id'],map_sha256=rec['sha256'],
            files={n:I.sha(d/n) for n in ('raw_metrics.json','raw_trace.npz','raw_provenance.json')}))
        origin='same_40M_checkpoint_raw_trace_reconstructed'
    array=trace['priority_map'];I.need(rt.utils.array_hash(array)==rec['sha256'],'Saved Raw map hash changed')
    ic=U.initial_condition(rt,SimpleNamespace(**rec),array)
    I.need(raw['initial_pair_id']==ic['pair_id'] and [int(raw['start_r']),int(raw['start_c'])]==list(ic['start']) and int(raw['start_heading'])==ic['heading'],'Saved Raw initialization mismatch')
    check=I.compare_metrics(raw,job['historical_rows'][rec['map_id']])
    ref,reftrace,audit,segments=C.common_refine(rt,raw,trace)
    ref.update(method=job['key'],model=job['key'],connection_refinement=True,raw_only=False,
               evaluation_phase=raw['evaluation_phase'],reference_Raw_compared=True)
    audit.update(method=job['key'],model=job['key'],map_id=rec['map_id'],map_sha256=rec['sha256'],
        source=rec['source'],split=rec['split'],pattern=rec['pattern'],tier=rec['tier'],
        checkpoint_step=I.STEP,checkpoint_sha256=job['checkpoint']['sha256'],native_guard_patch=rt.guard['patch_version'],
        original_raw_comparison=check,raw_origin=origin,optimizer_updates=0)
    I.save_npz(d/'refined_trace.npz',reftrace);I.write(d/'refined_metrics.json',ref)
    I.write(d/'refinement_audit.json',audit);I.csvwrite(d/'segments.csv',segments)
    result=dict(identity=identity,raw=raw,refined=ref,audit=audit,
        input_raw=dict(kind='same_exact40M_CPU_reproduction',trace_path=str((d/'raw_trace.npz').resolve()),trace_sha256=I.sha(d/'raw_trace.npz')),
        refined_trace_path=str((d/'refined_trace.npz').resolve()),refined_trace_sha256=I.sha(d/'refined_trace.npz'))
    I.write(d/'result.json',result)
    names=('result.json','raw_metrics.json','refined_metrics.json','raw_trace.npz','refined_trace.npz','refinement_audit.json','segments.csv','raw_complete.json','raw_provenance.json')
    I.write(d/'complete.json',dict(identity=identity,map_id=rec['map_id'],map_sha256=rec['sha256'],files={n:I.sha(d/n) for n in names}))
    return I.safe(raw),I.safe(ref),I.safe(audit),origin


def worker(context,identity,shard=0,nshards=1,cached_only=False,decision=None):
    job=context['jobs']['B6'];rt=load_runtime(job);base=Path(context['out'])/'methods'/'B6';base.mkdir(parents=True,exist_ok=True)
    work=base/'workers';work.mkdir(exist_ok=True)
    records=job['records'][shard::nshards]
    I.need(0<=shard<nshards and records,'Empty/invalid CPU worker assignment')
    I.write(work/f'part_{shard}_budget_guard.json',rt.guard)
    model=None;started=time.monotonic();origins={};success=changed=0
    def get_model():
        nonlocal model
        if model is None:model=load_model(job,rt)
        return model
    try:
        with I.lock(work/f'.part_{shard}.lock'):
            for i,rec in enumerate(records,1):
                d=base/'maps'/I.map_key(rec)
                with I.lock(d/'.map.lock'):
                    raw,ref,audit,origin=process_map(rt,job,rec,context['out'],identity,get_model,cached_only,decision)
                success+=I.flag(raw['is_success']);changed+=bool(audit['path_changed']);origins[origin]=origins.get(origin,0)+1
                if i==1 or i%10==0 or i==len(records):
                    I.write(work/f'part_{shard}_progress.json',dict(identity=identity,shard=shard,nshards=nshards,n_maps=len(records),
                        completed=i,success=success,changed=changed,elapsed_sec=time.monotonic()-started,origins=origins,
                        status='COMPLETE' if i==len(records) else 'RUNNING'))
                    print(f"[REFINE] B6 part{shard+1}/{nshards} {i}/{len(records)} {I.domain(rec)}; success={success}; changed={changed}; elapsed={time.monotonic()-started:.1f}s",flush=True)
    finally:model=None
    return dict(shard=shard,completed=len(records),origin_counts=origins)

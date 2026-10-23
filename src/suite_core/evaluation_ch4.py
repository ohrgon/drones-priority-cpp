"""One trajectory evaluator for all Chapter 4 planners, never filters classical paths.

Persisted evaluation rows identify method, reward case, beta, training seed, mapping,
initialization, manifest/map hash, checkpoint hash, and the actual first visit of every
active cell. Cached rows are only reused if the full evaluation identity is unchanged.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,math,time,os
from copy import deepcopy
import numpy as np
from core_ch4 import ExperimentCore,ExperimentEnvConfig,make_core
from planners_ch4 import ClassicalPlanner
from full_v2_utils import write_json,write_csv,read_json,array_hash,file_hash,pattern_family

CLASSICAL={'bcd','priority_dijkstra','priority_stc','weighted_latency_greedy'}

def stable_seed(*items):
    return int(hashlib.sha256(json.dumps(items,sort_keys=True,default=str).encode()).hexdigest()[:8],16)

def initial_condition(record,base_cfg,mode,repeat,seed):
    c=ExperimentCore(replace(base_cfg,initialization=mode))
    a=np.load(record.path,allow_pickle=False)
    c.reset(a,seed=stable_seed('initial',record.sha256,mode,repeat,seed))
    return dict(start=list(c.start_pos),heading=c.start_heading,initialization=mode,
                pair_id=hashlib.sha256(json.dumps([record.sha256,mode,repeat,seed,c.start_pos,c.start_heading],sort_keys=True).encode()).hexdigest())

class SB3Decision:
    def __init__(self,model,method='ppo'): self.model=model; self.method=method
    def act(self,obs,mask,deterministic=True,**kwargs):
        import torch
        with torch.no_grad():
            ob,_=self.model.policy.obs_to_tensor(obs)
            if self.method=='a2c': dist=self.model.policy.get_distribution(ob)
            else: dist=self.model.policy.get_distribution(ob,action_masks=mask[None])
            p=dist.distribution.probs[0].detach().cpu().numpy().astype(float)
            # Deterministic selection from precisely the distribution used by sampling.
            if deterministic: a=int(p.argmax())
            else: a=int(torch.multinomial(dist.distribution.probs,1)[0,0].item())
            return a,a,p


def load_decision(checkpoint,method,device='auto'):
    if method in {'ppo','a2c'}:
        if method=='ppo':
            from sb3_contrib import MaskablePPO
            model=MaskablePPO.load(checkpoint,device=device)
        else:
            from stable_baselines3 import A2C
            model=A2C.load(checkpoint,device=device)
        return SB3Decision(model,method)
    from agents_ch4 import NativeAgent
    return NativeAgent.load(checkpoint,device)


def evaluate_one(record,cfg,**kwargs):
    """A map/repetition has its own stochastic stream, independent of skipped caches."""
    agent=kwargs.get('agent');saved_rng=None;agent_rng=None
    if agent is not None:
        from resume_ch4 import capture_rng,restore_rng
        import torch
        saved_rng=capture_rng()
        seed=stable_seed('evaluation-map',kwargs.get('seed',42),record.sha256,
                         kwargs.get('method','ppo'),kwargs.get('repeat',0),
                         kwargs.get('initialization','priority_mass'),kwargs.get('initial_repeat',0))
        torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        if hasattr(agent,'rng'):
            agent_rng=deepcopy(agent.rng.bit_generator.state)
            agent.rng.bit_generator.state=np.random.default_rng(seed).bit_generator.state
    try:
        return _evaluate_one(record,cfg,**kwargs)
    finally:
        if saved_rng is not None:
            restore_rng(saved_rng)
        if agent_rng is not None:
            agent.rng.bit_generator.state=agent_rng


def _evaluate_one(record,cfg,*,method='ppo',agent=None,seed=42,training_seed=None,
                 repeat=0,initialization='priority_mass',deterministic=True,
                 comparison_id=None,run_metadata=None,planner_options=None,output_dir=None,
                 plot=False,initial_seed=20260907,initial_repeat=0,stress_mask_removal=False):
    if method in CLASSICAL: cfg=replace(cfg,mask_mode='boundary')
    if method in {'random','greedy'}: cfg=replace(cfg,mask_mode='progress')
    if stress_mask_removal: cfg=replace(cfg,mask_mode='none')
    cfg=replace(cfg,initialization=initialization)
    a=np.load(record.path,allow_pickle=False)
    if array_hash(a)!=record.sha256: raise ValueError(f"Changed map {record.path}")
    core=make_core(cfg)
    ic=initial_condition(record,cfg,initialization,initial_repeat,initial_seed)
    obs,_=core.reset(a,start=tuple(ic['start']),heading=ic['heading'],seed=seed,map_name=record.map_id)
    rng=np.random.default_rng(stable_seed(seed,record.sha256,method,repeat))
    planner=ClassicalPlanner(method,planner_options) if method in CLASSICAL else None
    positions=[core.drone_pos]; times=[0.]; lengths=[0.]; priorities=[1-core.remaining_priority_mass/core.total_priority_mass]
    actions=[]; masks=[]; probs=[]; rewards=[]; entropies=[]; norm_entropies=[]; disagreements=[]
    mask_seconds=policy_seconds=0.; error=None
    while not (core._done or core.complete_at_reset):
        t=time.perf_counter(); mask=core.action_masks(); mask_seconds+=time.perf_counter()-t
        t=time.perf_counter(); p=None
        if planner:
            action=planner.act(core)
            if action is None:
                error='planner_exhausted'; break
        elif method=='random':
            p=mask.astype(float)/mask.sum(); action=int(rng.choice(4,p=p))
        elif method=='greedy':
            action=core.greedy_action(); p=np.eye(4)[action]
        else:
            if agent is None: raise ValueError('Learned method requires a loaded agent')
            action,_,p=agent.act(obs,mask,deterministic=deterministic)
        policy_seconds+=time.perf_counter()-t
        if p is not None:
            p=np.asarray(p,dtype=float)
            if np.any(p[~mask]>1e-6) or abs(p.sum()-1)>1e-5: raise ValueError('Policy probability/mask mismatch')
            h=-float(np.sum(p[p>0]*np.log(p[p>0]))); entropies.append(h)
            if mask.sum()>1:
                norm_entropies.append(h/math.log(mask.sum()))
                if method=='ppo': disagreements.append(int(action!=core.greedy_action()))
        obs,r,_,_,_=core.step(int(action))
        positions.append(core.drone_pos); times.append(core.elapsed_time_sec); lengths.append(core.path_length_m)
        priorities.append(1-core.remaining_priority_mass/core.total_priority_mass)
        actions.append(action); masks.append(mask); probs.append(np.full(4,np.nan) if p is None else p); rewards.append(r)
    core.assert_consistent(); result=core.info(final=True)
    result.update(dict(map_id=record.map_id,map_path=record.path,map_sha256=record.sha256,source=record.source,
        pattern=record.pattern,tier=record.tier,family=pattern_family(record.pattern,record.source),
        split=record.split,method=method,comparison_id=comparison_id or method,
        training_seed=training_seed,evaluation_seed=seed,repeat=repeat,initial_repeat=initial_repeat,initialization=initialization,
        initial_pair_id=ic['pair_id'],deterministic=deterministic,
        masked_entropy_mean=float(np.mean(entropies)) if entropies else None,
        masked_entropy_normalized_multiaction=float(np.mean(norm_entropies)) if norm_entropies else None,
        ppo_greedy_disagreement_multiaction=float(np.mean(disagreements)) if disagreements else None,
        mask_compute_time_sec=mask_seconds,policy_compute_time_sec=policy_seconds,planner_error=error,
        probability_diagnostic='categorical' if entropies else 'not a categorical policy; entropy not fabricated',
        stress_mask_removal=stress_mask_removal))
    result.update(run_metadata or {})
    if output_dir:
        d=Path(output_dir);d.mkdir(parents=True,exist_ok=True)
        temporary=d/'trajectory.writing.npz'
        np.savez_compressed(temporary,**core.snapshot(),positions=np.asarray(positions,np.int32),
            time_sec=np.asarray(times),path_length_m=np.asarray(lengths),cumulative_priority_ratio=np.asarray(priorities),
            actions=np.asarray(actions,np.int8),pre_action_masks=np.asarray(masks,bool).reshape(-1,4),
            probabilities=np.asarray(probs,float).reshape(-1,4),step_rewards=np.asarray(rewards))
        events=[dict(row=int(r),col=int(c),priority=int(core.weight_map[r,c]),visited=bool(core.covered_map[r,c]),
                     first_step=int(core.first_visit_step[r,c]),first_time_sec=float(core.first_visit_time[r,c]),
                     first_distance_m=float(core.first_visit_distance[r,c])) for r,c in np.argwhere(core.active_mask)]
        from resume_ch4 import sync_file,sync_directory
        sync_file(temporary);os.replace(temporary,d/'trajectory.npz')
        write_csv(d/'first_visits.csv',events);write_json(d/'initial_condition.json',ic)
        if plot:
            plot_one(a,positions,times,priorities,d,result['comparison_id'])
        write_json(d/'metrics.json',result)
        artifacts=['trajectory.npz','first_visits.csv','initial_condition.json','metrics.json']
        if plot:artifacts+=['route.png','priority.png']
        write_json(d/'map_complete.json',dict(files={name:file_hash(d/name) for name in artifacts}))
        sync_directory(d)
    return result


def plot_one(a,positions,times,qs,out,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xy=np.asarray(positions)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(a,origin='upper',vmin=0,vmax=5)
    ax.plot(xy[:,1],xy[:,0],linewidth=1,label='Trajectory');ax.scatter(xy[0,1],xy[0,0],marker='o',label='Start');ax.scatter(xy[-1,1],xy[-1,0],marker='x',label='End')
    ax.legend();ax.set(xlabel='Column',ylabel='Row',title=title);fig.colorbar(im,ax=ax,label='Priority')
    fig.tight_layout();fig.savefig(Path(out)/'route.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.step(times,100*np.asarray(qs),where='post')
    ax.set(xlabel='Elapsed mission time (s)',ylabel='Visited priority (%)',ylim=(0,101),title=title);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(Path(out)/'priority.png',dpi=160);plt.close(fig)


def evaluate_records(records,cfg,*,method='ppo',agent=None,output_dir,identity=None,
                     repetitions=1,initializations=('priority_mass',),initial_condition_repetitions=1,plot_maps=0,**kwargs):
    d=Path(output_dir);d.mkdir(parents=True,exist_ok=True)
    identity=dict(identity or {},stochastic_seed_scheme='per_map_method_repeat_init_v2',environment=asdict(cfg),method=method,repetitions=repetitions,
                  initializations=list(initializations),initial_condition_repetitions=initial_condition_repetitions,records=[(r.map_id,r.sha256,r.split) for r in records],
                  evaluation_kwargs=kwargs)
    expected=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
    plan=d/'evaluation_identity.json'
    if plan.exists() and read_json(plan)['sha256']!=expected: raise ValueError(f'Evaluation configuration changed: {d}')
    write_json(plan,dict(sha256=expected,parameters=identity)); rows=[]
    for init in initializations:
        for initrep in range(initial_condition_repetitions if init == 'random_active' else 1):
            for i,record in enumerate(records):
                for rep in range(repetitions):
                    key=f'{record.sha256[:20]}_{init}_i{initrep}_r{rep}'
                    rd=d/'maps'/key; cache=rd/'metrics.json'
                    complete=rd/'map_complete.json'
                    if cache.exists() and complete.exists():
                        for name,expected_hash in read_json(complete)['files'].items():
                            if not (rd/name).exists() or file_hash(rd/name)!=expected_hash:
                                raise ValueError(f'Completed map evaluation artifact changed: {rd/name}')
                        row=read_json(cache)
                    else:
                        row=evaluate_one(record,cfg,method=method,agent=agent,output_dir=rd,initialization=init,
                                         repeat=rep,initial_repeat=initrep,plot=i<plot_maps,**kwargs)
                    rows.append(row);write_csv(d/'per_map.csv',rows)
                    if len(rows)%25==0 or len(rows)==1:
                        print(f'[EVAL MAPS] {d.name}: {len(rows)} completed/cached',flush=True)
    write_json(d/'evaluation_complete.json',dict(identity_sha256=expected,rows=len(rows)))
    return rows


def summary(rows):
    success=[r for r in rows if r['is_success']]
    return dict(n_maps=len(rows),success_rate=float(np.mean([r['is_success'] for r in rows])),
                mean_coverage_pct=float(np.mean([r['coverage_pct'] for r in rows])),
                latency_sec=float(np.mean([r['priority_weighted_first_visit_latency_sec'] for r in success])) if success else None,
                mission_time_sec=float(np.mean([r['mission_time_sec'] for r in success])) if success else None,
                failure_penalized_time_sec=float(np.mean([r['failure_penalized_mission_time_sec'] for r in rows])))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COST11 Strict beta=0,0.5,1 -> TOTAL 50M; validation every 1M.

Run beside the original uav_chapter4_suite_v2:
  python run_strict_50m.py --root . --gpus 1 2 3
  python run_strict_50m.py --root . --phase check --gpus 1 2 3

Original frozen COST11 core/CNN/logger/trainer are required. They are copied and
hash checked, never replaced with generic coverage_core or a newly built network.
The original trainer performs PPO updates and durable full-state recovery.
New validation runs in a separate CPU subprocess after every optimizer-boundary
1M checkpoint. The corresponding training worker waits; its RNG is untouched.
Old checkpoints are evaluated after training (or with --phase validate).
No training or test data are used as validation; no test/OSM rollout is run.

Only use your own trusted checkpoints (SB3/PyTorch files deserialize Python).
Requires the SAME Python/package environment as the original training run.
"""
from __future__ import annotations
import argparse, ast, csv, gc, hashlib, html, importlib, importlib.metadata
import io, json, math, os, shutil, signal, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
from copy import deepcopy
from contextlib import contextmanager
from collections import Counter, defaultdict
from types import SimpleNamespace

VERSION = 'cost11_strict50m_native_v2_20261001'
TARGET = 50_000_000
INTERVAL = 1_000_000
LR = 5e-5
DEFAULT_OUT = 'strict_beta_runs_seed42'
CASES = {'strict_b0': 0.0, 'strict_b05': 0.5, 'strict_b1': 1.0}
TRAINER = 'trainers_ch4_c3nt_cost_sweep'
CORE = 'c3nt_cost_sweep_generated_core'
FEATURES = 'nrg1_c3nt_cost5_features'
LOGGER = 'c3nt_cost_sweep_generated_logging'
TRAINER_SHA = '0049fb0f18f07cd80f2ec96a6ad16510843d01dcd1703ec7250435c80908585e'
CORE_VERSION = 'cnn3_noclock_cost11_strict_progress_v1_20260929'
REQUIRED_RUNTIME = [x+'.py' for x in (TRAINER, CORE, FEATURES, LOGGER,
                    'core_ch4', 'core_ch4_bounded_b6', 'coverage_core',
                    'full_v2_utils', 'resume_ch4')]
T = 'mission_time_sec'
LW = 'priority_weighted_first_visit_latency_sec'
METRICS = [T,LW,'priority_arrival_25_sec','priority_arrival_50_sec',
           'priority_arrival_75_sec','priority_arrival_90_sec','path_length_m',
           'turn_count','revisit_steps','zero_transit_steps','nrg_detour_ratio',
           'masked_entropy_mean','multi_action_normalized_entropy_mean']
DIAGNOSTICS = ['learning_rate','approx_kl','clip_fraction','entropy_loss',
               'explained_variance','value_loss','policy_gradient_loss']


def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def safe(v):
    if isinstance(v,Path): return str(v)
    if isinstance(v,dict): return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [safe(x) for x in v]
    if hasattr(v,'tolist'): return safe(v.tolist())
    if hasattr(v,'item'): return safe(v.item())
    if isinstance(v,float) and not math.isfinite(v): return None
    return v

def digest(v):
    return hashlib.sha256(json.dumps(safe(v),sort_keys=True,separators=(',',':'),
                          ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def write_text(p,text):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    require(not p.is_symlink(),'Symlink output refused: '+str(p))
    tmp=p.with_name(p.name+f'.writing.{os.getpid()}')
    try:
        with tmp.open('w',encoding='utf-8',newline='') as f:
            f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()

def write(p,v):
    write_text(p,json.dumps(safe(v),ensure_ascii=False,sort_keys=True,indent=2,
                           allow_nan=False)+'\n')

def freeze(p,v):
    if Path(p).exists():require(read(p)==safe(v),'Frozen input changed: '+str(p))
    else:write(p,v)

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def csv_write(p,rr):
    fields=list(dict.fromkeys(k for r in rr for k in r)) or ['status']
    s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
    for r in rr:
        w.writerow({k:json.dumps(safe(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple))
                    else safe(v) for k,v in r.items()})
    write_text(p,'\ufeff'+s.getvalue())

def number(v):
    try:
        f=float(v);return f if math.isfinite(f) else None
    except (ValueError,TypeError):return None

def truth(v):
    if isinstance(v,bool):return v
    if str(v).lower() in ('1','1.0','true'):return True
    if str(v).lower() in ('0','0.0','false'):return False
    raise ValueError('Invalid success flag: '+repr(v))

def mean(v):
    vv=[number(x) for x in v];vv=[x for x in vv if x is not None]
    return sum(vv)/len(vv) if vv else None

def near(a,b):
    x=number(a);return x is not None and abs(x-b)<1e-10

@contextmanager
def lock(p, blocking=False):
    import fcntl
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as e:raise RuntimeError('Already running: '+str(p)) from e
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def validate_spec(sp,st,beta):
    env=sp.get('environment',{});rd=sp.get('reward_definition',{});p=st.get('ppo',{})
    expected={'mask_mode':'progress','tradeoff_beta':beta,'detour_budget_max':6,
              'step_budget_factor':5.0,'grid_size':50,'coverage_reward_scale':0.0,
              'objective_cost_scale':0.0,'success_terminal_reward':0.0,
              'failure_terminal_penalty':0.0}
    require(sp.get('algorithm')=='ppo' and sp.get('seed')==42 and near(sp.get('beta'),beta),
            'Not requested Strict PPO beta/seed')
    for k,v in expected.items():require(env.get(k)==v,'Environment mismatch: '+k)
    require(rd.get('first_visit')=='delta_new_active_count / active_cell_count'
            and near(rd.get('cost_multiplier'),11) and near(rd.get('success_terminal'),2)
            and near(rd.get('failure_terminal'),0),'Not FULL COST11 replacement reward')
    require(st.get('num_envs')==8 and near(p.get('gamma'),1) and near(p.get('gae_lambda'),.95)
            and near(p.get('learning_rate_floor'),LR),'Wrong PPO settings')


def source_candidates(root,beta):
    tag={0.0:'beta0',.5:'beta0p5',1.0:'beta1'}[beta]
    hints=[root/'chapter4_C3NT_COST11_STRICT_BETA_0_05_1_seed42'/tag]
    if beta==.5:hints += [root/'chapter4_COST11_5M_to_20M_seed42'/'strict_b05']
    if beta==0:hints += [root/'chapter4_COST11_extra_betas_5M_to_20M_seed42'/'strict_b0']
    found=[]
    for base in hints:
        for rp in sorted((base/'runs').glob('*/run_spec.json')):
            try:
                run=rp.parent;sp=read(rp);st=read(run/'training_settings.json');status=read(run/'status.json')
                validate_spec(sp,st,beta)
                if status.get('state')!='completed':
                    print('[SOURCE SKIP unfinished]',run,flush=True);continue
                if int(status.get('actual_timesteps',0))>=TARGET+8192:continue
                found.append((base,run,sp,st,status))
            except (RuntimeError,KeyError,ValueError,FileNotFoundError) as exc:
                print('[SOURCE SKIP]',run,':',exc,flush=True)
    require(found,f'No completed COST11 Strict beta={beta:g} source under '+str(root))
    return found


def checkpoint_path(run,base,cp):
    name=Path(str(cp.get('path','')).replace('\\','/')).name
    if cp.get('source_baseline'):
        choices=[base/'source_5m'/'last_model.zip',base/'source_model'/'last_model.zip']
    elif cp.get('final'):choices=[run/'last_model.zip']
    else:choices=[run/'checkpoints'/name,run/name]
    p=next((x for x in choices if x.is_file()),None)
    require(p is not None,'Missing registered checkpoint: '+str(choices))
    require(not p.is_symlink() and sha(p)==cp['sha256'],'Checkpoint hash mismatch: '+str(p))
    return p.resolve()


def inspect_source(item,beta):
    base,run,sp,st,status=item
    proto_path=base/('continuation_protocol.json' if (base/'continuation_protocol.json').is_file() else 'protocol.json')
    proto=read(proto_path);hashes=proto['snapshot_hash'];snapshot=base/'source_snapshot'
    for name in REQUIRED_RUNTIME:
        require(name in hashes and (snapshot/name).is_file() and sha(snapshot/name)==hashes[name],
                'Frozen COST11 runtime missing/changed: '+str(snapshot/name))
    require(hashes[TRAINER+'.py']==TRAINER_SHA,'Unsupported native trainer revision; refusing blind patch')
    # Include only authenticated local Python modules; no generic core fallback.
    runtime_hashes={n:h for n,h in hashes.items() if n.endswith('.py') and (snapshot/n).is_file()}
    for n,h in runtime_hashes.items():require(sha(snapshot/n)==h,'Snapshot changed: '+n)
    manifest=base/'manifest.csv'
    expected_manifest=proto.get('manifest_sha256',proto.get('source_manifest_sha256'))
    require(expected_manifest and sha(manifest)==expected_manifest,'Manifest/protocol hash mismatch')
    mm=rows(manifest)
    require(Counter((r['source'],r['split']) for r in mm)==
            {('synthetic','train'):900,('synthetic','validation'):180,('synthetic','test'):720,('osm','test'):500},
            'Original 900 train / 180 validation / 720 synthetic test / 500 OSM split changed')
    for field in ('map_id','path','sha256'):
        require(len({r[field] for r in mm})==len(mm),'Duplicate or cross-split data: '+field)
    regs=read(run/'checkpoints.json');last=[r for r in regs if r.get('final')]
    require(len(last)==1,'No unique registered final model: '+str(run))
    cp=last[0];model=checkpoint_path(run,base,cp);step=int(cp['num_timesteps'])
    require(step==int(status['actual_timesteps']),'Source status/checkpoint step mismatch')
    with zipfile.ZipFile(model) as z:
        require({'data','policy.pth','policy.optimizer.pth'}<=set(z.namelist()),'Incomplete policy/Adam ZIP')
        data=json.loads(z.read('data'))
    require(int(data['num_timesteps'])==step,'SB3 metadata step mismatch')
    for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm','target_kl'):
        require(data[k]==st['ppo'][k],'SB3/source settings mismatch: '+k)
    require(data['n_envs']==st['num_envs'] and data['seed']==42,'SB3 seed/env count mismatch')
    require(data['_n_updates']==status['ppo_epoch_counter'],'SB3/source PPO epoch mismatch')
    protected={str(manifest.resolve()):sha(manifest),str(proto_path.resolve()):sha(proto_path),str(model):sha(model)}
    for n in ['run_spec.json','training_settings.json','status.json','checkpoints.json','software.json',
              'training_updates.csv']+[f'map_order_rank{i}.csv' for i in range(st['num_envs'])]:
        p=run/n;require(p.is_file(),'Missing source '+str(p));protected[str(p.resolve())]=sha(p)
    return dict(source_root=str(base.resolve()),source_run=str(run.resolve()),
                source_snapshot=str(snapshot.resolve()),source_model=str(model),source_step=step,
                source_sha256=sha(model),manifest=str(manifest.resolve()),manifest_sha256=sha(manifest),
                source_protocol_sha256=sha(proto_path),spec=sp,settings=st,status=status,
                software=read(run/'software.json'),runtime_hashes=runtime_hashes,protected=protected)


def make_plan(args):
    root=args.root.resolve();out=args.out.resolve();pp=out/'plan.json'
    if pp.exists():
        plan=read(pp)
        require(plan['version']==VERSION and plan['launcher_sha256']==sha(__file__),
                'Existing output belongs to different code. Do not mix versions.')
        require(plan['gpus']==args.gpus and plan['root']==str(root),
                'Use the same root and GPU mapping to resume')
        return plan
    out.mkdir(parents=True,exist_ok=True)
    jobs={};val_signature=None
    for (key,beta),gpu in zip(CASES.items(),args.gpus):
        candidates=source_candidates(root,beta)
        highest=max(int(x[4]['actual_timesteps']) for x in candidates)
        chosen=[x for x in candidates if int(x[4]['actual_timesteps'])==highest]
        require(len(chosen)==1,'Ambiguous equally advanced sources for '+key)
        job=inspect_source(chosen[0],beta);job.update(key=key,beta=beta,gpu=gpu,case_dir=str(out/key))
        vv=sorted((r['map_id'],r['sha256']) for r in rows(job['manifest']) if r['split']=='validation')
        if val_signature is None:val_signature=vv
        require(vv==val_signature,'The three models do not share the same validation maps')
        # Record historical checkpoint lineage; never train from the best validation model.
        history={}
        for base,run,sp,st,status in sorted(candidates,key=lambda x:int(x[4]['actual_timesteps'])):
            for cp in read(run/'checkpoints.json'):
                if cp.get('source_baseline'):continue
                step=int(cp['num_timesteps']);m=step//INTERVAL
                if not (1<=m<=50 and m*INTERVAL<=step<m*INTERVAL+st['num_envs']*st['ppo']['n_steps']):continue
                p=checkpoint_path(run,base,cp)
                entry=dict(path=str(p),sha256=cp['sha256'],num_timesteps=step,nominal_m=m,origin='history')
                # Prefer final over periodic duplicate at exactly the same model step.
                if m not in history or cp.get('final'):history[m]=entry
        job['history']=list(history.values());jobs[key]=job
    plan=dict(version=VERSION,root=str(root),out=str(out),target=TARGET,interval=INTERVAL,
              learning_rate=LR,gpus=args.gpus,launcher_sha256=sha(__file__),jobs=jobs)
    write(pp,plan);return plan


def verify_job(job):
    for p,h in job['protected'].items():require(Path(p).is_file() and sha(p)==h,'Source changed: '+p)


def derived(job):
    st=deepcopy(job['settings']);sp=deepcopy(job['spec'])
    e=dict(source_model=str(Path(job['case_dir'])/'source_model'/'last_model.zip'),
           source_sha256=job['source_sha256'],source_step=job['source_step'],
           source_epoch_counter=job['status']['ppo_epoch_counter'],
           source_episodes=job['status']['episodes'],target=TARGET,lr=LR)
    # Source metadata remains in plan.json; do not mislabel the old protocol as this run.
    for key in ('continuation', 'continuation_protocol_sha256', 'nrg1_protocol_sha256'):
        sp.pop(key, None)
    st.pop('continuation', None)
    sp['source_spec_sha256'] = digest(job['spec'])
    # Reset only the continuation description, NOT environment/CNN/PPO hyperparameters.
    st.update(total_timesteps=TARGET,device='cuda:0',checkpoint_every_timesteps=INTERVAL,
              retain_completed_resume=True,strict50m=e)
    st['ppo']['learning_rate_initial']=LR;st['ppo']['learning_rate_floor']=LR
    sp.update(run_id=f'ppo_progress_COST11_FULL_b{job["beta"]:g}_seed42_to50M_v2',
              configuration_id=f'strict_COST11_b{job["beta"]:g}_seed42_to50M_v2',
              stage='strict_50m_continuation',strict50m=e)
    return sp,st


def patch_trainer(source):
    tree=ast.parse(source)
    fs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='train_sb3']
    require(len(fs)==1,'One native train_sb3 definition required');f=fs[0]
    text='\n'.join(source.splitlines()[f.lineno-1:f.end_lineno])+'\n'
    changes=[
      ("checkpoints = deepcopy(state['checkpoints']) if state else []", "checkpoints = deepcopy(state['checkpoints']) if state else _s50_baseline(settings)"),
      ("episode_count = state['episode_count'] if state else 0", "episode_count = state['episode_count'] if state else settings['strict50m']['source_episodes']"),
      ("next_cp = state['next_cp'] if state else cp_interval", "next_cp = state['next_cp'] if state else (settings['strict50m']['source_step']//cp_interval+1)*cp_interval"),
      ("next_resume = state['next_resume'] if state else resume_interval", "next_resume = state['next_resume'] if state else (settings['strict50m']['source_step']//resume_interval+1)*resume_interval"),
      ("                model = SaveAfterUpdate(**common)", "                model = _s50_load(SaveAfterUpdate, env, spec, settings, out, records)"),
      ("reset_num_timesteps=not bool(state)", "reset_num_timesteps=False"),
      ("            super().train()\n", "            _s50_assert(self, spec, settings)\n            super().train()\n"),
      ("            stop.raise_if_requested(self.num_timesteps)\n", "            stop.raise_if_requested(self.num_timesteps)\n            _s50_boundary(self, spec, settings, out, checkpoints, save_resume)\n"),
    ]
    for before,after in changes:
        require(text.count(before)==1,'Native trainer patch pattern changed: '+before)
        text=text.replace(before,after,1)
    compile(text,'strict50m_native_patch.py','exec');return text


def prepare(job):
    case=Path(job['case_dir']);case.mkdir(parents=True,exist_ok=True)
    verify_job(job);snapshot=case/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for n,h in job['runtime_hashes'].items():
        dst=snapshot/n
        if not dst.exists():shutil.copyfile(Path(job['source_snapshot'])/n,dst)
        require(sha(dst)==h,'Copied runtime changed: '+str(dst))
    dst=case/'source_model'/'last_model.zip';dst.parent.mkdir(exist_ok=True)
    if not dst.exists():shutil.copyfile(job['source_model'],dst)
    require(sha(dst)==job['source_sha256'],'Copied source checkpoint changed')
    sp,st=derived(job)
    freeze(case/'continuation_protocol.json',dict(version=VERSION,spec=sp,settings=st,
           runtime_hashes=job['runtime_hashes'],source=job['source_sha256'],manifest=job['manifest_sha256']))
    patch=patch_trainer((snapshot/(TRAINER+'.py')).read_text(encoding='utf-8'))
    p=case/'native_train_sb3_patch.py'
    if p.exists():require(p.read_text()==patch,'Native patch changed')
    else:write_text(p,patch)
    return sp,st


def runtime(job, *, train=False):
    import numpy as np
    import torch
    snap=(Path(job['case_dir'])/'source_snapshot').resolve()
    sys.dont_write_bytecode=True;sys.path.insert(0,str(snap));importlib.invalidate_caches()
    mods=[]
    for n in (TRAINER,'full_v2_utils',CORE,FEATURES,LOGGER,'resume_ch4','core_ch4'):
        mod=importlib.import_module(n)
        require(Path(mod.__file__).resolve().parent==snap,'Non-frozen module imported: '+n);mods.append(mod)
    trainer,utils,core,features,logger,recovery,initial=mods
    for mod in list(sys.modules.values()):
        f=getattr(mod,'__file__',None)
        if f and Path(f).suffix=='.py' and Path(f).resolve().parent==snap:
            require(Path(f).name in job['runtime_hashes'] and sha(f)==job['runtime_hashes'][Path(f).name],
                    'Unverified runtime dependency: '+str(f))
    require(core.VERSION==CORE_VERSION and core.COST_MULTIPLIER==11.0 and
            core.NORMALIZATION_STEP_FACTOR==5.0 and logger.C3NT_VERSION==core.VERSION,'Wrong Strict COST11 runtime')
    for n in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib'):
        actual=importlib.metadata.version(n);expected=job['software'].get(n)
        require(expected is not None and expected==actual,
                f'Use original training environment: {n} expected {expected}, got {actual}. Do not upgrade automatically.')
    torch.set_num_threads(int(job['settings'].get('torch_threads',4)))
    if train:require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Assigned GPU is unavailable')
    logger.install(trainer,utils)
    cfg=trainer.ExperimentEnvConfig(**job['spec']['environment'])
    require(isinstance(trainer.make_core(cfg),core.C3NoClockMixin),'COST11 replacement reward is not installed')
    rr=utils.load_manifest(job['manifest'],50,verify_files=False)
    tr=[r for r in rr if r.split=='train' and r.source=='synthetic']
    va=sorted([r for r in rr if r.split=='validation' and r.source=='synthetic'],key=lambda r:(r.source,str(r.tier),r.pattern,r.map_id))
    require(len(tr)==900 and len(va)==180,'Changed data partition')
    require(trainer.run_identity(job['spec'],job['settings'],tr)==job['status']['identity'],
            'Source train record order/spec/settings identity changed')
    return SimpleNamespace(trainer=trainer,utils=utils,core=core,features=features,
                           recovery=recovery,initial=initial,cfg=cfg,train=tr,val=va)


def rebuild_orders(rt,job):
    """Replay deterministic sampler draws, checking the logged suffix (5M or 20M)."""
    states=[];st=job['settings']
    for rank in range(st['num_envs']):
        rr=rows(Path(job['source_run'])/f'map_order_rank{rank}.csv')
        require(rr,'Missing sampler history')
        pos=[int(r['sequence_position']) for r in rr]
        require(pos==list(range(pos[0],pos[-1]+1)),'Non-contiguous source sampler suffix')
        require(all(int(r['episode'])==p and int(r['rank'])==rank for r,p in zip(rr,pos)),
                'Source episode/sequence/rank mismatch')
        known=dict(zip(pos,rr));order=rt.utils.BalancedMapOrder(rt.train,st['map_sequence_seed']+42*1009+rank)
        for i in range(1,pos[-1]+1):
            rec=rt.train[order.next_index()]
            if i in known:
                r=known[i]
                require(rec.map_id==r['map_id'] and rec.pattern==r['pattern'] and
                        str(rec.tier)==str(r['tier']) and rec.source==r['source'],'Sampler replay differs from original')
        states.append(dict(order=order.state_dict(),episode=pos[-1]))
    return states


def tensor_digest(x):
    import torch
    h=hashlib.sha256()
    def rec(v):
        if isinstance(v,torch.Tensor):
            v=v.detach().cpu().contiguous();h.update((str(v.dtype)+str(tuple(v.shape))).encode());h.update(v.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda a:(type(a).__name__,repr(a))):rec(k);rec(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for a in v:rec(a)
        else:h.update((type(v).__name__+':'+repr(v)+';').encode())
    rec(x);return h.hexdigest()


def assert_model(model,sp,st):
    for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm','target_kl'):
        require(getattr(model,k)==st['ppo'][k],'PPO parameter changed: '+k)
    require(model.n_envs==st['num_envs'] and model.seed==42,'PPO seed/n_envs mismatch')
    require(getattr(model,'strict50m_source_sha256',None)==st['strict50m']['source_sha256'],'Continuation provenance mismatch')
    require(st['strict50m']['source_step']<=model.num_timesteps<TARGET+model.n_envs*model.n_steps,'Invalid total steps')
    for p in (1.,.5,0.,-.001):require(near(model.lr_schedule(p),LR),'Continuation LR restarted')
    for group in model.policy.optimizer.param_groups:require(near(group['lr'],LR),'Optimizer LR changed')


def run_subprocess(plan,key,mode,cp=None):
    """Isolated evaluation process: no torch/RNG/state mutation in the trainer."""
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['MPLBACKEND']='Agg'
    cmd=[sys.executable,str(Path(__file__).resolve()),'--_worker',mode,'--_plan',str(Path(plan['out'])/'plan.json'),'--key',key]
    if cp:cmd+=['--checkpoint',str(cp)]
    result=subprocess.run(cmd,env=env,cwd=plan['root'])
    require(result.returncode==0,f'{mode} failed for {key}; model/recovery and completed map cache are retained')


def registered_new(job):
    sp,_=derived(job);run=Path(job['case_dir'])/'runs'/sp['run_id']
    p=run/'checkpoints.json'
    return (run,read(p) if p.is_file() else [])


def validation_done(plan,key,cp):
    p=Path(plan['out'])/'validation'/key/f"step_{cp['num_timesteps']}"/'complete.json'
    if not p.exists():return False
    d=read(p);base=p.parent
    require(d['checkpoint_sha256']==cp['sha256'],'Completed validation checkpoint changed')
    for n,h in d['files'].items():require(sha(base/n)==h,'Completed validation report changed: '+str(base/n))
    return True


def evaluate_pending(plan,key,history=False):
    job=plan['jobs'][key];run,regs=registered_new(job);cps={}
    if history:
        for cp in job['history']:cps[int(cp['num_timesteps'])]=cp
    # Do not replace a periodic checkpoint with its identical-policy final ZIP:
    # serialized ZIP hashes may differ; select the first registry entry per step.
    for r in regs:
        if r.get('source_baseline'):continue
        step=int(r['num_timesteps'])
        if step in cps:continue
        cp=dict(r,path=str(checkpoint_path(run,Path(job['case_dir']),r)),nominal_m=step//INTERVAL,origin='continuation')
        cps[step]=cp
    for step,cp in sorted(cps.items()):
        if not validation_done(plan,key,cp):run_subprocess(plan,key,'eval',cp['path'])


def train_worker(plan,key,check=False):
    job=plan['jobs'][key];sp,st=prepare(job);rt=runtime(job,train=True)
    import numpy as np
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.save_util import load_from_zip_file
    for rec in rt.train:
        a=np.load(rec.path,allow_pickle=False)
        require(rt.utils.array_hash(a)==rec.sha256,'Training map changed: '+rec.map_id)
    orders=rebuild_orders(rt,job)
    run=Path(job['case_dir'])/'runs'/sp['run_id']
    def baseline(settings):
        e=settings['strict50m']
        return [dict(path=e['source_model'],num_timesteps=e['source_step'],sha256=e['source_sha256'],
                     source_baseline=True,final=False)]
    def load(cls,env,spec,settings,out,records):
        e=settings['strict50m'];source=e['source_model']
        require(sha(source)==e['source_sha256'],'Source model changed')
        _,params,_=load_from_zip_file(source,device='cpu',load_data=False)
        require(params and 'policy' in params and 'policy.optimizer' in params,'Missing full optimizer')
        expected=tensor_digest(params)
        model=cls.load(source,env=env,device='cuda:0',force_reset=True)
        require(tensor_digest(model.get_parameters())==expected,'Policy/Adam not restored exactly')
        require(model.num_timesteps==e['source_step'] and model._n_updates==e['source_epoch_counter'],'Source counters mismatch')
        opt=model.policy.optimizer.state_dict()
        require(opt['state'] and all(float(v['step'])>0 for v in opt['state'].values()),'Missing Adam moments')
        model.learning_rate=LR;model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:group['lr']=LR
        require(tensor_digest(model.get_parameters())==expected,'Source optimizer LR was not at expected floor')
        model.tensorboard_log=str(Path(out)/'tensorboard') if settings.get('tensorboard') else None
        model.strict50m_source_sha256=e['source_sha256'];model._last_obs=None;model._last_episode_starts=None
        for i,adapter in enumerate(env.envs):
            adapter.stream.order.load_state_dict(orders[i]['order']);adapter.stream.episode=orders[i]['episode']
            adapter.stream.total_steps=e['source_step']//settings['num_envs']
        assert_model(model,spec,settings);rt.features.audit_cnn3_model(model);rt.core.audit_model(model)
        write(Path(out)/'load_audit.json',dict(source_step=e['source_step'],policy_adam_exact=True,
              tensor_digest=expected,source_environment_rng_restored=False,subsequent_full_recovery=True,
              sampler_positions=[x['episode'] for x in orders]))
        print(f'[LOAD] {key}: {model.num_timesteps:,} -> TOTAL {TARGET:,}; LR={LR}; policy+Adam restored',flush=True)
        return model
    def boundary(model,spec,settings,out,checkpoints,save_resume):
        cp=next((r for r in checkpoints if not r.get('source_baseline') and not r.get('final') and
                 int(r['num_timesteps'])==int(model.num_timesteps)),None)
        if cp and not validation_done(plan,key,cp):
            save_resume(model,'before_1M_validation')
            print(f'[VALIDATION BOUNDARY] {key} step={model.num_timesteps:,}; CPU validation180',flush=True)
            run_subprocess(plan,key,'eval',cp['path'])
    rt.trainer.__dict__.update(_s50_baseline=baseline,_s50_load=load,_s50_assert=assert_model,_s50_boundary=boundary)
    patch=(Path(job['case_dir'])/'native_train_sb3_patch.py').read_text()
    exec(compile(patch,str(Path(job['case_dir'])/'native_train_sb3_patch.py'),'exec'),rt.trainer.__dict__)
    if check:
        model=MaskablePPO.load(job['source_model'],device='cuda:0')
        _,params,_=load_from_zip_file(job['source_model'],device='cpu',load_data=False)
        require(tensor_digest(model.get_parameters())==tensor_digest(params),'Preflight policy/Adam mismatch')
        require(model.policy.optimizer.state_dict()['state'],'Preflight optimizer has no state')
        rt.features.audit_cnn3_model(model);rt.core.audit_model(model)
        c=rt.trainer.make_core(rt.cfg);obs,_=c.reset(np.load(rt.train[0].path,allow_pickle=False),seed=42,map_name=rt.train[0].map_id)
        require(set(obs)==set(model.observation_space.spaces),'Core/policy observation mismatch')
        a,_=model.predict(obs,deterministic=True,action_masks=c.action_masks())
        _,reward,_,_,info=c.step(int(a))
        require(info.get('nrg_reward_replaced') and math.isfinite(float(reward)),'Native COST11 smoke test failed')
        write(Path(job['case_dir'])/'check.json',dict(passed=True,source_step=model.num_timesteps,
              train_maps=900,validation_maps=180,optimizer_updates=0,core_version=rt.core.VERSION))
        print('[CHECK PASS]',key,'COST11 core/CNN/policy/Adam/sampler',flush=True);return
    # A validation interrupted after a committed checkpoint is completed before learning resumes.
    evaluate_pending(plan,key,history=False)
    rt.trainer.train_run(sp,st,rt.train,run)
    evaluate_pending(plan,key,history=True)
    status_path = run/'status.json'
    status = read(status_path)
    status.update(validation_executed=True, test_executed=False,
                  validation_schedule='each new 1M checkpoint in a separate CPU process; old history backfilled')
    write(status_path,status)
    verify_job(job)


def initial_condition(rt,rec,array):
    from dataclasses import replace
    mode=rt.cfg.initialization
    require(mode=='priority_mass','Unexpected original initialization')
    initial_seed=20260907
    seed=int(hashlib.sha256(json.dumps(('initial',rec.sha256,mode,0,initial_seed),
                     sort_keys=True,default=str).encode()).hexdigest()[:8],16)
    c=rt.initial.ExperimentCore(replace(rt.cfg,initialization=mode));c.reset(array,seed=seed)
    pair=hashlib.sha256(json.dumps([rec.sha256,mode,0,initial_seed,c.start_pos,c.start_heading],
                                 sort_keys=True).encode()).hexdigest()
    return dict(start=list(c.start_pos),heading=c.start_heading,initialization=mode,pair_id=pair)


def validate_row(row,job):
    require(row.get('nrg_version')==CORE_VERSION,'Wrong evaluation core version')
    require(near(row.get('c3nt_cost_multiplier'),11) and near(row.get('c3nt_normalization_step_factor'),5),
            'Wrong evaluated cost/normalization')
    require(int(row['max_steps'])==math.ceil(5*int(row['active_cells'])),'Wrong external safety cap')
    require(int(row['invalid_action_count'])==0 and int(row['progress_violation_count'])==0,'Invalid Strict movement')
    require(number(row.get('nrg_return_identity_error')) is not None and abs(float(row['nrg_return_identity_error']))<1e-6,
            'Native reward identity check failed')
    require(near(row.get('nrg_detour_ratio'),0),'Strict policy made a detour')
    require(number(row.get(T)) is not None and float(row[T])>=0,'Invalid mission time')
    if truth(row['is_success']):
        require(near(row['coverage_pct'],100) and number(row.get(LW)) is not None and
                0<=float(row[LW])<=float(row[T])+1e-7,'Invalid completed-path latency/coverage')
    else:
        require(row.get('termination_reason')=='external_step_cap' and row.get('c3nt_stop_kind')=='safety_cap'
                and truth(row['c3nt_truncated']) and int(row['steps'])==int(row['max_steps'])
                and number(row.get(LW)) is None,'Failure must retain undefined Lw and native safety-cap status')
    if job['beta']==0:require(abs(float(row['reward_sum_latency']))<1e-8,'beta0 received a latency reward')


def evaluate_one(rt,job,model,rec,array,ic):
    import numpy as np
    import torch
    core=rt.trainer.make_core(rt.cfg)
    obs,_=core.reset(array,start=tuple(ic['start']),heading=ic['heading'],seed=42,map_name=rec.map_id)
    ent=[];norm=[];steps=0
    if core.complete_at_reset:core.step(0)
    else:
        while not core._done:
            mask=np.asarray(core.action_masks(),dtype=bool)
            require(mask.shape==(4,) and mask.any(),'Empty/invalid native mask')
            with torch.no_grad():
                ob,_=model.policy.obs_to_tensor(obs)
                dist=model.policy.get_distribution(ob,action_masks=mask.reshape(1,-1))
                p=dist.distribution.probs[0].detach().cpu().numpy().astype(float)
            require(np.isfinite(p).all() and abs(p.sum()-1)<1e-5 and np.all(p[~mask]<1e-6),'Invalid masked distribution')
            action=int(np.argmax(p));require(mask[action],'Masked argmax action')
            h=-float(np.sum(p[p>0]*np.log(p[p>0])));ent.append(h)
            if mask.sum()>1:norm.append(h/math.log(int(mask.sum())))
            obs,_,terminated,truncated,_=core.step(action);steps+=1
            require(steps<=int(core.max_steps)+1 and not(terminated and truncated),'Native stopping rule failed')
    core.assert_consistent();row=core.info(final=True)
    row.update(map_id=rec.map_id,map_sha256=rec.sha256,source=rec.source,split=rec.split,
               pattern=rec.pattern,tier=rec.tier,beta=job['beta'],deterministic=True,
               initial_pair_id=ic['pair_id'],start_r=ic['start'][0],start_c=ic['start'][1],
               start_heading=ic['heading'],masked_entropy_mean=mean(ent),
               multi_action_normalized_entropy_mean=mean(norm))
    validate_row(row,job);return safe(row)


def find_registered(plan,key,path):
    job=plan['jobs'][key];wanted=Path(path).resolve()
    for cp in job['history']:
        if Path(cp['path']).resolve()==wanted:
            require(sha(wanted)==cp['sha256'],'Historical checkpoint changed');return cp
    run,regs=registered_new(job)
    for r in regs:
        if r.get('source_baseline'):continue
        # Match path before hashing; avoid re-hashing every preceding checkpoint.
        name=Path(str(r['path']).replace('\\','/')).name
        local=run/('last_model.zip' if r.get('final') else 'checkpoints/'+name)
        if local.resolve()==wanted:
            p=checkpoint_path(run,Path(job['case_dir']),r)
            return dict(r,path=str(p),nominal_m=int(r['num_timesteps'])//INTERVAL,origin='continuation')
    raise RuntimeError('Unregistered checkpoint refused: '+str(path))


def eval_worker(plan,key,path):
    job=plan['jobs'][key];cp=find_registered(plan,key,path)
    base=Path(plan['out'])/'validation'/key/f"step_{cp['num_timesteps']}"
    with lock(base/'.eval.lock'):
        if validation_done(plan,key,cp):return
        verify_job(job);rt=runtime(job)
        import numpy as np
        from sb3_contrib import MaskablePPO
        params=dict(version=VERSION,beta=job['beta'],checkpoint_sha256=cp['sha256'],
                    checkpoint_step=int(cp['num_timesteps']),environment=job['spec']['environment'],
                    runtime_hashes={n:job['runtime_hashes'][n] for n in REQUIRED_RUNTIME},
                    initial_seed=20260907,seed=42,deterministic=True,device='cpu',
                    validation=[(r.map_id,r.sha256) for r in rt.val])
        ident=digest(params);freeze(base/'identity.json',dict(sha256=ident,parameters=params))
        model=None;rr=[]
        try:
            for i,rec in enumerate(rt.val,1):
                array=np.load(rec.path,allow_pickle=False)
                require(rt.utils.array_hash(array)==rec.sha256,'Validation map hash changed: '+rec.map_id)
                ic=initial_condition(rt,rec,array)
                dest=base/'maps'/(hashlib.sha256(rec.map_id.encode()).hexdigest()+'.json')
                if dest.exists():
                    cache=read(dest);row=cache['row']
                    require(cache['identity']==ident and cache['row_sha256']==digest(row),
                            'Invalid/changed validation cache: '+str(dest))
                    require(row['map_id']==rec.map_id and row['map_sha256']==rec.sha256 and
                            row['initial_pair_id']==ic['pair_id'],'Validation cache map/initial mismatch')
                    validate_row(row,job)
                else:
                    if model is None:
                        require(sha(cp['path'])==cp['sha256'],'Checkpoint changed before inference')
                        model=MaskablePPO.load(cp['path'],device='cpu')
                        require(model.num_timesteps==int(cp['num_timesteps']),'Loaded checkpoint step mismatch')
                        rt.core.audit_model(model);rt.features.audit_cnn3_model(model)
                        model.policy.set_training_mode(False)
                    row=evaluate_one(rt,job,model,rec,array,ic)
                    row.update(model=key,nominal_m=cp['nominal_m'],checkpoint_step=int(cp['num_timesteps']),
                               checkpoint_sha256=cp['sha256'])
                    write(dest,dict(identity=ident,row_sha256=digest(row),row=row))
                rr.append(row)
                if i==1 or i%25==0 or i==180:
                    csv_write(base/'per_map.csv',rr)
                    print(f'[VAL] {key} {cp["nominal_m"]}M {i}/180; success={sum(truth(x["is_success"]) for x in rr)}',flush=True)
        finally:
            del model;gc.collect()
        summary=aggregate(rr);summary.update(model=key,beta=job['beta'],nominal_m=cp['nominal_m'],
                                             checkpoint_step=int(cp['num_timesteps']))
        write(base/'summary.json',summary)
        write(base/'complete.json',dict(checkpoint_sha256=cp['sha256'],identity=ident,n_maps=180,
              files={n:sha(base/n) for n in ('summary.json','per_map.csv')}))
    build_report(plan)


def aggregate(rr):
    good=[r for r in rr if truth(r['is_success'])]
    out=dict(n_maps=len(rr),success_n=len(good),success_pct=100*len(good)/len(rr) if rr else None,
             coverage_pct=mean(r.get('coverage_pct') for r in rr))
    for k in METRICS:out[k]=mean(r.get(k) for r in good)
    tv=out[T];lw=out[LW]
    out['Lw_over_T_pct']=100*lw/tv if tv and lw is not None else None
    out['mean_mapwise_Lw_over_T_pct']=mean(100*float(r[LW])/float(r[T]) for r in good
                                          if number(r.get(LW)) is not None and number(r.get(T)) not in (None,0))
    return out


def build_report(plan,plots=True):
    out=Path(plan['out']);report=out/'report';report.mkdir(parents=True,exist_ok=True)
    with lock(out/'.report.lock',blocking=True):
        all_rr={};summaries=[]
        for key,beta in CASES.items():
            for complete in sorted((out/'validation'/key).glob('step_*/complete.json')):
                d=read(complete);base=complete.parent
                for n,h in d['files'].items():require(sha(base/n)==h,'Report input changed: '+str(base/n))
                s=read(base/'summary.json');rr=rows(base/'per_map.csv')
                require(len(rr)==180==d['n_maps'],'Partial validation presented as complete')
                all_rr[(key,int(s['nominal_m']))]=rr;summaries.append(s)
        summaries.sort(key=lambda x:(float(x['beta']),int(x['nominal_m'])))
        csv_write(report/'checkpoint_summary.csv',summaries)
        fixed_rows=[];global_rows=[];group_rows=[];available={}
        fixed_sets={}
        for key,beta in CASES.items():
            own={m:rr for (k,m),rr in all_rr.items() if k==key};available[key]=sorted(own)
            sets=[{r['map_id'] for r in rr if truth(r['is_success'])} for rr in own.values()]
            fixed=set.intersection(*sets) if sets else set();fixed_sets[key]=fixed
            for m,rr in sorted(own.items()):
                sr=aggregate([r for r in rr if r['map_id'] in fixed])
                sr.update(model=key,beta=beta,nominal_m=m,fixed_n=len(fixed));fixed_rows.append(sr)
                groups=defaultdict(list)
                for r in rr:groups[(r['pattern'],str(r['tier']))].append(r)
                for (pattern,tier),gg in groups.items():
                    g=aggregate(gg);g.update(model=key,beta=beta,nominal_m=m,pattern=pattern,tier=tier);group_rows.append(g)
        glob=set.intersection(*(fixed_sets[k] for k in CASES)) if all(available.values()) else set()
        if all(available.values()):
            for (key,m),rr in sorted(all_rr.items()):
                sr=aggregate([r for r in rr if r['map_id'] in glob]);sr.update(model=key,beta=CASES[key],nominal_m=m,fixed_n=len(glob))
                global_rows.append(sr)
        csv_write(report/'fixed_success_by_beta.csv',fixed_rows)
        csv_write(report/'global_common_success.csv',global_rows)
        csv_write(report/'pattern_tier_summary.csv',group_rows)
        diags=[]
        for key,job in plan['jobs'].items():
            run,_=registered_new(job)
            for folder,origin in [(Path(job['source_run']),'source'),(run,'continuation')]:
                p=folder/'training_updates.csv'
                if not p.is_file():continue
                for r in rows(p):
                    step=number(r.get('timesteps'))
                    if step is None:continue
                    d={k:number(r.get(k)) for k in DIAGNOSTICS}
                    d.update(model=key,beta=job['beta'],step=int(step),origin=origin);diags.append(d)
        csv_write(report/'training_diagnostics.csv',diags)
        best=[]
        for key in CASES:
            rr=[r for r in summaries if r['model']==key]
            if not rr:continue
            # Never pick a lower-success policy simply because its successful subset is easier.
            max_success=max(int(r['success_n']) for r in rr)
            fixed=[r for r in fixed_rows if r['model']==key and
                   any(s['nominal_m']==r['nominal_m'] and int(s['success_n'])==max_success for s in rr)]
            for field in (T,LW,'Lw_over_T_pct'):
                eligible=[r for r in fixed if number(r.get(field)) is not None]
                if eligible:
                    b=min(eligible,key=lambda r:(float(r[field]),int(r['nominal_m'])))
                    best.append(dict(model=key,metric=field,nominal_m=b['nominal_m'],value=b[field],
                                     success_n=max_success,fixed_n=b['fixed_n'],selection='descriptive_only_not_used_for_training'))
        csv_write(report/'best_observed_validation.csv',best)
        expected=set(range(1,51));missing={k:sorted(expected-set(v)) for k,v in available.items()}
        full=all(not v for v in missing.values())
        write(report/'report_status.json',dict(complete_1M_to_50M=full,available_milestones=available,
              missing_milestones=missing,fixed_n={k:len(v) for k,v in fixed_sets.items()},global_common_n=len(glob)))
        imgs=[];plot_warning=None
        if plots and summaries:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                chart_specs=[('success_pct','Success (%)',summaries),('coverage_pct','Coverage (%)',summaries),
                             (T,'Mission time T (s)',fixed_rows),(LW,'Weighted latency Lw (s)',fixed_rows),
                             ('Lw_over_T_pct','Ratio of means Lw/T (%)',fixed_rows),
                             ('masked_entropy_mean','Masked entropy',fixed_rows)]
                for field,label,data in chart_specs:
                    fig,ax=plt.subplots(figsize=(8,4.5))
                    for key,beta in CASES.items():
                        rr=sorted([r for r in data if r['model']==key],key=lambda r:r['nominal_m'])
                        if rr:ax.plot([r['nominal_m'] for r in rr],
                                      [r[field] if r.get(field) is not None else float('nan') for r in rr],
                                      marker='o',markersize=2.5,label=f'Strict beta={beta:g}')
                    ax.set(xlabel='Training budget (million interactions)',ylabel=label);ax.grid(alpha=.25);ax.legend()
                    fig.tight_layout();name=field+'.png';fig.savefig(report/name,dpi=140);plt.close(fig);imgs.append(name)
            except ImportError as e:plot_warning=str(e)
        def table(rr,fields):
            h='<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'
            for r in rr:
                cells=[]
                for k in fields:
                    v=r.get(k);n=number(v)
                    val='—' if v is None else f'{n:,.3f}' if n is not None else str(v)
                    cells.append('<td>'+html.escape(val)+'</td>')
                h+='<tr>'+''.join(cells)+'</tr>'
            return h+'</tbody></table>'
        latest=[]
        for key in CASES:
            own=[r for r in summaries if r['model']==key]
            if own:latest.append(max(own,key=lambda r:r['nominal_m']))
        text='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Strict COST11 50M validation</title>
<style>body{font:15px system-ui,sans-serif;line-height:1.7;max-width:1200px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{background:#f4f4f4}img{max-width:100%}.scroll{overflow:auto}pre{white-space:pre-wrap}</style>
<h1>Strict COST11: β=0 / 0.5 / 1 · 총 50M</h1>'''
        text+='<p><b>'+('1M–50M 전체 평가 완료' if full else '중간 보고서 — 아직 완료되지 않은 checkpoint가 있습니다.')+'</b></p>'
        text+='<p>기존 policy와 Adam에서 총 50M까지 추가학습. LR=5e-5. 1M 경계를 지난 첫 완료 PPO update를 저장합니다. 새 checkpoint마다 별도 CPU 프로세스에서 동일한 synthetic validation 180개를 평가합니다. Test/OSM 및 경로개선은 포함하지 않습니다.</p>'
        text+='<p>첫 source→50M 연장 경계에서는 새 episode를 시작하고 원본 균형 map sampler의 누적 위치를 재구성합니다. 이후 중단 재개는 native optimizer-boundary 전체 복구 상태를 사용합니다.</p>'
        text+='<h2>마지막으로 평가된 checkpoint</h2><div class="scroll">'+table(latest,['model','nominal_m','checkpoint_step','success_n','n_maps',T,LW,'Lw_over_T_pct'])+'</div>'
        text+='<p>위 시간 평균은 해당 시점 성공 맵 기준입니다. 아래 시간 곡선은 각 β의 모든 <em>평가 완료된</em> checkpoint 공통 성공 맵에 고정합니다. 평가가 추가되면 고정집합도 다시 계산합니다. β 간 직접 비교는 global_common_success.csv를 사용합니다.</p>'
        text+='<pre>'+html.escape(json.dumps(dict(fixed_n={k:len(v) for k,v in fixed_sets.items()},global_n=len(glob),missing=missing),ensure_ascii=False,indent=2))+'</pre>'
        text+=''.join('<img src="'+html.escape(n)+'" alt="'+html.escape(n)+'">' for n in imgs)
        if plot_warning:text+='<p>PNG 미생성: '+html.escape(plot_warning)+'. HTML/CSV는 생성되었습니다.</p>'
        text+='<h2>Checkpoint별 결과</h2><div class="scroll">'+table(summaries,['model','nominal_m','success_pct',T,LW,'Lw_over_T_pct',*METRICS[2:6]])+'</div>'
        text+='<h2>원자료 및 집계</h2><p>checkpoint_summary.csv · fixed_success_by_beta.csv · global_common_success.csv · pattern_tier_summary.csv · training_diagnostics.csv · best_observed_validation.csv</p></html>'
        write_text(report/'report.html',text)
        # Bundle all per-map numerical rows, not just aggregate means; models are excluded.
        zp=report/'UPLOAD_strict_50M_results.zip';tmp=zp.with_name(zp.name+'.writing')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in report.iterdir():
                if p.is_file() and p.suffix!='.zip' and '.writing' not in p.name:z.write(p,p.name)
            for key,m in all_rr:
                step=next(s['checkpoint_step'] for s in summaries if s['model']==key and int(s['nominal_m'])==m)
                d=out/'validation'/key/f'step_{step}'
                for n in ('per_map.csv','identity.json','complete.json'):
                    z.write(d/n,f'per_map/{key}/{m:02d}M/{n}')
        os.replace(tmp,zp)
        print('[REPORT]',report/'report.html',flush=True)


def launch(plan,mode):
    logs=Path(plan['out'])/'logs';logs.mkdir(exist_ok=True);jobs=[]
    try:
        for key,job in plan['jobs'].items():
            env=os.environ.copy();env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            env['CUDA_VISIBLE_DEVICES']=str(job['gpu']) if mode in ('train','check') else ''
            env['MPLBACKEND']='Agg'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--_worker',mode,
                 '--_plan',str(Path(plan['out'])/'plan.json'),'--key',key]
            logpath=logs/f'{key}_{mode}.log';f=logpath.open('a',encoding='utf-8',buffering=1)
            p=subprocess.Popen(cmd,cwd=plan['root'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((key,p,f,logpath))
            print(f'[LAUNCH {mode}] {key} GPU={job["gpu"] if mode in ("train","check") else "CPU"}; log={logpath}',flush=True)
        failed=[]
        for key,p,f,lp in jobs:
            rc=p.wait();f.close();print(f'[DONE {mode}] {key} exit={rc}',flush=True)
            if rc:failed.append((key,rc,str(lp)))
        require(not failed,'Worker failure(s); recovery/cache retained: '+str(failed))
    except KeyboardInterrupt:
        for _,p,_,_ in jobs:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for _,p,f,_ in jobs:
            try:p.wait()
            finally:f.close()
        raise
    finally:
        for _,p,f,_ in jobs:
            if p.poll() is not None and not f.closed:f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out', '--out-root',dest='out',type=Path)
    p.add_argument('--gpus',nargs=3,type=int,default=[1,2,3],metavar=('BETA0','BETA05','BETA1'))
    p.add_argument('--phase',choices=['all','check','train','validate','report'],default='all')
    p.add_argument('--_worker',choices=['check','train','eval','validate'],help=argparse.SUPPRESS)
    p.add_argument('--_plan',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--key',choices=list(CASES),help=argparse.SUPPRESS)
    p.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args._worker:
        require(args._plan is not None and args.key is not None,'Internal worker arguments missing')
        plan=read(args._plan);job=plan['jobs'][args.key]
        require(plan['launcher_sha256']==sha(__file__),'Launcher changed after plan was frozen')
        if args._worker=='eval':
            require(args.checkpoint is not None,'Checkpoint required');eval_worker(plan,args.key,args.checkpoint)
        elif args._worker=='validate':
            prepare(job);evaluate_pending(plan,args.key,history=True)
        else:
            with lock(Path(job['case_dir'])/'.worker.lock'):
                train_worker(plan,args.key,check=args._worker=='check')
        return 0
    args.root=args.root.expanduser().resolve()
    require(args.root.is_dir(),'Suite root not found: '+str(args.root))
    args.out=(args.out.expanduser().resolve() if args.out else args.root/DEFAULT_OUT)
    require(len(set(args.gpus))==3 and min(args.gpus)>=0,'Three distinct physical GPU IDs required')
    with lock(args.out/'.launcher.lock'):
        plan=make_plan(args)
        for key,job in plan['jobs'].items():
            print(f'[SOURCE] {key}: beta={job["beta"]:g} GPU={job["gpu"]} step={job["source_step"]:,} -> TOTAL {TARGET:,}',flush=True)
            print('         '+job['source_model'],flush=True)
        if args.phase in ('all','check'):
            launch(plan,'check')
            if args.phase=='check':return 0
        if args.phase in ('all','train'):launch(plan,'train')
        if args.phase=='validate':launch(plan,'validate')
        if args.phase in ('all','train','validate','report'):build_report(plan)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[PAUSED] Re-run the same command; committed recovery and validation caches are retained.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('[ERROR]',exc,file=sys.stderr)
        raise SystemExit(1)

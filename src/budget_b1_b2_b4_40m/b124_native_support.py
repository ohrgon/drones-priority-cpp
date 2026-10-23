"""Shared native-run utilities, adapted from uploaded source/run_strict_50m.py.
No PPO, reward, CNN or environment implementation is replaced here.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, os
from pathlib import Path
from contextlib import contextmanager
LR = 5e-5
TRAINER = 'trainers_ch4_c3nt_cost_sweep'
CORE = 'c3nt_cost_sweep_generated_core'
FEATURES = 'nrg1_c3nt_cost5_features'
LOGGER = 'c3nt_cost_sweep_generated_logging'
TRAINER_SHA = '0049fb0f18f07cd80f2ec96a6ad16510843d01dcd1703ec7250435c80908585e'
T = 'mission_time_sec'
LW = 'priority_weighted_first_visit_latency_sec'


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

"""Native Torch DDQN, discrete SAC, SAC/TD3 with declared continuous-to-grid mapping.

These are new common-framework baselines, not reproductions of paper-adapted CPP.
Formulas: van Hasselt 2015; Christodoulou 2019; Haarnoja 2018; Fujimoto 2018.
Off-policy hyperparameters and action mappings are declared implementation choices.
No native agent needs Gym to be tested. PPO uses official sb3_contrib in trainers.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def tensor_obs(obs, device):
    m = torch.as_tensor(obs["map"], device=device)
    single=m.ndim==3
    if single:m=m.unsqueeze(0)
    if 'scalars' in obs:
        h=torch.as_tensor(obs['scalars'],device=device,dtype=torch.float32)
        if single:h=h.unsqueeze(0)
    else:
        heading=torch.as_tensor(obs['heading'],device=device,dtype=torch.long).reshape(-1)
        h=F.one_hot(heading,4).float()
    return m.float()/255.,h

class Encoder(nn.Module):
    def __init__(self,grid_size=50,channels=6,extra_dim=4,features=256):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(channels,32,8,4),nn.ReLU(),
                               nn.Conv2d(32,64,4,2),nn.ReLU(),
                               nn.Conv2d(64,64,3,1),nn.ReLU(),nn.Flatten())
        try:
            with torch.no_grad(): n=self.cnn(torch.zeros(1,channels,grid_size,grid_size)).shape[1]
        except RuntimeError as exc:
            raise ValueError("Nature-style CNN needs grid >=36; exact small maps do not train this network") from exc
        self.fc=nn.Sequential(nn.Linear(n,features),nn.ReLU())
        self.output_dim=features+extra_dim
    def forward(self,obs):
        m,h=obs
        return torch.cat((self.fc(self.cnn(m)),h),dim=-1)

class StateNet(nn.Module):
    def __init__(self,size=50,out=4,hidden=(256,256),features=256,channels=6,extra_dim=4):
        super().__init__(); self.encoder=Encoder(size,channels=channels,extra_dim=extra_dim,features=features)
        seq=[]; n=self.encoder.output_dim
        for k in hidden: seq.extend((nn.Linear(n,k),nn.ReLU())); n=k
        seq.append(nn.Linear(n,out)); self.head=nn.Sequential(*seq)
    def forward(self,obs): return self.head(self.encoder(obs))

class ContinuousCritic(nn.Module):
    def __init__(self,size,action_dim,hidden=(256,256),features=256,channels=6,extra_dim=4):
        super().__init__(); self.encoder=Encoder(size,channels=channels,extra_dim=extra_dim,features=features)
        n=self.encoder.output_dim+action_dim; seq=[]
        for k in hidden: seq.extend((nn.Linear(n,k),nn.ReLU())); n=k
        seq.append(nn.Linear(n,1)); self.head=nn.Sequential(*seq)
    def forward(self,obs,a): return self.head(torch.cat((self.encoder(obs),a),dim=-1)).squeeze(-1)

def masked_distribution(logits,mask):
    if not bool(mask.any(dim=-1).all()): raise ValueError("Empty action set")
    logits=logits.masked_fill(~mask,-1e9)
    logp=F.log_softmax(logits,dim=-1)
    p=logp.exp()*mask
    return p,logp

def continuous_to_discrete(values,mask,mapping,heading):
    """preference4: argmax admissible preference; heading_angle: closest valid heading.
    heading_angle u in [-1,1] specifies relative angle pi*u, not a stay action.
    Same values/order/rule are saved for training and evaluation.
    """
    mask=np.asarray(mask,dtype=bool); v=np.asarray(values,dtype=float).reshape(-1)
    if not mask.any() or not np.isfinite(v).all(): raise ValueError("Invalid mapping input")
    if mapping=="preference4":
        if len(v)!=4: raise ValueError("preference4 requires 4 values")
        scores=v
    elif mapping=="heading_angle":
        if len(v)!=1: raise ValueError("heading_angle requires 1 value")
        target=heading*math.pi/2+float(np.clip(v[0],-1,1))*math.pi
        scores=np.cos(np.arange(4)*math.pi/2-target)
    else: raise ValueError(f"Unknown continuous mapping: {mapping}")
    return int(np.argmax(np.where(mask,scores,-np.inf)))

class Replay:
    """Bounded uint8 buffer; explicitly stores CURRENT and NEXT state masks."""
    def __init__(self,capacity,size,action_dim=0,channels=6,extra_dim=None):
        self.capacity=int(capacity); self.n=self.pos=0
        if self.capacity<1: raise ValueError("Replay capacity must be positive")
        self.maps=np.empty((self.capacity,channels,size,size),dtype=np.uint8)
        self.extra_dim=extra_dim
        self.extras=np.empty((self.capacity,extra_dim),np.float32) if extra_dim is not None else None
        self.next_extras=np.empty_like(self.extras) if extra_dim is not None else None
        self.next_maps=np.empty_like(self.maps)
        self.headings=np.empty(self.capacity,dtype=np.int64)
        self.next_headings=np.empty_like(self.headings)
        self.masks=np.empty((self.capacity,4),dtype=bool); self.next_masks=np.empty_like(self.masks)
        self.actions=np.empty((self.capacity,action_dim),np.float32) if action_dim else np.empty(self.capacity,np.int64)
        self.rewards=np.empty(self.capacity,np.float32); self.dones=np.empty(self.capacity,bool)
    @property
    def estimated_bytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))
    def add(self,obs,mask,action,reward,next_obs,next_mask,done):
        i=self.pos
        self.maps[i]=obs['map']; self.headings[i]=obs.get('heading',0); self.masks[i]=mask
        if self.extra_dim is not None:self.extras[i]=obs['scalars'];self.next_extras[i]=next_obs['scalars']
        self.actions[i]=action; self.rewards[i]=reward; self.dones[i]=done
        self.next_maps[i]=next_obs['map']; self.next_headings[i]=next_obs.get('heading',0); self.next_masks[i]=next_mask
        self.pos=(i+1)%self.capacity; self.n=min(self.n+1,self.capacity)
    def save_recovery(self, directory):
        """Store only valid entries; no uninitialized np.empty bytes are persisted."""
        from full_v2_utils import write_json
        from resume_ch4 import sync_file
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for name, values in vars(self).items():
            if isinstance(values, np.ndarray):
                path = directory / (name + '.npy')
                np.save(path, values[:self.n], allow_pickle=False)
                sync_file(path)
                arrays[name] = dict(shape=list(values.shape), dtype=values.dtype.str)
        write_json(directory/'replay.json', dict(capacity=self.capacity, n=self.n, pos=self.pos,
                   extra_dim=self.extra_dim, arrays=arrays))

    def load_recovery(self, directory):
        from full_v2_utils import read_json
        directory = Path(directory); state = read_json(directory/'replay.json')
        if state['capacity'] != self.capacity or state['extra_dim'] != self.extra_dim:
            raise ValueError('Replay configuration changed on resume')
        n, pos = int(state['n']), int(state['pos'])
        if not 0 <= n <= self.capacity or not 0 <= pos < self.capacity or (n < self.capacity and pos != n):
            raise ValueError('Invalid replay length or circular-buffer position')
        expected = {k for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        if expected != set(state['arrays']):
            raise ValueError('Recovery replay fields differ')
        for name, metadata in state['arrays'].items():
            target = getattr(self, name)
            if list(target.shape) != metadata['shape'] or target.dtype.str != metadata['dtype']:
                raise ValueError('Replay array schema changed: '+name)
            source = np.load(directory/(name+'.npy'), mmap_mode='r', allow_pickle=False)
            if source.shape != (n, *target.shape[1:]) or source.dtype != target.dtype:
                raise ValueError('Replay array is incomplete: '+name)
            target[:n] = source
        self.n, self.pos = n, pos

    def sample(self,batch,rng,device):
        if not self.n: raise ValueError("Empty replay")
        i=rng.integers(0,self.n,size=batch)
        obs=tensor_obs({'map':self.maps[i],**({'heading':self.headings[i]} if self.extra_dim is None else {'scalars':self.extras[i]})},device)
        nxt=tensor_obs({'map':self.next_maps[i],**({'heading':self.next_headings[i]} if self.extra_dim is None else {'scalars':self.next_extras[i]})},device)
        return obs,torch.as_tensor(self.masks[i],device=device),torch.as_tensor(self.actions[i],device=device),torch.as_tensor(self.rewards[i],device=device),nxt,torch.as_tensor(self.next_masks[i],device=device),torch.as_tensor(self.dones[i],device=device).float()

class NativeAgent:
    """One saved format with method and action mapping required on reload."""
    def __init__(self,method,config,seed=42,device="auto"):
        if method not in {'ddqn','sac_discrete','sac_continuous','td3'}: raise ValueError(method)
        self.method=method; self.cfg=deepcopy(config); self.seed=int(seed)
        self.device=torch.device('cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device)
        torch.manual_seed(seed); self.rng=np.random.default_rng(seed)
        size=config.get('grid_size',50); hidden=tuple(config.get('hidden',[256,256])); features=config.get('features',256)
        self.gamma=config.get('gamma',.998); self.tau=config.get('tau',.005)
        self.mapping=config.get('mapping','preference4')
        self.adim=4 if self.mapping=='preference4' else 1
        self.updates=0; self.num_timesteps=0
        channels=config.get('channels',6);extra_dim=config.get('extra_dim',4)
        lr=config.get('learning_rate',.0003)
        self.alpha=float(config.get('alpha',.2))
        self.auto_alpha=bool(config.get('auto_alpha',False))
        self.log_alpha=torch.tensor(math.log(self.alpha),device=self.device,requires_grad=self.auto_alpha)
        if self.auto_alpha: self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=lr)
        self.nets={}; self.opts={}
        def state(key,out): self.nets[key]=StateNet(size,out,hidden,features,channels,extra_dim).to(self.device)
        if method=='ddqn':
            state('q',4); self.nets['qt']=deepcopy(self.nets['q'])
            self.opts['q']=torch.optim.Adam(self.nets['q'].parameters(),lr=lr)
        elif method=='sac_discrete':
            for key in ('actor','q1','q2'): state(key,4)
            for key in ('q1','q2'): self.nets[key+'t']=deepcopy(self.nets[key])
            self.opts['actor']=torch.optim.Adam(self.nets['actor'].parameters(),lr=lr)
            self.opts['q']=torch.optim.Adam(list(self.nets['q1'].parameters())+list(self.nets['q2'].parameters()),lr=lr)
        else:
            state('actor',self.adim*(2 if method=='sac_continuous' else 1))
            for key in ('q1','q2'):
                self.nets[key]=ContinuousCritic(size,self.adim,hidden,features,channels,extra_dim).to(self.device)
                self.nets[key+'t']=deepcopy(self.nets[key])
            if method=='td3': self.nets['actort']=deepcopy(self.nets['actor'])
            self.opts['actor']=torch.optim.Adam(self.nets['actor'].parameters(),lr=lr)
            self.opts['q']=torch.optim.Adam(list(self.nets['q1'].parameters())+list(self.nets['q2'].parameters()),lr=lr)
        for key,net in self.nets.items():
            if key.endswith('t'):
                for p in net.parameters(): p.requires_grad_(False)

    def _canonical(self,a,mask):
        if self.mapping=='preference4': return torch.where(mask,a,torch.full_like(a,-1.0))
        return a

    def _continuous_policy(self,obs,mask,deterministic=False):
        raw=self.nets['actor'](obs)
        if self.method=='td3': return self._canonical(raw.tanh(),mask),None
        mean,logstd=raw.chunk(2,dim=-1); logstd=logstd.clamp(-20,2)
        dist=torch.distributions.Normal(mean,logstd.exp())
        z=mean if deterministic else dist.rsample(); a=z.tanh()
        # Preference coordinates for inadmissible actions never affect the transition.
        # Do not encourage their entropy; critic receives the same canonical vector.
        logp=dist.log_prob(z)-torch.log(1-a.square()+1e-6)
        if self.mapping=='preference4': logp=(logp*mask).sum(-1)
        else: logp=logp.sum(-1)
        return self._canonical(a,mask),logp

    def act(self,obs,mask,*,deterministic=True,epsilon=0.0,noise=0.0):
        m=np.asarray(mask,dtype=bool)
        if not m.any(): raise ValueError("Empty mask")
        with torch.no_grad():
            x=tensor_obs(obs,self.device); mt=torch.as_tensor(m[None],device=self.device)
            if self.method=='ddqn':
                q=self.nets['q'](x).masked_fill(~mt,-1e9)[0].cpu().numpy()
                best=int(np.argmax(q)); allowed=np.flatnonzero(m)
                p=np.zeros(4); p[allowed]=epsilon/len(allowed); p[best]+=1-epsilon
                a=best if deterministic else int(self.rng.choice(4,p=p))
                return a,a,p
            if self.method=='sac_discrete':
                p,_=masked_distribution(self.nets['actor'](x),mt); pn=p[0].cpu().numpy().astype(float); pn/=pn.sum()
                a=int(np.argmax(pn)) if deterministic else int(self.rng.choice(4,p=pn))
                return a,a,pn
            a,_=self._continuous_policy(x,mt,deterministic=deterministic)
            a=a[0].cpu().numpy()
            if self.method=='td3' and not deterministic and noise:
                a=np.clip(a+self.rng.normal(0,noise,size=a.shape),-1,1)
            if self.mapping=='preference4': a=np.where(m,a,-1.).astype(np.float32)
            grid=continuous_to_discrete(a,m,self.mapping,int(obs.get('heading',np.argmax(obs['scalars'][-4:]) if 'scalars' in obs and len(obs['scalars'])>=4 else 0)))
            # Continuous density is NOT a categorical policy entropy. Report NA.
            return grid,a.astype(np.float32),None

    def update(self,batch):
        o,m,a,r,n,nm,d=batch; g=self.gamma; alpha=self.log_alpha.exp().detach()
        self.updates+=1
        if self.method=='ddqn':
            with torch.no_grad():
                sel=self.nets['q'](n).masked_fill(~nm,-1e9).argmax(1)
                target=r+g*(1-d)*self.nets['qt'](n).gather(1,sel[:,None]).squeeze(1)
            q=self.nets['q'](o).gather(1,a.long()[:,None]).squeeze(1)
            qloss=F.smooth_l1_loss(q,target)
            self._opt('q',qloss)
            if self.updates % self.cfg.get('target_update_updates',1000)==0: self.nets['qt'].load_state_dict(self.nets['q'].state_dict())
            return {'q_loss':float(qloss.detach()),'actor_loss':None,'alpha':None}
        if self.method=='sac_discrete':
            with torch.no_grad():
                pn,ln=masked_distribution(self.nets['actor'](n),nm)
                v=(pn*(torch.minimum(self.nets['q1t'](n),self.nets['q2t'](n))-alpha*ln)).sum(-1)
                target=r+g*(1-d)*v
            q1=self.nets['q1'](o).gather(1,a.long()[:,None]).squeeze(1)
            q2=self.nets['q2'](o).gather(1,a.long()[:,None]).squeeze(1)
            qloss=F.mse_loss(q1,target)+F.mse_loss(q2,target); self._opt('q',qloss)
            p,lp=masked_distribution(self.nets['actor'](o),m)
            with torch.no_grad(): q=torch.minimum(self.nets['q1'](o),self.nets['q2'](o))
            aloss=(p*(alpha*lp-q)).sum(-1).mean(); self._opt('actor',aloss)
            if self.auto_alpha:
                h=-(p*lp).sum(-1).detach()
                target_h=self.cfg.get('entropy_fraction',.98)*m.sum(-1).float().log()
                # H < target => log_alpha increases under gradient descent.
                loss=self.log_alpha*(h-target_h).mean()
                self.alpha_opt.zero_grad(); loss.backward(); self.alpha_opt.step()
            self._soft('q1','q1t'); self._soft('q2','q2t')
            return {'q_loss':float(qloss.detach()),'actor_loss':float(aloss.detach()),'alpha':float(self.log_alpha.exp().detach())}
        with torch.no_grad():
            if self.method=='td3':
                an=self.nets['actort'](n).tanh()
                noise=(torch.randn_like(an)*self.cfg.get('target_policy_noise',.2)).clamp(-self.cfg.get('target_noise_clip',.5),self.cfg.get('target_noise_clip',.5))
                an=self._canonical((an+noise).clamp(-1,1),nm)
                logpn=0.
            else: an,logpn=self._continuous_policy(n,nm)
            qn=torch.minimum(self.nets['q1t'](n,an),self.nets['q2t'](n,an))
            target=r+g*(1-d)*(qn-(alpha*logpn if self.method=='sac_continuous' else 0))
        qloss=F.mse_loss(self.nets['q1'](o,a.float()),target)+F.mse_loss(self.nets['q2'](o,a.float()),target)
        self._opt('q',qloss); aloss=None
        if self.method=='sac_continuous' or self.updates % self.cfg.get('policy_delay',2)==0:
            for key in ('q1','q2'):
                for p in self.nets[key].parameters(): p.requires_grad_(False)
            ap,logp=self._continuous_policy(o,m)
            aloss=(-self.nets['q1'](o,ap)).mean() if self.method=='td3' else (alpha*logp-torch.minimum(self.nets['q1'](o,ap),self.nets['q2'](o,ap))).mean()
            self._opt('actor',aloss)
            for key in ('q1','q2'):
                for p in self.nets[key].parameters(): p.requires_grad_(True)
            if self.method=='td3': self._soft('actor','actort')
            elif self.auto_alpha:
                target=-float(self.adim)
                loss=-(self.log_alpha*(logp.detach()+target)).mean()
                self.alpha_opt.zero_grad(); loss.backward(); self.alpha_opt.step()
            self._soft('q1','q1t'); self._soft('q2','q2t')
        return {'q_loss':float(qloss.detach()),'actor_loss':None if aloss is None else float(aloss.detach()),'alpha':float(alpha) if self.method=='sac_continuous' else None}

    def _opt(self,key,loss):
        opt=self.opts[key]; opt.zero_grad(set_to_none=True); loss.backward()
        params=[p for group in opt.param_groups for p in group['params']]
        nn.utils.clip_grad_norm_(params,self.cfg.get('max_grad_norm',.5)); opt.step()
    def _soft(self,key,target):
        with torch.no_grad():
            for p,t in zip(self.nets[key].parameters(),self.nets[target].parameters()): t.lerp_(p,self.tau)
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(format='chapter4_native_v1',method=self.method,config=self.cfg,seed=self.seed,
                     networks={k:v.state_dict() for k,v in self.nets.items()},optimizers={k:v.state_dict() for k,v in self.opts.items()},
                     updates=self.updates,num_timesteps=self.num_timesteps,log_alpha=self.log_alpha.detach().cpu(),rng=self.rng.bit_generator.state,
                     alpha_optimizer=self.alpha_opt.state_dict() if self.auto_alpha else None)
        temp=path.with_name(path.name+'.tmp'); torch.save(payload,temp); temp.replace(path)
    @classmethod
    def load(cls,path,device='auto'):
        # Only load checkpoints generated by this trusted package.
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p.get('format')!='chapter4_native_v1': raise ValueError("Not a Chapter4 native checkpoint")
        agent=cls(p['method'],p['config'],p['seed'],device)
        for k,v in p['networks'].items(): agent.nets[k].load_state_dict(v)
        for k,v in p['optimizers'].items(): agent.opts[k].load_state_dict(v)
        with torch.no_grad(): agent.log_alpha.copy_(p['log_alpha'].to(agent.device))
        if agent.auto_alpha: agent.alpha_opt.load_state_dict(p['alpha_optimizer'])
        agent.updates=p['updates']; agent.num_timesteps=p['num_timesteps']; agent.rng.bit_generator.state=p['rng']
        return agent
    def architecture(self):
        return {'method':self.method,'mapping':self.mapping if self.method in {'td3','sac_continuous'} else 'native_discrete',
                'networks':{k:str(v) for k,v in self.nets.items()},
                'trainable_parameters':sum(p.numel() for net in self.nets.values() for p in net.parameters() if p.requires_grad),
                'optimizer':{k:str(v) for k,v in self.opts.items()},'torch_version':torch.__version__}

"""Supplementary paper-adapted controls + instrumentation of the old mask.

Reward tables and channel roles follow the uploaded 3_0(1).docx Tables 5--7.
They are adaptations to this grid, NOT reproductions of the original papers.
Unspecified local/coarse resolutions are explicit configuration choices.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import csv
import numpy as np
from core_ch4 import ExperimentCore
from full_v2_utils import write_json,write_csv


def total_variation(mask):
    a=np.asarray(mask,dtype=float)
    return float(np.abs(np.diff(a,axis=0)).sum()+np.abs(np.diff(a,axis=1)).sum())

class PaperAdaptedCore(ExperimentCore):
    def __init__(self,config):
        self.paper_mode=config.paper_mode
        # Legacy control supplies its own reward once; do not add revised FULL reward.
        zero=replace(config,coverage_reward_scale=0.,objective_cost_scale=0.,success_terminal_reward=0.,failure_terminal_penalty=0.,invalid_action_penalty=0.,mask_mode='none')
        super().__init__(zero)
    def reset(self,*args,**kwargs):
        obs,_=super().reset(*args,**kwargs);self.reward_sums['paper_adapted']=0.
        return obs,self.info(final=self.complete_at_reset)
    def observation(self):
        size=self.grid_size;pos=np.zeros((size,size),bool);pos[self.drone_pos]=True
        if self.paper_mode=='ddqn':
            start=np.zeros_like(pos);start[self.start_pos]=True
            channels=[start,self.active_mask,~self.bbox_mask,self.covered_map]
            extra=np.r_[self.drone_pos[0]/max(1,size-1),self.drone_pos[1]/max(1,size-1),
                         max(0,1-self.steps/self.max_steps),int(self._in_bbox(*self.drone_pos)),np.eye(4)[self.heading]].astype(np.float32)
        elif self.paper_mode=='a2c':
            channels=[~self.active_mask,self.uncovered_mask(),self.covered_map,pos];extra=np.empty(0,np.float32)
        elif self.paper_mode=='sac':
            local=[self.covered_map,~self.bbox_mask,self.frontier_mask(),pos]
            coarse=[self.covered_map,~self.bbox_mask,self.frontier_mask(),self.uncovered_mask()]
            radius=self.config.paper_local_radius;factor=self.config.paper_coarse_factor;r,c=self.drone_pos
            channels=[]
            for m in local:
                padded=np.pad(m,((radius,radius),(radius,radius)),constant_values=False)
                crop=padded[r:r+2*radius+1,c:c+2*radius+1]
                idx=np.minimum((np.arange(size)*len(crop)/size).astype(int),len(crop)-1);channels.append(crop[idx[:,None],idx[None,:]])
            for m in coarse:
                coarse_m=m[::factor,::factor];idx=np.minimum((np.arange(size)*len(coarse_m)/size).astype(int),len(coarse_m)-1)
                channels.append(coarse_m[idx[:,None],idx[None,:]])
            extra=np.eye(4,dtype=np.float32)[self.heading]
        else:raise ValueError('Unknown paper adapter')
        return {'map':np.stack(channels).astype(np.uint8)*255,'scalars':extra}
    def step(self,action):
        old_new=self.new_cover_count;old_rev=self.revisit_steps;old_invalid=self.invalid_action_count;tv=total_variation(self.uncovered_mask())
        obs,_,done,truncated,info=super().step(action)
        if self.complete_at_reset:return obs,0.,done,truncated,info
        new=self.new_cover_count>old_new;rev=self.revisit_steps>old_rev;invalid=self.invalid_action_count>old_invalid
        n=self.target_cells_count;turn=bool(info['is_turn']);dt=info['traversal_time_sec']/self.time_straight_sec
        if self.paper_mode=='ddqn':
            reward=float(new)/n-.1*rev/n-.02*dt-.01*turn-1.*invalid
            if done:reward+=1 if info['is_success'] else -1
        elif self.paper_mode=='a2c':
            reward=float(new)-.2*rev-.01-1.*invalid
            if done:reward+=10 if info['is_success'] else -10
        else:
            reward=(float(new)-.08*rev-1.*invalid-.01+.2*(tv-total_variation(self.uncovered_mask())))/n
            if done:reward+=1 if info['is_success'] else -1
        self.total_return+=reward;self.reward_sums['paper_adapted']+=reward
        info=self.info(final=done);info.update(paper_adapted_mode=self.paper_mode,step_reward_paper_adapted=reward)
        return obs,reward,done,truncated,info

class LegacyMaskAudit:
    """Wrap an EXISTING legacy env. Records phase per executed step, not API call.

    A fallback phase is inferred from the original documented direct/progress branch
    conditions. Code containing other undocumented branch conditions needs its own
    instrumentation; this wrapper does not pretend old historical logs contain it.
    """
    def __init__(self,env,log_path):self.env=env;self.log_path=Path(log_path);self.episode=0;self.recorded_decisions=0;self.fallback_count=0;self.pending=None
    def __getattr__(self,name):return getattr(self.env,name)
    def reset(self,*args,**kwargs):self.episode+=1;self.pending=None;return self.env.reset(*args,**kwargs)
    def action_masks(self):
        m=self.env.action_masks();base=self.env
        while hasattr(base,'env'):base=base.env
        r,c=base.drone_pos;u=base.active_mask&~base.covered_map;direct=False;progress=False
        if u.any():
            d=base._get_dist_to_uncovered()
            for a,(dr,dc) in base.moves.items():
                rr,cc=r+dr,c+dc
                if base._in_bbox(rr,cc):direct|=bool(u[rr,cc]);progress|=bool(d[rr,cc]<d[r,c])
            phase='direct' if direct else 'progress' if progress else 'fallback'
        else:phase='terminal'
        self.pending=dict(episode=self.episode,step=int(base.steps),map_name=base.current_map_name,phase=phase,
                          mask_cardinality=int(np.sum(m)),fallback_active=int(phase=='fallback'))
        return m
    def step(self,a):
        if self.pending is None:self.action_masks()
        row=dict(self.pending,action=int(a));out=self.env.step(a);self.recorded_decisions+=1;self.fallback_count+=row['fallback_active'];self.pending=None
        # A dedicated append-only file avoids O(N^2) rewriting for full training.
        new=not self.log_path.exists();self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_path.open('a',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row))
            if new:writer.writeheader()
            writer.writerow(row)
        return out

def audit_fallback_logs(paths,out,expected_steps=None):
    total=active=0;maps=set()
    for path in paths:
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if 'fallback_active' not in (reader.fieldnames or []):raise ValueError('No fallback instrumentation column: absence of a column is NOT zero activations')
            for row in reader:total+=1;active+=int(row['fallback_active']);maps.add(row.get('map_name',''))
    result=dict(recorded_decisions=total,fallback_activations=active,recorded_maps=len(maps),
                full_log_coverage_verified=expected_steps is not None and total==expected_steps,
                expected_decisions=expected_steps,interpretation='zero/nonzero only for recorded decisions; historical unrecorded execution remains unverified')
    write_json(out,result);return result

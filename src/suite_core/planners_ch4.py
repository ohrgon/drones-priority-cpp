"""Chapter 4.3.3 planners and exact small-instance reference.

BCD uses discrete slice connectivity changes to form sweepable cells. STC uses
priority-layer connected-component spanning trees on the original unit-cell graph.
These are explicit implementations of the described families, not claims that the
unprovided legacy BCD/Dijkstra/STC source was reproduced byte for byte.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import heapq
import math
import time
import numpy as np
from coverage_core import MOVES


def shortest_paths(core):
    """Dijkstra on (row,column,arrival heading), with initial heading at source."""
    source=(*core.drone_pos,int(core.heading)); dist={source:0.}; prev={}; q=[(0.,source)]
    while q:
        cost,state=heapq.heappop(q)
        if cost!=dist[state]: continue
        r,c,h=state
        for a,(dr,dc) in enumerate(MOVES):
            nr,nc=r+dr,c+dc
            if not core._in_bbox(nr,nc): continue
            nxt=(nr,nc,a)
            new=cost+(core.time_straight_sec if a==h else core.time_turn_sec)
            if new < dist.get(nxt,math.inf)-1e-12:
                dist[nxt]=new; prev[nxt]=(state,a); heapq.heappush(q,(new,nxt))
    return source,dist,prev


def target_path(search,target):
    source,dist,prev=search
    ends=[(dist[(*target,h)],(*target,h)) for h in range(4) if (*target,h) in dist]
    if not ends: raise ValueError(f"Target is unreachable: {target}")
    cost,state=min(ends)
    actions=[]
    while state!=source:
        state,a=prev[state]; actions.append(a)
    actions.reverse(); return float(cost),actions


def connected_components(mask):
    seen=np.zeros_like(mask,dtype=bool); comps=[]
    H,W=mask.shape
    for p in map(tuple,np.argwhere(mask)):
        if seen[p]: continue
        q=deque([p]); seen[p]=True; comp=[]
        while q:
            r,c=q.popleft(); comp.append((int(r),int(c)))
            for dr,dc in MOVES:
                rr,cc=r+dr,c+dc
                if 0<=rr<H and 0<=cc<W and mask[rr,cc] and not seen[rr,cc]:
                    seen[rr,cc]=True; q.append((rr,cc))
        comps.append(comp)
    return comps


def boustrophedon_cells(mask):
    """Continue a slice only through one-to-one overlap; split/merge starts a cell."""
    H,W=mask.shape; cells=[]; previous=[]
    for c in range(W):
        rows=np.flatnonzero(mask[:,c]); runs=[]
        for group in np.split(rows,np.flatnonzero(np.diff(rows)>1)+1):
            if len(group): runs.append((int(group[0]),int(group[-1])))
        overlaps=[[i for i,(lo,hi,_) in enumerate(previous) if not (b<lo or a>hi)] for a,b in runs]
        old_counts=[sum(i in o for o in overlaps) for i in range(len(previous))]
        current=[]
        for j,(lo,hi) in enumerate(runs):
            ov=overlaps[j]
            if len(ov)==1 and old_counts[ov[0]]==1:
                idx=previous[ov[0]][2]
            else: idx=len(cells); cells.append([])
            cells[idx].append((c,lo,hi)); current.append((lo,hi,idx))
        previous=current
    return cells


def sweep_order(cell,reverse_columns=False,reverse_first=False):
    result=[]
    for j,(c,lo,hi) in enumerate(reversed(cell) if reverse_columns else cell):
        rs=list(range(lo,hi+1))
        if bool(j%2)^bool(reverse_first): rs.reverse()
        result.extend((r,c) for r in rs)
    return result

class ClassicalPlanner:
    def __init__(self,method,options=None):
        if method not in {'bcd','priority_dijkstra','priority_stc','weighted_latency_greedy'}: raise ValueError(method)
        self.method=method; self.options=options or {}; self.actions=deque(); self.targets=deque(); self.initialized=False
    def _prepare(self,core):
        self.initialized=True
        if self.method=='bcd':
            cells=boustrophedon_cells(core.active_mask)
            # Deterministic spatial cell order; choose a starting cell near start,
            # then tie row-major. Orientation uses Manhattan entry distance.
            remaining=list(range(len(cells))); pos=core.drone_pos
            while remaining:
                candidates=[]
                for i in remaining:
                    for rev in (False,True):
                        for flip in (False,True):
                            route=sweep_order(cells[i],rev,flip); e=route[0]
                            candidates.append((abs(e[0]-pos[0])+abs(e[1]-pos[1]),i,rev,flip,route))
                _,i,_,_,route=min(candidates,key=lambda z:z[:4]); remaining.remove(i)
                self.targets.extend(route); pos=route[-1]
        elif self.method=='priority_stc':
            pos=core.drone_pos
            for level in (5,4,3,2,1):
                comps=connected_components(core.weight_map==level)
                while comps:
                    entries=[]
                    for j,comp in enumerate(comps):
                        root=min(comp,key=lambda p:(abs(p[0]-pos[0])+abs(p[1]-pos[1]),p))
                        entries.append((abs(root[0]-pos[0])+abs(root[1]-pos[1]),root,j))
                    _,root,j=min(entries); nodes=set(comps.pop(j)); seen={root}; route=[root]
                    # Explicit DFS spanning tree traversal with edge backtracking.
                    stack=[(root,iter(self._neighbors(root,nodes)))]
                    while stack:
                        p,it=stack[-1]
                        try: nxt=next(it)
                        except StopIteration:
                            stack.pop()
                            if stack: route.append(stack[-1][0])
                            continue
                        if nxt in seen: continue
                        seen.add(nxt); route.append(nxt); stack.append((nxt,iter(self._neighbors(nxt,nodes))))
                    self.targets.extend(route); pos=route[-1]
    @staticmethod
    def _neighbors(p,nodes):
        return [(p[0]+dr,p[1]+dc) for dr,dc in MOVES if (p[0]+dr,p[1]+dc) in nodes]
    def act(self,core):
        if not self.initialized: self._prepare(core)
        while not self.actions:
            if self.method in {'bcd','priority_stc'}:
                while self.targets and self.targets[0]==core.drone_pos: self.targets.popleft()
                if not self.targets:
                    if core.covered_cells_count==core.target_cells_count: return None
                    raise RuntimeError("Planner exhausted before complete coverage; no hidden fallback")
                target=self.targets.popleft()
                # Consecutive sweep/tree neighbors are taken directly, preserving
                # their method's route. Gaps use explicit turn-aware shortest paths.
                delta=(target[0]-core.drone_pos[0],target[1]-core.drone_pos[1])
                if delta in MOVES: self.actions.append(MOVES.index(delta))
                else: self.actions.extend(target_path(shortest_paths(core),target)[1])
            else:
                search=shortest_paths(core); candidates=[]
                for target in map(tuple,np.argwhere(core.uncovered_mask())):
                    cost,actions=target_path(search,target); priority=float(core.weight_map[target])
                    if self.method=='weighted_latency_greedy':
                        score=cost/(priority+self.options.get('epsilon',1e-9))
                    else:
                        score=self.options.get('distance_weight',1.)*cost-self.options.get('priority_weight',3.)*priority/5.
                    candidates.append((score,cost,-priority,target,actions))
                if not candidates: return None
                chosen=min(candidates,key=lambda z:z[:4]); self.actions.extend(chosen[4])
                if not self.actions: raise RuntimeError("Unvisited target cannot equal serviced current cell")
        return int(self.actions.popleft())

@dataclass
class ExactResult:
    status: str
    objective: float|None
    lower_bound: float|None
    actions: list[int]
    expanded_states: int
    elapsed_sec: float
    max_active_cells: int


def exact_small_map(core,beta=.5,*,time_limit_sec=60.,max_states=1000000,max_active_cells=16):
    """Exact shortest path in (cell, heading, visited-bitmask, used steps) with Eq.(77) cost.

    No progress mask is imposed. Transition cost is
      [beta * W_remaining / P_total + (1-beta)] * dt.
    On SOLVED this equals beta*Lw+(1-beta)*T exactly. Resource-limit exits do
    not fabricate an optimum or an optimality gap. A finite incumbent may be absent.
    """
    if not 0<=beta<=1: raise ValueError("beta must be in [0,1]")
    active=list(map(tuple,np.argwhere(core.active_mask))); n=len(active)
    if n>max_active_cells:
        return ExactResult('active_limit',None,None,[],0,0.,max_active_cells)
    bit={p:1<<i for i,p in enumerate(active)}; allbits=(1<<n)-1
    weights=[float(core.weight_map[p]) for p in active]; total=sum(weights)
    initial=sum(bit[p] for p in active if core.covered_map[p])
    start=(*core.drone_pos,core.heading,initial,0)
    distances={start:0.}; previous={}; queue=[(0.,start)]; expanded=0; t0=time.monotonic()
    mass_cache={}
    while queue:
        cost,state=heapq.heappop(queue)
        if cost!=distances[state]: continue
        if state[3]==allbits:
            actions=[]; s=state
            while s!=start: s,a=previous[s]; actions.append(a)
            return ExactResult('optimal',cost,cost,list(reversed(actions)),expanded,time.monotonic()-t0,max_active_cells)
        if expanded>=max_states or time.monotonic()-t0>=time_limit_sec:
            return ExactResult('resource_limit',None,cost,[],expanded,time.monotonic()-t0,max_active_cells)
        expanded+=1; r,c,h,b,used_steps=state
        if used_steps>=core.max_steps:continue
        if b not in mass_cache: mass_cache[b]=sum(w for i,w in enumerate(weights) if not b&(1<<i))
        factor=beta*mass_cache[b]/total+1-beta
        for a,(dr,dc) in enumerate(MOVES):
            p=(r+dr,c+dc)
            if not core._in_bbox(*p): continue
            bb=b|bit.get(p,0); dt=core.time_straight_sec if a==h else core.time_turn_sec
            nxt=(*p,a,bb,used_steps+1); new=cost+factor*dt
            if new < distances.get(nxt,math.inf)-1e-12:
                distances[nxt]=new; previous[nxt]=(state,a); heapq.heappush(queue,(new,nxt))
    return ExactResult('unreachable',None,None,[],expanded,time.monotonic()-t0,max_active_cells)

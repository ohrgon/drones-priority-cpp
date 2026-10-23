"""Exact fixed-next-first-visit path diagnostics. No policy inference or learning.

The search state is (cell, absolute heading, remaining non-strict budget).
Other currently unvisited active cells are forbidden destinations, but STILL
participate in the original Strict mask and its multi-source distance field.
Reaching the target is allowed only with the recorded arrival heading.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import heapq
import math
import numpy as np

MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))
EPS = 1e-9


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def finite_close(actual, expected, label, atol=2e-6):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=2e-9, abs_tol=atol),
            f'{label}: {actual!r} != {expected!r}')


def distance_field(unvisited, bbox):
    """Exact four-neighbor STEP distance for the original Strict mask.

    This is NOT the weighted objective of SegmentGraph.shortest(). The A* search
    below still uses heading-dependent straight/turn travel time in seconds.

    The source task has a completely traversable rectangular bbox. Therefore a
    separable L1 distance transform is exactly its multi-source BFS distance.
    Compute it with NumPy only, without importing optional SciPy extensions.
    In particular, do not retry a broken SciPy/NumPy binary import per segment.
    """
    unvisited = np.asarray(unvisited, dtype=bool)
    bbox = np.asarray(bbox, dtype=bool)
    require(unvisited.ndim == 2 and unvisited.shape == bbox.shape, 'Bad grid shapes')
    shape = unvisited.shape
    inf = np.int32(1_000_000)
    rr, cc = np.where(bbox)
    require(len(rr) > 0, 'Empty bbox')
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    rectangle = np.zeros(shape, bool)
    rectangle[r0:r1+1, c0:c1+1] = True
    require(np.array_equal(rectangle, bbox),
            'Diagnostic supports the original rectangular traversable bbox only.')
    require(np.all(~unvisited | bbox), 'Unvisited cells outside bbox')
    dist = np.full(shape, inf, dtype=np.int32)
    if not unvisited.any():
        # Preserve the original terminal/no-target convention exactly.
        dist[bbox] = 0
        return dist
    patch = unvisited[r0:r1+1, c0:c1+1]
    cost = np.where(patch, np.int32(0), inf).astype(np.int32)
    for axis in (0, 1):
        index_shape = [1, 1]
        index_shape[axis] = cost.shape[axis]
        index = np.arange(cost.shape[axis], dtype=np.int32).reshape(index_shape)
        # min_j (cost[j] + abs(i-j)), first j<=i, then j>=i.
        forward = np.minimum.accumulate(cost - index, axis=axis) + index
        backward = np.flip(
            np.minimum.accumulate(np.flip(cost + index, axis=axis), axis=axis),
            axis=axis,
        ) - index
        cost = np.minimum(forward, backward)
    dist[r0:r1+1, c0:c1+1] = cost
    return dist


@dataclass
class PathResult:
    seconds: float
    actions: tuple[int, ...]
    expanded: int = 0
    @property
    def steps(self):
        return len(self.actions)


class SegmentGraph:
    def __init__(self, unvisited, bbox, time_straight, time_turn, bmax=6):
        self.u = np.asarray(unvisited, bool).copy()
        self.box = np.asarray(bbox, bool).copy()
        require(self.u.shape == self.box.shape and self.u.ndim == 2, 'Bad grid shapes')
        require(self.u.any(), 'A next-target search needs an unvisited cell')
        self.rows, self.cols = self.u.shape
        self.n = self.u.size
        self.ts, self.tt = float(time_straight), float(time_turn)
        require(self.tt >= self.ts > 0, 'Need t_turn >= t_straight > 0')
        self.bmax = int(bmax)
        self.d = distance_field(self.u, self.box)
        self.neighbor = np.full((self.n,4), -1, np.int32)
        self.strict_bits = np.zeros(self.n, np.uint8)
        rr, cc = np.indices(self.u.shape)
        direct_bits = np.zeros(self.n, np.uint8)
        progress_bits = np.zeros(self.n, np.uint8)
        for a,(dr,dc) in enumerate(MOVES):
            nr, nc = rr+dr, cc+dc
            valid = self.box & (nr>=0)&(nr<self.rows)&(nc>=0)&(nc<self.cols)
            src = np.flatnonzero(valid)
            nr1, nc1 = nr.ravel()[src], nc.ravel()[src]
            good = self.box[nr1,nc1]
            src, nr1, nc1 = src[good], nr1[good], nc1[good]
            self.neighbor[src,a] = nr1*self.cols+nc1
            direct_bits[src] |= (self.u[nr1,nc1].astype(np.uint8) << a)
            progress_bits[src] |= ((self.d[nr1,nc1] == self.d.ravel()[src]-1).astype(np.uint8) << a)
        self.strict_bits = np.where(direct_bits != 0, direct_bits, progress_bits).astype(np.uint8)

    def cell(self, rc):
        r,c = map(int, rc)
        require(0<=r<self.rows and 0<=c<self.cols and self.box[r,c], f'Outside bbox: {rc}')
        return r*self.cols+c

    def encode(self, cell, heading, budget):
        return (int(budget)*4+int(heading))*self.n+int(cell)

    def decode(self, state):
        bh, p = divmod(int(state), self.n)
        b, h = divmod(bh, 4)
        return p, h, b

    def check_path(self, start, heading, budget, target, end_heading, actions):
        p = self.cell(start); t = self.cell(target)
        require(not self.u.ravel()[p] and self.u.ravel()[t], 'Wrong first-visit segment endpoints')
        h,b = int(heading), int(budget)
        cost = 0.0
        for k,a in enumerate(actions):
            a = int(a)
            require(0<=a<4, 'Bad action')
            q = int(self.neighbor[p,a])
            strict = bool(int(self.strict_bits[p]) & (1<<a))
            require(q>=0 and (b>0 or strict), 'Illegal B6 reference movement')
            if self.u.ravel()[q]:
                require(q==t and k==len(actions)-1 and a==end_heading,
                        'Reference reaches wrong/new/early target or wrong arrival heading')
            cost += self.ts if h==a else self.tt
            if not strict: b -= 1
            require(b>=0, 'Negative detour budget')
            p,h = q,a
        require(p==t and h==end_heading, 'Reference misses endpoint')
        return cost

    def shortest(self, start, heading, budget, target, end_heading, upper_time,
                 step_limit=None):
        """A* with an admissible Manhattan*time_straight lower bound.

        Without a step limit, standard state labels are sufficient. With a limit,
        keep nondominated (time, steps) labels per state (not a single label).
        No sampled actions, heuristic fallback, or approximation is used.
        """
        s,t = self.cell(start),self.cell(target)
        require(not self.u.ravel()[s] and self.u.ravel()[t], 'Invalid search target')
        require(0<=int(heading)<4 and 0<=int(end_heading)<4 and 0<=budget<=self.bmax, 'Bad heading/budget')
        require(upper_time>0 and math.isfinite(upper_time), 'Bad recorded-time upper bound')
        if step_limit is not None:
            require(int(step_limit)>=1, 'Bad resource limit')
        tr,tc = divmod(t,self.cols)
        def heuristic(p):
            r,c = divmod(p,self.cols)
            return (abs(r-tr)+abs(c-tc))*self.ts
        sid=self.encode(s,heading,budget)
        # A directly adjacent target is a provably shortest one-edge connection
        # when it has the required arrival heading.
        a=int(end_heading)
        if int(self.neighbor[s,a])==t and (budget>0 or int(self.strict_bits[s])&(1<<a)):
            dt=self.ts if int(heading)==a else self.tt
            require(dt<=upper_time+EPS, 'Direct reference slower than observed upper bound')
            return PathResult(dt,(a,),0)
        if step_limit is None:
            best={sid:0.0}; counts={sid:0}; parent={}
            heap=[(heuristic(s),0.0,0,sid)]; expanded=0
            while heap:
                _,g,k,st=heapq.heappop(heap)
                if g>best.get(st,math.inf)+EPS or k!=counts.get(st): continue
                p,h,b=self.decode(st)
                if p==t:
                    path=[]
                    while st!=sid:
                        st,a=parent[st];path.append(a)
                    return PathResult(g,tuple(reversed(path)),expanded)
                expanded+=1
                bits=int(self.strict_bits[p])
                for a in range(4):
                    q=int(self.neighbor[p,a]); strict=bool(bits&(1<<a))
                    if q<0 or (b==0 and not strict): continue
                    if self.u.ravel()[q] and (q!=t or a!=end_heading): continue
                    nb=b-int(not strict)
                    ng=g+(self.ts if a==h else self.tt); nk=k+1
                    if ng+heuristic(q)>upper_time+EPS: continue
                    ns=self.encode(q,a,nb)
                    old=best.get(ns,math.inf)
                    if ng<old-EPS or (abs(ng-old)<=EPS and nk<counts.get(ns,10**9)):
                        best[ns]=ng;counts[ns]=nk;parent[ns]=(st,a)
                        heapq.heappush(heap,(ng+heuristic(q),ng,nk,ns))
        else:
            # A resource-constrained shortest path: an expensive short prefix
            # cannot be discarded merely because a cheap longer prefix exists.
            limit=int(step_limit)
            labels=[(sid,0.0,0,-1,-1)]
            fronts={sid:[0]}; alive={0}
            heap=[(heuristic(s),0.0,0,0)];expanded=0
            while heap:
                _,g,k,lid=heapq.heappop(heap)
                if lid not in alive: continue
                st,g,k,par,action=labels[lid]
                p,h,b=self.decode(st)
                if p==t:
                    path=[]
                    while labels[lid][3]>=0:
                        path.append(labels[lid][4]);lid=labels[lid][3]
                    return PathResult(g,tuple(reversed(path)),expanded)
                expanded+=1
                if k==limit: continue
                bits=int(self.strict_bits[p])
                for a in range(4):
                    q=int(self.neighbor[p,a]); strict=bool(bits&(1<<a))
                    if q<0 or (b==0 and not strict): continue
                    if self.u.ravel()[q] and (q!=t or a!=end_heading): continue
                    nr,nc=divmod(q,self.cols); nk=k+1
                    if nk+abs(nr-tr)+abs(nc-tc)>limit: continue
                    ng=g+(self.ts if a==h else self.tt)
                    if ng+heuristic(q)>upper_time+EPS: continue
                    ns=self.encode(q,a,b-int(not strict)); prev=fronts.get(ns,[])
                    if any(labels[x][1]<=ng+EPS and labels[x][2]<=nk for x in prev): continue
                    kept=[]
                    for x in prev:
                        if ng<=labels[x][1]+EPS and nk<=labels[x][2]: alive.discard(x)
                        else: kept.append(x)
                    new=len(labels);labels.append((ns,ng,nk,lid,a));alive.add(new)
                    fronts[ns]=kept+[new]
                    heapq.heappush(heap,(ng+heuristic(q),ng,nk,new))
        raise RuntimeError('No admissible reference path; the observed segment must itself be feasible. '
                           'Check source/trajectory identity. No unconstrained fallback used.')


def action_positions(start, actions):
    p=list(map(int,start)); out=[tuple(p)]
    for a in actions:
        dr,dc=MOVES[int(a)];p[0]+=dr;p[1]+=dc;out.append(tuple(p))
    return np.asarray(out,dtype=np.int32)


def analyze_connections(priority, bbox, positions, actions, times, start_heading,
                        budgets, new_visits, ts, tt, max_steps, native_original=None,
                        native_reference=None, progress=None, recorded_masks=None,
                        recorded_rewards=None, recorded_ratios=None):
    """Validate then optimize recorded reached-target segments.

    Main E is the exact fixed-target/heading/B minimum under remaining original H.
    A second nonexpanding-step witness is computed ONLY when necessary, so the
    concatenated diagnostic path is guaranteed not to require more steps than
    the recorded path. It is NOT a policy rollout, and does not repair failures.
    """
    priority=np.asarray(priority);bbox=np.asarray(bbox,bool)
    pos=np.asarray(positions);act=np.asarray(actions);tm=np.asarray(times,float)
    L=len(act);B=6
    require(priority.ndim==2 and pos.shape==(L+1,2) and tm.shape==(L+1,), 'Trajectory shape mismatch')
    require(np.asarray(budgets).shape==(L,) and np.asarray(new_visits).shape==(L,), 'Budget/event shape mismatch')
    require(np.array_equal(pos,action_positions(pos[0],act)), 'Positions differ from recorded actions')
    finite_close(tm[0],0,'Initial time')
    covered=np.zeros_like(priority,bool)
    start=tuple(map(int,pos[0]))
    if priority[start]>0: covered[start]=True
    b=6;h=int(start_heading);n0=int(covered.sum());W0=float(priority.sum())
    require(W0>0 and L<=max_steps, 'Empty map or overrun')
    rows=[]; reference_actions=[]; segment_start=0; graph=None
    if L == 0:
        require(np.array_equal(covered, priority>0), 'Zero-step incomplete path')
        for native in (native_original, native_reference):
            if native is not None:
                require(native.complete_at_reset, 'Native singleton disagrees')
                native.step(0)
    original_wait=0.0; reference_wait=0.0; elapsed_reference=0.0
    gap_exact=0.0; lw_gap_exact=0.0; expansions=0; graph_count=0
    for t,a0 in enumerate(act):
        a=int(a0);require(0<=a<4, 'Invalid action code')
        if graph is None:
            graph=SegmentGraph((priority>0)&~covered,bbox,ts,tt)
            graph_count+=1
        p=tuple(map(int,pos[t])); q=tuple(map(int,pos[t+1]));cell=graph.cell(p)
        strict=bool(int(graph.strict_bits[cell])&(1<<a))
        expectedmask=(graph.neighbor[cell]>=0) & ((b>0)|np.array([bool(int(graph.strict_bits[cell])&(1<<x)) for x in range(4)]))
        require(expectedmask[a], f'Original path violates B6 at step {t}')
        if recorded_masks is not None:
            require(np.array_equal(np.asarray(recorded_masks[t],bool),expectedmask),
                    f'Stored/native mask differs at step {t}')
        if recorded_ratios is not None:
            finite_close(float(recorded_ratios[t]),1-float(priority[(priority>0)&~covered].sum())/W0,
                         'Recorded cumulative priority ratio')
        finite_close(float(budgets[t]),b,'Recorded pre-action budget',atol=2e-6)
        dt=ts if a==h else tt
        finite_close(tm[t+1]-tm[t],dt,f'Time at step {t}')
        is_new=bool(priority[q]>0 and not covered[q])
        require(int(new_visits[t])==int(is_new), f'First-visit event differs at step {t}')
        mass=float(priority[(priority>0)&~covered].sum())
        original_wait += mass/W0*dt
        if native_original is not None:
            require(np.array_equal(np.asarray(native_original.action_masks(),bool),expectedmask),
                    f'Independent/native mask disagreement at step {t}')
            require(tuple(native_original.drone_pos)==p and int(native_original.heading)==h,
                    'Native original state mismatch')
            ob,r,term,trunc,info=native_original.step(a)
            require(tuple(native_original.drone_pos)==q, 'Native original movement mismatch')
            if recorded_rewards is not None:
                finite_close(r,float(recorded_rewards[t]),'Native vs recorded step reward',atol=2e-8)
            finite_close(native_original.elapsed_time_sec,tm[t+1],'Native original elapsed time')
            require(bool(term or trunc)==(t==L-1), 'Native stop does not match recorded trajectory')
        b=6 if is_new else b-int(not strict);h=a
        if is_new:
            s=segment_start;e=t+1
            sh=int(start_heading) if s==0 else int(act[s-1])
            sb=int(round(float(budgets[s])))
            original_seg=tuple(map(int,act[s:e])); actual=float(tm[e]-tm[s])
            graph.check_path(pos[s],sh,sb,q,h,original_seg)
            opt=graph.shortest(pos[s],sh,sb,q,h,actual)
            remaining=max_steps-s
            if opt.steps>remaining:
                opt=graph.shortest(pos[s],sh,sb,q,h,actual,step_limit=remaining)
            witness=opt
            if witness.steps>e-s:
                witness=graph.shortest(pos[s],sh,sb,q,h,actual,step_limit=e-s)
            finite_close(graph.check_path(pos[s],sh,sb,q,h,opt.actions),opt.seconds,'Optimal path verification')
            finite_close(graph.check_path(pos[s],sh,sb,q,h,witness.actions),witness.seconds,'Witness path verification')
            require(opt.seconds<=witness.seconds+EPS and witness.seconds<=actual+2e-6,
                    'Invalid reference lower/upper bound ordering')
            require(witness.steps<=e-s,'Constructive path exceeds per-segment step budget')
            gap=max(0.,actual-opt.seconds);wgap=max(0.,actual-witness.seconds)
            dnear=int(graph.d[tuple(map(int,pos[s]))])
            require(e-s<=dnear+2*sb, 'Original segment violates B6 distance-progress bound')
            # All unvisited masses are constant until the final edge of this segment.
            reference_wait += mass/W0*witness.seconds
            elapsed_reference += witness.seconds
            gap_exact += gap;lw_gap_exact += mass/W0*gap
            rows.append(dict(segment=len(rows),start_step=s,end_step=e,
                start_r=int(pos[s,0]),start_c=int(pos[s,1]),start_heading=sh,start_budget=sb,
                target_r=q[0],target_c=q[1],target_priority=int(priority[q]),end_heading=h,
                remaining_active=int(((priority>0)&~covered).sum()),remaining_priority_ratio=mass/W0,
                nearest_new_cell_steps=dnear,actual_steps=e-s,actual_time_sec=actual,
                actual_revisit_steps=int(sum(priority[tuple(x)]>0 for x in pos[s+1:e])),
                actual_zero_transit_steps=int(sum(priority[tuple(x)]==0 for x in pos[s+1:e])),
                optimal_steps=opt.steps,optimal_time_sec=opt.seconds,
                connection_excess_sec=gap,connection_excess_pct=100*gap/actual,
                latency_saving_fixed_order_sec=mass/W0*gap,
                witness_steps=witness.steps,witness_time_sec=witness.seconds,
                witness_connection_saving_sec=wgap,
                witness_latency_saving_sec=mass/W0*wgap,
                nonexpanding_step_constraint_active=int(opt.steps>e-s),
                counterfactual_witness_first_time_sec=elapsed_reference,
                astar_expanded=opt.expanded+witness.expanded*(witness is not opt)))
            expansions+=rows[-1]['astar_expanded']
            reference_actions.extend(witness.actions)
            if native_reference is not None:
                for ar in witness.actions:
                    require(bool(native_reference.action_masks()[ar]),'Native rejects diagnostic witness')
                    native_reference.step(ar)
                require(tuple(native_reference.drone_pos)==q and int(native_reference.heading)==h
                        and int(native_reference.detour_budget)==6,'Native witness endpoint mismatch')
                finite_close(native_reference.elapsed_time_sec,elapsed_reference,'Native witness time')
            covered[q]=True;segment_start=e;graph=None
            if progress and len(rows)%200==0: progress(len(rows),L)
    tail=L-segment_start
    tail_time=float(tm[-1]-tm[segment_start])
    if tail:
        # No observed next target exists: preserve, separately report, never erase it.
        require(not np.asarray(new_visits)[segment_start:].any(),'Tail contains a visit')
        mass=float(priority[(priority>0)&~covered].sum())
        reference_wait+=mass/W0*tail_time
        reference_actions.extend(map(int,act[segment_start:]))
        elapsed_reference+=tail_time
        if native_reference is not None:
            for ar in act[segment_start:]:
                require(bool(native_reference.action_masks()[int(ar)]),'Native rejects unchanged unresolved tail')
                native_reference.step(int(ar))
    success=bool(np.array_equal(covered,priority>0))
    if native_reference is not None:
        require(np.array_equal(native_reference.covered_map,covered),'Witness changes visited set/order')
        finite_close(native_reference.elapsed_time_sec,elapsed_reference,'Witness total time')
        finite_close(native_reference.weighted_waiting_sum/W0,reference_wait,'Witness weighted waiting')
        require(len(reference_actions)<=L,'Witness longer in steps than observed')
        if success: require(native_reference._done,'Constructive complete reference did not terminate')
    if recorded_ratios is not None:
        finite_close(float(recorded_ratios[-1]),float(priority[covered].sum())/W0,'Final priority ratio')
    summary=dict(is_success=int(success),steps=L,mission_time_sec=float(tm[-1]),
        active_cells=int((priority>0).sum()),initial_visited_active=n0,visited_active=int(covered.sum()),
        coverage_pct=100*covered.sum()/(priority>0).sum(),reached_segments=len(rows),
        reached_segment_time_sec=float(tm[segment_start]),unresolved_tail_steps=tail,
        unresolved_tail_time_sec=tail_time,
        connection_excess_sec=gap_exact,
        connection_excess_pct=100*gap_exact/tm[-1] if tm[-1]>0 else 0.,
        fixed_order_min_segment_sum_time_sec=float(tm[-1])-gap_exact,
        # This lower bound is not a full new policy rollout; the witness below is cap-safe.
        witness_steps=len(reference_actions),witness_time_sec=elapsed_reference,
        witness_saving_sec=float(tm[-1])-elapsed_reference,
        witness_saving_pct=100*(float(tm[-1])-elapsed_reference)/tm[-1] if tm[-1]>0 else 0.,
        witness_respects_original_execution_cap=len(reference_actions)<=max_steps,
        original_Lw_sec=original_wait if success else None,
        witness_Lw_sec=reference_wait if success else None,
        fixed_order_min_Lw_sec=original_wait-lw_gap_exact if success else None,
        original_censored_waiting_sec=original_wait,
        witness_censored_waiting_sec=reference_wait,
        latency_connection_saving_sec=lw_gap_exact if success else None,
        witness_latency_saving_sec=original_wait-reference_wait if success else None,
        limited_segments=sum(r['nonexpanding_step_constraint_active'] for r in rows),
        reference_is_policy_rollout=False,
        tail_scope='No target invented; incomplete observed tail retained unchanged',
        astar_expanded=expansions,segment_distance_fields=graph_count)
    return rows,summary,np.asarray(reference_actions,np.int8),covered

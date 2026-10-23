"""Source-based mask/reward tests used BEFORE any production optimizer update."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import math
import numpy as np


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def audit_fourway(rt,spec):
    trainer,utils,ev,new,features,entry=rt
    old=new.original
    cfg=trainer.ExperimentEnvConfig(**spec['environment'])
    require(cfg.tradeoff_beta==.5 and cfg.step_budget_factor==5 and cfg.detour_budget_max==6,'Wrong fixed comparison settings.')
    arr=np.zeros((50,50),np.uint8);arr[2:9,2:9]=1;arr[8,8]=5
    # The new reward wrapper must leave every physical mode untouched.
    native_factory=getattr(trainer,'_nrg_original_make_core',None)
    require(callable(native_factory),'Unwrapped ORIGINAL factory missing.')
    state_keys=('steps','drone_pos','heading','detour_budget','covered_cells_count',
        'remaining_priority_mass','elapsed_time_sec','weighted_waiting_sum',
        'valid_move_count','invalid_action_count','path_length_m',
        'move_turn_count','move_straight_count','revisit_steps','zero_transit_steps')
    mask_checks=[];paired=0;invalid_probes=0
    for mode in new.MODES:
        config=replace(cfg,mask_mode=mode)
        physical=native_factory(config);core=trainer.make_core(config)
        a,_=physical.reset(arr,start=(2,2),heading=1,seed=42)
        b,_=core.reset(arr,start=(2,2),heading=1,seed=42)
        for key in a:require(np.array_equal(a[key],b[key]),'Mask wrapper changed initial observation.')
        rng=np.random.default_rng(510)
        for j in range(45):
            mask=core.action_masks();boundary,strict=new.reference_masks(core)
            expected=new.selected_mask(core,boundary,strict,new.observed_budget(b,6))
            require(np.array_equal(mask,expected) and np.array_equal(mask,physical.action_masks()),'Mask differs from native.')
            if mode=='none' and j==0:action=0   # deliberate out-of-bbox attempt
            else:action=int(rng.choice(np.flatnonzero(mask)))
            before_mass=core.remaining_priority_mass;before_n=core.covered_cells_count
            before_t=core.elapsed_time_sec;before_invalid=core.invalid_action_count
            preview=core.immediate_reward(action)
            a,pr,pd,pt,pi=physical.step(action);b,r,d,tr,info=core.step(action)
            require((pd,pt)==(d or tr,False),'Public/native stop relation changed.')
            for key in state_keys:require(getattr(physical,key)==getattr(core,key),'Physical state changed: '+key)
            for key in a:require(np.array_equal(a[key],b[key]),'Observation changed: '+key)
            dn=core.covered_cells_count-before_n;dt=core.elapsed_time_sec-before_t
            inv=core.invalid_action_count-before_invalid
            expected_parts=old.reward_parts(new_count=dn,n_active=core.target_cells_count,
                mass_before=before_mass,total_mass=core.total_priority_mass,dt=dt,time_cap=core._nrg_cap,
                beta=.5,completed=d,exhausted=tr)
            expected_parts['invalid']=-cfg.invalid_action_penalty*inv
            for k,v in expected_parts.items():
                require(math.isclose(info['step_reward_'+k],v,abs_tol=1e-9),'Wrong reward component: '+k)
            require(math.isclose(r,preview,abs_tol=1e-9),'Preview and step differ.')
            require(math.isclose(r,sum(expected_parts.values()),abs_tol=1e-9),'Raw reward mismatch.')
            paired+=1;invalid_probes+=inv
            if d or tr:break
        physical.assert_consistent();core.assert_consistent()
        mask_checks.append(dict(mode=mode,steps=core.steps,invalid=core.invalid_action_count,
            physical_states_and_observations_identical=True))
    require(invalid_probes>0,'No invalid attempt exercised.')
    # All modes have exactly the same legal reward for the same state/action.
    rr=[]
    for mode in new.MODES:
        c=trainer.make_core(replace(cfg,mask_mode=mode));c.reset(arr,start=(2,2),heading=1,seed=42)
        _,r,d,tr,i=c.step(1);rr.append(r)
    require(max(rr)-min(rr)<1e-12,'Legal reward differs by mask.')
    # Strict must forbid bypassing an adjacent low-priority cell. B6 allows it.
    bypass=np.zeros((50,50),np.uint8);bypass[4,4]=1;bypass[4,3]=1;bypass[4,7]=5
    strict_core=trainer.make_core(replace(cfg,mask_mode='progress'))
    relaxed=trainer.make_core(replace(cfg,mask_mode='relaxed'))
    strict_core.reset(bypass,start=(4,4),heading=1);relaxed.reset(bypass,start=(4,4),heading=1)
    require(not strict_core.action_masks()[1] and relaxed.action_masks()[1],'B6 cannot bypass adjacent low priority.')
    relaxed.step(1);require(relaxed.detour_budget==5,'Non-strict bypass did not consume one budget.')
    # Verify the old FULL Relaxed environment and the new Relaxed environment
    # step-for-step using the actual native factory, NOT re-created physics.
    old_factory=old.wrap_factory(native_factory)
    a=old_factory(replace(cfg,mask_mode='relaxed'))
    b=trainer.make_core(replace(cfg,mask_mode='relaxed'))
    oa,_=a.reset(arr,start=(5,5),heading=1,seed=42)
    ob,_=b.reset(arr,start=(5,5),heading=1,seed=42)
    equivalent=0;exhausted_budget=0;free_reset=False
    for j in range(a.max_steps):
        require(np.array_equal(a.action_masks(),b.action_masks()),'New Relaxed mask differs from current FULL.')
        for key in oa:require(np.array_equal(oa[key],ob[key]),'New Relaxed observation differs.')
        _,strict=new.reference_masks(a);detours=np.flatnonzero(a.action_masks()&~strict)
        action=int(detours[0]) if len(detours) else int(np.flatnonzero(a.action_masks())[0])
        if a.detour_budget==0:exhausted_budget+=1
        oa,ra,da,ta,ia=a.step(action);ob,rb,db,tb,ib=b.step(action)
        require((da,ta)==(db,tb),'New Relaxed termination differs.')
        require(math.isclose(ra,rb,abs_tol=1e-10),'New Relaxed reward differs from original FULL.')
        for key in state_keys:require(getattr(a,key)==getattr(b,key),'New Relaxed state differs: '+key)
        equivalent+=1
        if ib['is_new_active_cell']:require(b.detour_budget==6,'New cell failed to reset B6.');free_reset=True
        if da or ta:break
    require(exhausted_budget>0 and free_reset,'Budget exhaustion/reset not exercised.')
    a.assert_consistent();b.assert_consistent()
    # At a boundary, repeated invalid attempts must cost time and penalty, but
    # never create distance, new coverage or an in-place history update.
    c=trainer.make_core(replace(cfg,mask_mode='none'));obs,_=c.reset(bypass,start=(4,3),heading=1)
    before={k:deepcopy(getattr(c,k)) for k in ('drone_pos','heading','visited_map','position_history','covered_map')}
    for _ in range(c.max_steps):
        _,r,d,tr,i=c.step(0)
        require(i['step_reward_invalid']==-1. and r<-1.,'Invalid attempt is free or penalty changed.')
    for k,v in before.items():require(np.array_equal(v,getattr(c,k)),'Invalid action changed spatial state: '+k)
    require(c.steps==c.invalid_action_count and c.valid_move_count==0 and c.path_length_m==0,'Phantom movement.')
    require(tr and not d and not i['is_success'] and i['step_reward_terminal']==0.,'Cap confused with completion.')
    require(math.isclose(c.elapsed_time_sec,c.steps*c.time_straight_sec,abs_tol=1e-9),'Invalid control time missing.')
    c.assert_consistent()
    invalid_total = invalid_probes + c.invalid_action_count
    # Every non-None case must reject rather than silently repair a masked action.
    rejected=[]
    for mode in ('boundary','relaxed','progress'):
        c=trainer.make_core(replace(cfg,mask_mode=mode));c.reset(bypass,start=(4,3),heading=1)
        try:c.step(0)
        except (RuntimeError,ValueError,AssertionError):rejected.append(mode)
        else:raise RuntimeError('Masked action silently executed.')
    return dict(mask_modes=mask_checks,paired_native_transitions=paired,
        current_FULL_relaxed_equivalent_transitions=equivalent,
        invalid_control_attempts_checked=invalid_total,
        invalid_penalty=1.,same_legal_reward_all_four=True,
        no_phantom_movement=True,budget_exhaustion_checked=exhausted_budget,
        masked_action_rejections=rejected,adjacent_low_priority_bypass=True,
        no_clock_or_new_channel=True,optimizer_updates=0)

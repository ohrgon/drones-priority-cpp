"""Four-way current FULL comparisons. Actual final policies, not corrected paths.
No training/model selection. Preserve failed Lw as null and report invalid
control attempts separately. Full1400 includes validation180.
"""
from __future__ import annotations
from collections import Counter
import csv, hashlib, html, io, itertools, json, math
from pathlib import Path
import zipfile
import numpy as np
T='mission_time_sec'
LW='priority_weighted_first_visit_latency_sec'
METRICS=(T,LW,'steps','valid_move_count','invalid_action_count','invalid_control_time_sec','turn_count','revisit_steps','zero_transit_steps')
CASES=('none','boundary','relaxed','progress')
VERSION='full_cost11_cnn3_fourway_externalH5_v1_20260928'
REFERENCE_LABEL='original_relaxed_reference'
BASE_VERSION='nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926'

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def num(v):
    try:x=float(v)
    except (ValueError,TypeError):return None
    return x if math.isfinite(x) else None

def yes(v):return str(v).lower() in ('1','1.0','true')

def ok(row):return yes(row.get('is_success'))

def mean(values):
    x=[num(v) for v in values];x=[v for v in x if v is not None]
    return float(np.mean(x)) if x else None

def canon(v):
    s=str(v).lower();return 'validation' if s in ('val','valid','validation') else s

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clean(v):
    if isinstance(v,Path):return str(v)
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    return v

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    tmp.write_text(json.dumps(clean(value),indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def csvwrite(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.writing')
    fields=list(dict.fromkeys(k for r in data for k in r)) or ['status']
    with tmp.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)
    tmp.replace(p)

def physical_pair(a,b):
    for k in ('map_id','map_sha256','initial_pair_id'):
        require(a.get(k) not in (None,'') and a[k]==b.get(k),'Map/initial pair mismatch: '+k)
    for k in ('start_r','start_c','start_heading','time_straight_sec','time_turn_sec','active_cells','total_priority_mass','max_steps'):
        x,y=num(a.get(k)),num(b.get(k))
        require(x is not None and y is not None and math.isclose(x,y,abs_tol=1e-8,rel_tol=1e-10),'Physical conditions differ: '+k)

def mode_for_label(label):
    if label==REFERENCE_LABEL:return 'relaxed'
    require(label in CASES,'Unknown mask result label: '+label)
    return label

def load_evaluation(root,phase,label):
    root=Path(root);d=root/('validation_final' if phase=='validation' else 'full_test')
    if not (d/'evaluation_complete.json').is_file():return None
    identity=read(d/'evaluation_identity.json');param=identity['parameters'];end=read(d/'evaluation_complete.json')
    require(digest(param)==identity['sha256']==end['identity_sha256'],'Evaluation identity mismatch: '+str(d))
    require(param['phase']==phase,'Evaluation scope differs.')
    spec=param['spec'];rd=spec['reward_definition'];env=spec['environment'];b=.5;mode=mode_for_label(label)
    require(spec['beta']==env['tradeoff_beta']==b,'Recorded beta not requested beta: '+label)
    require(rd.get('version')==(BASE_VERSION if label==REFERENCE_LABEL else VERSION) and rd.get('cost_multiplier')==11.,'Wrong COST11 reward adapter.')
    require(env.get('mask_mode')==mode,'Evaluation mask differs.')
    require(env.get('invalid_action_penalty')==1.0 and env.get('invalid_time_rule')=='straight_control_interval' and env.get('invalid_heading_rule')=='preserve','Invalid conventions changed.')
    if label!=REFERENCE_LABEL:require(spec.get('fourway',{}).get('case')==label,'Missing four-way provenance.')
    require(rd['success_terminal']==2. and rd['failure_terminal']==0. and rd.get('coverage_multiplier',1.)==1.,'Not FULL event rewards.')
    require(env['detour_budget_max']==6 and env['step_budget_factor']==5.,'Wrong B6/H5.')
    require(spec.get('remaining_step_observation') is False and spec['encoder_ablation']['merged_feature_dim']==261,'Wrong network/input.')
    require(spec['external_stop']['normalization_step_factor']==5.,'Normalizer changed.')
    if (root/'protocol.json').is_file():
        require(sha(root/'protocol.json')==param['protocol_sha256'],'Evaluation belongs to another source protocol.')
    data=rows(d/'per_map.csv');expected=180 if phase=='validation' else 1400
    require(len(data)==end['n_maps']==expected and len({r['map_id'] for r in data})==expected,'Incomplete or duplicate map rows.')
    listed={x[0]:x[1] for x in param['maps']}
    require(set(listed)=={r['map_id'] for r in data},'Evaluation map IDs differ.')
    if phase=='test':
        require(sum(canon(r['original_split'])=='validation' for r in data)==180 and sum(r['source']=='osm' for r in data)==500,'Wrong full1400 scope.')
    maxerr=0.
    for r in data:
        require(r['run_id']==spec['run_id'] and r['map_sha256']==listed[r['map_id']],'Mixed model/map.')
        require(r['phase']==phase and yes(r['deterministic']),'Wrong/stochastic evaluation.')
        require(canon(r['original_split']) in ('test','validation'),'Train/unknown map in evaluation.')
        if phase=='validation':require(canon(r['original_split'])=='validation' and r['source']=='synthetic','Wrong validation subset.')
        require((num(r[LW]) is not None)==ok(r),'Lw validity must depend on completion, never beta.')
        n=num(r['active_cells']);steps=num(r['steps']);tm=num(r[T]);wait=num(r['censored_weighted_waiting_time_sec'])
        require(n is not None and n>0 and steps is not None and tm is not None and wait is not None,'Missing physical metrics.')
        z=math.ceil(5*n)*num(r['time_turn_sec'])
        require(num(r['max_steps'])==math.ceil(5*n) and steps<=math.ceil(5*n),'H5 violated.')
        require(num(r['c3nt_normalization_steps'])==math.ceil(5*n) and math.isclose(num(r['nrg_time_cap_sec']),z,rel_tol=1e-10),'Normalizer not5N.')
        require(math.isclose(num(r['nrg_first_visit_coefficient']),1/n,abs_tol=1e-12),'First-visit reward removed.')
        expected_parts=dict(coverage=(num(r['first_visit_count'])-num(r['nrg_initial_covered_count']))/n,
            latency=-11*b*wait/z,mission_time=-11*(1-b)*tm/z,terminal=2. if ok(r) else 0.,invalid=-env['invalid_action_penalty']*num(r['invalid_action_count']))
        for k,x in expected_parts.items():
            y=num(r['reward_sum_'+k]);require(y is not None and math.isclose(x,y,abs_tol=2e-7,rel_tol=1e-9),'Wrong beta reward sum: '+label+' '+k)
            maxerr=max(maxerr,abs(x-y))
        require(math.isclose(num(r['total_reward']),sum(expected_parts.values()),abs_tol=3e-7,rel_tol=1e-9),'Raw return identity failed.')
        require(num(r['steps'])==num(r['valid_move_count'])+num(r['invalid_action_count']),'Attempt/movement count mismatch.')
        require(mode=='none' or num(r['invalid_action_count'])==0.,'Invalid action in a constrained mask.')
        physical_time=num(r['move_straight_count'])*num(r['time_straight_sec'])+num(r['move_turn_count'])*num(r['time_turn_sec'])+num(r['invalid_control_time_sec'])
        require(math.isclose(tm,physical_time,abs_tol=1e-6,rel_tol=1e-9),'Control/movement time mismatch.')
        if ok(r):require(math.isclose(num(r[LW]),wait,abs_tol=1e-7,rel_tol=1e-9),'Completed Lw differs from weighted waiting integral.')
        r['beta']=b
    return dict(label=label,mask=mode,beta=b,phase=phase,root=root,spec=spec,rows=data,byid={r['map_id']:r for r in data},
        identity=identity,max_reward_component_error=maxerr)

def baseline_targets(path):
    """Validation-only original WLG/BCD. Full-test targets are not inferred."""
    p=Path(path)
    if p.is_dir():p=p/'baseline_targets_validation180.csv'
    raw={}
    if p.suffix.lower()=='.zip':
        with zipfile.ZipFile(p) as z:
            names=[n for n in z.namelist() if n.endswith('baseline_targets_validation180.csv')]
            require(len(names)==1,'Need original baseline_targets_validation180.csv in baseline archive.')
            require(z.getinfo(names[0]).file_size<10_000_000,'Oversized baseline target.')
            data=list(csv.DictReader(io.StringIO(z.read(names[0]).decode('utf-8-sig'))))
            for tag in ('bcd','wlg'):
                ns=[n for n in z.namelist() if n.endswith(tag+'_validation180.csv')]
                if len(ns)==1:
                    require(z.getinfo(ns[0]).file_size<30_000_000,'Oversized raw baseline.')
                    rr=list(csv.DictReader(io.StringIO(z.read(ns[0]).decode('utf-8-sig'))))
                    require(len(rr)==180 and len({r['map_id'] for r in rr})==180,'Wrong raw baseline map count.')
                    raw[tag]={r['map_id']:r for r in rr}
    else:data=rows(p)
    require(len(data)==180 and len({r['map_id'] for r in data})==180,'Expected unique baseline180.')
    for r in data:
        require(r['wlg_run_id']=='WeightedLatencyGreedy_model_free_priority_mass_seed42' and r['bcd_run_id']=='BCD_model_free_priority_mass_seed42','Wrong WLG/BCD identity.')
        require(yes(r['wlg_success']) and yes(r['bcd_success']),'Incomplete baseline needs separate handling.')
        require(num(r['T_WLG_sec'])>0 and num(r['Lw_BCD_sec'])>0,'Invalid baseline values.')
        if set(raw)=={'bcd','wlg'}:
            a,c=raw['bcd'][r['map_id']],raw['wlg'][r['map_id']]
            physical_pair(a,c)
            require(a['map_sha256']==r['map_sha256'] and a['initial_pair_id']==r['initial_pair_id'],'Baseline target/raw identity mismatch.')
            require(ok(a) and ok(c),'Raw baseline incomplete.')
            for key,x in (('T_BCD_sec',a[T]),('Lw_BCD_sec',a[LW]),('T_WLG_sec',c[T]),('Lw_WLG_sec',c[LW])):
                if key in r:require(math.isclose(num(r[key]),num(x),abs_tol=1e-7,rel_tol=1e-9),'Baseline values disagree.')
                r[key]=x
    return {r['map_id']:r for r in data}

def scopes_for(phase,data):
    if phase=='validation':return [('validation180',data)]
    return [('full1400',data),('test_only1220',[r for r in data if canon(r['original_split'])=='test']),
        ('synthetic_test720',[r for r in data if canon(r['original_split'])=='test' and r['source']=='synthetic']),
        ('OSM500',[r for r in data if r['source']=='osm']),
        ('validation_subset180',[r for r in data if canon(r['original_split'])=='validation'])]

def summarize(v,data,scope):
    good=[r for r in data if ok(r)]
    return dict(case=v['label'],mask=v['mask'],beta=v['beta'],phase=v['phase'],scope=scope,n_maps=len(data),success_n=len(good),
        success_pct=100*len(good)/len(data) if data else None,
        T_success_mean=mean(r[T] for r in good),Lw_success_mean=mean(r[LW] for r in good),
        coverage_pct_mean=mean(r['coverage_pct'] for r in data),T_all_including_unfinished=mean(r[T] for r in data),
        invalid_action_mean_all=mean(r['invalid_action_count'] for r in data),
        attempted_steps_mean_all=mean(r['steps'] for r in data),actual_movements_mean_all=mean(r['valid_move_count'] for r in data),
        reward_is_not_ranking_criterion=True,normalizer_and_cap='5N',cost_multiplier=11.)

def compare_pair(a,b,scope,ids):
    matched=[];gain=loss=0
    for mid in ids:
        x,y=a['byid'][mid],b['byid'][mid];physical_pair(x,y)
        if ok(x) and ok(y):matched.append((x,y))
        gain+=ok(y) and not ok(x);loss+=ok(x) and not ok(y)
    result=[]
    for metric in METRICS:
        x=mean(r[metric] for r,s in matched);y=mean(s[metric] for r,s in matched)
        result.append(dict(phase=a['phase'],scope=scope,reference=a['label'],candidate=b['label'],
            reference_mask=a['mask'],candidate_mask=b['mask'],n_evaluated=len(ids),n_common_success=len(matched),metric=metric,
            reference_mean=x,candidate_mean=y,difference=y-x if x is not None and y is not None else None,
            change_pct=100*(y/x-1) if x is not None and x>0 and y is not None else None,
            gained_success_n=int(gain),lost_success_n=int(loss)))
    return result

def case_review(root):
    root=Path(root);chosen=[]
    chosen.extend(p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.html'))
    for dirname in ('validation_final','full_test'):
        d=root/dirname
        if d.is_dir():chosen.extend(p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.html'))
    for d in (root/'runs').glob('*'):
        if d.is_dir():chosen.extend(p for p in d.iterdir() if p.is_file() and p.suffix in ('.json','.csv'))
    d=root/'source_snapshot'
    source_names={'c3nt_cost_sweep_generated_core.py','c3nt_cost_sweep_generated_logging.py',
        'trainers_ch4_c3nt_cost_sweep.py','trainers_ch4_cost11_fourway.py','nrg1_c3nt_cost5_features.py','core_ch4.py',
        'core_ch4_bounded_b6.py','coverage_core.py','full_v2_utils.py','evaluation_ch4.py','resume_ch4.py'}
    if d.is_dir():chosen.extend(p for p in d.iterdir() if p.is_file() and (p.name in source_names or 'cost11_fourway' in p.name or p.name=='fourway_source_audit.json'))
    tmp=root/'fourway_review.writing.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(chosen)):z.write(p,p.relative_to(root))
    tmp.replace(root/'fourway_review.zip')

def combined_report(source_root,parent,cases,targets_path=None):
    parent=Path(parent);parent.mkdir(parents=True,exist_ok=True)
    sources={REFERENCE_LABEL:Path(source_root),**{case:parent/case for case in cases}}
    models={};pending=[]
    for phase in ('validation','test'):
        models[phase]={}
        for label,root in sources.items():
            data=load_evaluation(root,phase,label)
            if data is None:
                pending.append(dict(case=label,phase=phase,status='missing_or_incomplete',role='read_only_reference' if label==REFERENCE_LABEL else 'requested_mask'));continue
            models[phase][label]=data
        vv=models[phase]
        if vv:
            first=next(iter(vv.values()))
            for v in vv.values():
                require(set(first['byid'])==set(v['byid']),'Different beta map sets.')
                for mid in first['byid']:physical_pair(first['byid'][mid],v['byid'][mid])
    for label in sources:
        a,b=models['validation'].get(label),models['test'].get(label)
        if a and b:
            for mid,x in a['byid'].items():
                y=b['byid'][mid];physical_pair(x,y);require(ok(x)==ok(y),'Validation/test success mismatch.')
                for key in METRICS+('total_reward',):
                    u,v=num(x[key]),num(y[key])
                    require((u is None and v is None) or (u is not None and v is not None and math.isclose(u,v,abs_tol=1e-7,rel_tol=1e-10)),
                        'Validation/test metric mismatch: '+key)
    target=baseline_targets(targets_path) if targets_path else None
    summaries=[];bygroup=[];commonrows=[];pairs=[];allrows=[];goals=[];group_goals=[]
    for phase,vv in models.items():
        for v in vv.values():
            for scope,rr in scopes_for(phase,v['rows']):
                summaries.append(summarize(v,rr,scope))
                for key in sorted({(r['source'],str(r['tier'])) for r in rr}):
                    sub=[r for r in rr if (r['source'],str(r['tier']))==key]
                    bygroup.append(dict(source=key[0],tier=key[1],**summarize(v,sub,scope)))
            allrows.extend(dict(case=v['label'],**r) for r in v['rows'])
            if phase=='validation' and target:
                require(set(target)==set(v['byid']),'Baseline maps differ.')
                for r in v['rows']:
                    b=target[r['map_id']]
                    require(r['map_sha256']==b['map_sha256'] and r['initial_pair_id']==b['initial_pair_id'],'Baseline physical pair differs.')
                    tt,ll=num(r[T]),num(r[LW]);bt,bl=num(b['T_WLG_sec']),num(b['Lw_BCD_sec'])
                    tw=ok(r) and tt<bt-1e-6;lw=ok(r) and ll<bl-1e-6
                    full=num(b.get('T_BCD_sec')) is not None and num(b.get('Lw_WLG_sec')) is not None
                    lowboth=loweither=dominates=None
                    if full:
                        tgood=ok(r) and tt<=max(bt,num(b['T_BCD_sec']))+1e-6
                        lgood=ok(r) and ll<=max(bl,num(b['Lw_WLG_sec']))+1e-6
                        lowboth=tgood and lgood;loweither=tgood or lgood
                        dominates=ok(r) and tt<min(bt,num(b['T_BCD_sec']))-1e-6 and ll<min(bl,num(b['Lw_WLG_sec']))-1e-6
                    goals.append(dict(case=v['label'],beta=v['beta'],map_id=r['map_id'],success=ok(r),T=tt,Lw=ll,
                        T_WLG=bt,Lw_BCD=bl,T_BCD=num(b.get('T_BCD_sec')),Lw_WLG=num(b.get('Lw_WLG_sec')),
                        T_win=tw,Lw_win=lw,both_win=tw and lw,lower_middle_both=lowboth,
                        lower_middle_either=loweither,strictly_better_both_baselines=dominates))
        if vv:
            first=next(iter(vv.values()))
            for scope,rr in scopes_for(phase,first['rows']):
                ids=[r['map_id'] for r in rr]
                for ka,kb in itertools.combinations(vv,2):pairs.extend(compare_pair(vv[ka],vv[kb],scope,ids))
                groups={'all_available':vv,'requested_only':{k:v for k,v in vv.items() if k!=REFERENCE_LABEL},
                    'requested_with_nonzero_success':{k:v for k,v in vv.items() if k!=REFERENCE_LABEL and any(ok(v['byid'][mid]) for mid in ids)}}
                for group,selected in groups.items():
                    if not selected:continue
                    common=set.intersection(*[{m for m in ids if ok(v['byid'][m])} for v in selected.values()])
                    for label,v in selected.items():
                        sub=[v['byid'][m] for m in sorted(common)]
                        c=dict(case=label,beta=v['beta'],phase=phase,scope=scope,comparison_group=group,
                            case_count=len(selected),n_scope=len(ids),n_common_success=len(sub))
                        for metric in METRICS:c[metric]=mean(r[metric] for r in sub)
                        if phase=='validation' and target:
                            c['T_WLG_same_common']=mean(target[r['map_id']]['T_WLG_sec'] for r in sub)
                            c['Lw_BCD_same_common']=mean(target[r['map_id']]['Lw_BCD_sec'] for r in sub)
                        commonrows.append(c)
    for label in sources:
        rr=[r for r in goals if r['case']==label]
        if not rr:continue
        good=[r for r in rr if r['success']];tm,lm=mean(r['T'] for r in good),mean(r['Lw'] for r in good)
        bt,bl=mean(r['T_WLG'] for r in good),mean(r['Lw_BCD'] for r in good)
        full=all(r['lower_middle_both'] is not None for r in rr)
        group_goals.append(dict(case=label,beta=rr[0]['beta'],n=180,success_n=len(good),
            T_wins=sum(r['T_win'] for r in rr),Lw_wins=sum(r['Lw_win'] for r in rr),both_wins=sum(r['both_win'] for r in rr),
            lower_middle_both=sum(r['lower_middle_both'] for r in rr) if full else None,
            lower_middle_either=sum(r['lower_middle_either'] for r in rr) if full else None,
            strictly_better_both_baselines=sum(r['strictly_better_both_baselines'] for r in rr) if full else None,
            T_own_success=tm,Lw_own_success=lm,WLG_T_same_success=bt,BCD_Lw_same_success=bl,
            all180_goal_pass=bool(len(good)==180 and tm<bt-1e-6 and lm<bl-1e-6)))
    csvwrite(parent/'summary_masks.csv',summaries);csvwrite(parent/'by_source_tier.csv',bygroup)
    csvwrite(parent/'common_success_comparison.csv',commonrows);csvwrite(parent/'paired_masks.csv',pairs)
    csvwrite(parent/'paired_vs_original_relaxed.csv',[r for r in pairs if r['reference']==REFERENCE_LABEL])
    csvwrite(parent/'per_map_all_masks.csv',allrows);csvwrite(parent/'baseline_goals_validation.csv',goals)
    csvwrite(parent/'baseline_goal_summary.csv',group_goals)
    status=dict(pending=pending,validation_complete=list(models['validation']),full_test_complete=list(models['test']),
        requested_masks=list(cases),reference_mask="relaxed_B6",beta=.5,reference_read_only=True,
        full1400_includes_validation180=True,independent_test_count=1220,
        baseline_targets='validation_only_loaded' if target else 'not_provided',
        full_test_baseline_goal='Not computed: validation targets do not cover full test.',
        invalid_attempt_is_not_movement=True,invalid_penalty_all_cases=1.,
        architecture_and_reward_runtime='CNN3/261/physical core unchanged; same FULL legal-transition formula; fixed original invalid convention',
        zero_success_group_note='requested_with_nonzero_success explicitly excludes any zero-success model only for conditional efficiency; main success table and all/requested common groups retain it',
        automatic_best_model_selection=False,raw_reward_rank=False,
        subset_warning='Common-success tables are conditional; inspect completion counts for all maps.',
        reward_component_audit={p:{k:v['max_reward_component_error'] for k,v in vv.items()} for p,vv in models.items()})
    write(parent/'comparison_status.json',status)
    text='<!doctype html><meta charset="utf-8"><title>COST11 FULL H5 four-way mask comparison</title><style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #bbb}pre{white-space:pre-wrap}</style>'
    text+='<h1>FULL COST11 / H5 — four-way masks</h1><p>Actual final-policy results, no corrected paths. Every mask has its own fresh PPO model; no mask switching at inference. FULL1400 includes validation180; independent test1220 is separate. No Mask invalid attempts retain positive control time and the fixed penalty. Raw reward is not a performance rank.</p>'
    for title,data,fields in [('Completion and own-success metrics',summaries,('case','scope','success_n','n_maps','coverage_pct_mean','T_success_mean','Lw_success_mean','invalid_action_mean_all')),
                             ('Matched original Relaxed-B6 comparison',[r for r in pairs if r['reference']==REFERENCE_LABEL and r['metric'] in (T,LW)],
                              ('candidate','scope','metric','n_common_success','reference_mean','candidate_mean','change_pct','gained_success_n','lost_success_n'))]:
        text+='<h2>'+title+'</h2><table><tr>'+''.join('<th>'+html.escape(f)+'</th>' for f in fields)+'</tr>'
        for r in data:text+='<tr>'+''.join('<td>'+html.escape(str(r.get(f,'')))+'</td>' for f in fields)+'</tr>'
        text+='</table>'
    text+='<h2>Status</h2><pre>'+html.escape(json.dumps(status,ensure_ascii=False,indent=2))+'</pre>'
    (parent/'index.html').write_text(text,encoding='utf-8')
    for label,root in sources.items():
        if label!=REFERENCE_LABEL and (root/'protocol.json').is_file():case_review(root)
    tmp=parent/'cost11_fourway_review.writing.zip'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(parent.iterdir()):
            if p.is_file() and p.suffix in ('.json','.csv','.html'):z.write(p,p.name)
        for label,root in sources.items():
            if label==REFERENCE_LABEL:continue
            p=root/'fourway_review.zip'
            if p.is_file():z.write(p,label+'/fourway_review.zip')
        for phase in ('validation','test'):
            v=models[phase].get(REFERENCE_LABEL)
            if v:
                d=v['root']/('validation_final' if phase=='validation' else 'full_test')
                for name in ('per_map.csv','evaluation_identity.json','evaluation_complete.json'):
                    z.write(d/name,'reference_original_relaxed/'+d.name+'/'+name)
    tmp.replace(parent/'cost11_fourway_review.zip')
    return status

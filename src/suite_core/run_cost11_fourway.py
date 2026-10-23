#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current FULL COST11: train/test No Mask, Boundary, Relaxed B6 and Strict PPO.
All cases use CNN3/261, beta=.5, gamma=1, H5; fresh 5M per case.
--phase all = train -> validation180 -> full1400 (independent test1220).
Original source/checkpoints are read-only. No model selection or teacher.
"""
from __future__ import annotations
import argparse, ast, csv, difflib, hashlib, importlib, importlib.util, json, math, os
from dataclasses import replace
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import shutil, subprocess, sys, tempfile, time, zipfile
import numpy as np

VERSION='full_cost11_cnn3_fourway_externalH5_v1_20260928'
SOURCE_RUN='ppo_relaxedB6_NRG1_C3NT_COST11_b0.5_g1_l0.95_budgetobs_seed42'
BASE_VERSION='nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926'
SOURCE_TRAINER='trainers_ch4_c3nt_cost_sweep'
SOURCE_CORE='c3nt_cost_sweep_generated_core'
SOURCE_LOG='c3nt_cost_sweep_generated_logging'
SOURCE_EVAL='c3nt_cost5_frozen_evaluation.py'
SOURCE_CORE_SHA='807817035c9d5c3da4c36caf2b56a52128f68f7eb50b2787873ac0215a4960b0'
EVAL_MODULE='cost11_fourway_generated_evaluation.py'
PREFLIGHT_MODULE='cost11_fourway_generated_preflight.py'
PACKAGE=('run_cost11_fourway.py','cost11_fourway_core.py','cost11_fourway_checks.py','cost11_fourway_report.py')
CASES=('none','boundary','relaxed','progress')
TRAINER='trainers_ch4_cost11_fourway'
CORE='cost11_fourway_core'
LOG='cost11_fourway_generated_logging'
LABELS={'none':'No Mask','boundary':'Boundary Only','relaxed':'Relaxed B6','progress':'Strict Progress'}

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def safe(value):
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.writing')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(safe(value),f,indent=2,ensure_ascii=False,allow_nan=False,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def abspath(value):
    p=Path(value).expanduser()
    return (p if p.is_absolute() else Path(__file__).resolve().parent/p).resolve()

def overlap(a,b):return a==b or a in b.parents or b in a.parents

@contextmanager
def output_lock(path):
    import fcntl
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError(f'Another process owns {p.parent}') from e
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def child(root,name):
    p=Path(root)/name
    require(not Path(name).is_absolute() and Path(root).resolve() in p.resolve().parents,'Unsafe source member: '+name)
    require(p.is_file() and not p.is_symlink(),'Missing/symlink source: '+str(p))
    return p

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None,'Cannot load '+str(path))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def source_context(root):
    root=Path(root).resolve();pp=root/'protocol.json';mf=root/'manifest.csv';snapshot=root/'source_snapshot'
    require(pp.is_file() and mf.is_file(),'Provide the COMPLETED COST11 root, not NRG1/COST1: '+str(root))
    p=read(pp);hashes=p.get('snapshot_hash')
    require(isinstance(hashes,dict) and hashes,'Frozen COST11 source hashes missing.')
    require(sha(mf)==p.get('manifest_sha256'),'Frozen manifest changed.')
    for name,h in hashes.items():require(sha(child(snapshot,name))==h,'Frozen source changed: '+name)
    required=(SOURCE_TRAINER+'.py',SOURCE_EVAL,SOURCE_CORE+'.py','nrg1_c3nt_cost5_features.py',
        SOURCE_LOG+'.py','run_nrg1_c3nt_cost5.py','coverage_core.py','core_ch4.py',
        'core_ch4_bounded_b6.py','full_v2_utils.py','resume_ch4.py','evaluation_ch4.py')
    require(all(n in hashes for n in required),'COST11 frozen runtime is incomplete.')
    require(hashes[SOURCE_CORE+'.py']==SOURCE_CORE_SHA,'Source COST11 reward core differs from reviewed version; do not guess.')
    run=root/'runs'/SOURCE_RUN
    for n in ('status.json','run_spec.json','training_settings.json','checkpoints.json','last_model.zip'):
        require((run/n).is_file(),'Original COST11 file missing: '+str(run/n))
    spec=read(run/'run_spec.json');settings=read(run/'training_settings.json');status=read(run/'status.json')
    require(spec.get('run_id')==SOURCE_RUN and spec.get('reward_case')=='NRG1_C3NT_COST11','Wrong source experiment.')
    require(spec.get('nrg1_protocol_sha256')==sha(pp),'Run/protocol mismatch.')
    clean={k:v for k,v in spec.items() if k!='nrg1_protocol_sha256'}
    require(clean==p['request']['spec'] and settings==p['request']['settings'],'Spec/settings not the frozen protocol.')
    require(status.get('state')=='completed' and status.get('actual_timesteps',0)>=5_000_000,'COST11 training not completed.')
    require(spec.get('seed')==42 and spec.get('algorithm')=='ppo' and spec.get('beta')==.5,'Expected original PPO beta0.5 seed42 source.')
    for k,v in dict(grid_size=50,mask_mode='relaxed',detour_budget_max=6,step_budget_factor=5.0,tradeoff_beta=.5,
        coverage_reward_scale=0.,objective_cost_scale=0.,success_terminal_reward=0.,failure_terminal_penalty=0.,paper_mode='none').items():
        require(spec['environment'].get(k)==v,'Unexpected source environment.'+k)
    require(spec['reward_definition']['version']==BASE_VERSION and spec['reward_definition']['cost_multiplier']==11.,'Not COST11.')
    require(spec['reward_definition']['success_terminal']==2. and spec['reward_definition']['failure_terminal']==0.,'Terminal reward differs.')
    require(spec.get('remaining_step_observation') is False,'Source still has a clock.')
    require(settings['total_timesteps']==5_000_000 and settings['ppo']['gamma']==1. and settings['ppo']['gae_lambda']==.95,
            'Source training hyperparameters differ.')
    cp=run/'last_model.zip';h=sha(cp);registered=read(run/'checkpoints.json')
    require(any(c.get('final') and Path(c['path']).name=='last_model.zip' and c['sha256']==h for c in registered),
            'COST11 final checkpoint hash/registry mismatch.')
    with zipfile.ZipFile(cp) as z:
        require(z.getinfo('data').file_size<32_000_000,'Oversized model metadata.')
        d=json.loads(z.read('data'));sb3=z.read('_stable_baselines3_version').decode().strip()
    require(sb3=='2.7.0' and d.get('num_timesteps')==status['actual_timesteps'],'Source checkpoint status/version mismatch.')
    require(d.get('gamma')==1. and d.get('gae_lambda')==.95,'Source checkpoint gamma/lambda differs.')
    with mf.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    require(rows and {'map_id','path','split','source','sha256'}.issubset(rows[0]),'Wrong manifest schema.')
    canon=lambda s:'validation' if s.lower() in ('val','valid','validation') else s.lower()
    require(all(Path(r['path']).is_absolute() for r in rows),'Frozen manifest requires original absolute paths.')
    require(all(r['source'] in ('synthetic','osm') and canon(r['split']) in ('train','test','validation') for r in rows),'Unknown split/source.')
    require(all(r['source']!='osm' or canon(r['split'])=='test' for r in rows),'OSM must stay test-only.')
    tr=[r for r in rows if canon(r['split'])=='train'];he=[r for r in rows if canon(r['split'])!='train']
    require(tr and sum(canon(r['split'])=='validation' for r in rows)==180,'Original validation180 missing.')
    for k in ('map_id','sha256','path'):
        require(len({r[k] for r in he})==len(he) and not {r[k] for r in tr}&{r[k] for r in he},'Split overlap/held-out duplicate: '+k)
    counts={'total':len(rows),'counts':dict(Counter(r['source']+'/'+canon(r['split']) for r in rows))}
    require(len(tr)==900 and len(he)==1400 and sum(r['source']=='osm' for r in he)==500, 'Expected original train900/validation180/test1220(OSM500) manifest; stop before training.')
    return dict(root=root,run=run,snapshot=snapshot,protocol=p,spec=spec,settings=settings,
        counts=counts,source_model_sha256=h,sb3_version=sb3)

def dump_tree(tree,name):
    ast.fix_missing_locations(tree);s=ast.unparse(tree)+'\n';compile(s,name,'exec');return s

def expression(s):return ast.parse(s,mode='eval').body

def replace_expr(tree,old,new,expected=1):
    """Exact AST pattern replacement; never perform global numeric replacement."""
    target=ast.dump(expression(old),include_attributes=False);nfound=0
    class Replace(ast.NodeTransformer):
        def visit(self,node):
            nonlocal nfound
            if isinstance(node,ast.expr) and ast.dump(node,include_attributes=False)==target:
                nfound+=1;return expression(new)
            return super().visit(node)
    result=Replace().visit(tree)
    require(nfound==expected,f'Expression source drift ({nfound} != {expected}): {old}')
    return result

def run_id(case):
    require(case in CASES,'Unknown four-way case.')
    return f'ppo_{case}_C3NT_COST11_FULL_H5_b0.5_g1_l0.95_budgetobsB6_fourway_seed42'


def make_plan(ctx,case,device=None):
    require(case in CASES,'Unknown mask case.')
    spec=deepcopy(ctx['spec']);spec.pop('nrg1_protocol_sha256',None)
    rid=run_id(case)
    for k in ('cost_sweep','cost_scale_ablation','beta_sweep'):spec.pop(k,None)
    spec.update(run_id=rid,configuration_id=rid.rsplit('_seed',1)[0],beta=.5,
        mask='relaxed_b6' if case=='relaxed' else case,stage='cost11_full_h5_fourway',
        reward_case='FULL_COST11_FOURWAY')
    spec['environment']['mask_mode']=case
    env=spec['environment']
    require(env['invalid_action_penalty']==1.0 and env['invalid_time_rule']=='straight_control_interval'
        and env['invalid_heading_rule']=='preserve','Original invalid-action conventions differ; refuse to guess.')
    spec['reward_definition'].update(version=VERSION,source_reward_version=BASE_VERSION,
        valid_transition_formula_unchanged=True,invalid_action_penalty=1.,
        invalid_action_extension='Original No-Mask convention: stay, heading preserved, positive straight control time, -1 invalid penalty; shared across all four cases.')
    spec['fourway']=dict(version=VERSION,case=case,label=LABELS[case],candidate_set=list(CASES),
        source_run=SOURCE_RUN,source_weights_used=False,fresh_weights=True,
        CNN='CNN3',merged_feature_dim=261,budget_scalar='b/6' if case=='relaxed' else 'constant_zero',
        cost_multiplier=11.,beta=.5,coverage_multiplier=1.,completion_reward=2.,
        execution_cap_factor=5.,normalization_step_factor=5.,gamma=1.,gae_lambda=.95,
        training_count=900,validation_count=180,full_count=1400,test_only_count=1220,
        initialization_seed=int(ctx['protocol']['request']['evaluation']['initial_seed']),
        algorithm='MaskablePPO for all; No Mask supplies four True values in train AND test',
        invalid_action_penalty=1.,no_policy_switch_at_evaluation=True)
    settings=deepcopy(ctx['settings'])
    if device is not None:settings['device']=device
    expected=deepcopy(ctx['spec']['environment']);expected['mask_mode']=case
    require(env==expected,'Unexpected environment change beyond mask_mode.')
    require(spec['external_stop']==ctx['spec']['external_stop'],'Stop/normalization changed.')
    require(spec['encoder_ablation']==ctx['spec']['encoder_ablation'],'CNN input changed.')
    ecfg=deepcopy(ctx['protocol']['request']['evaluation'])
    ecfg.update(validation='Final model; original validation180; no checkpoint selection',
        full_test='test1220(synthetic720+OSM500) + validation180; separate test-only tables')
    return spec,settings,ecfg


def transform_trainer(source):
    tree=ast.parse(source);hits=0
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module==SOURCE_CORE:
            node.module=CORE;hits+=1
    require(hits==3,'Original trainer core-import structure changed.')
    # All actual training/optimizer/recovery function AST bodies stay unchanged.
    return dump_tree(tree,TRAINER+'.py')


def transform_logger(source):
    tree=ast.parse(source);hits=0
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='C3NT_VERSION' for t in n.targets):
            require(isinstance(n.value,ast.Constant) and n.value.value==BASE_VERSION,'Unexpected logger version.')
            n.value=ast.Constant(VERSION);hits+=1
    require(hits==1,'Logger version field missing.')
    return dump_tree(tree,LOG+'.py')


def transform_preflight(source):
    tree=ast.parse(source);names=('preflight','software_check','canonical_split')
    found={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    require(set(found)==set(names),'Source preflight functions missing.')
    pre=found['preflight']
    pre=replace_expr(pre,"'nrg1_c3nt_cost5_logging'",repr(LOG))
    pre=replace_expr(pre,'audit_cost_multiplier(rt, spec, reference_core_path)','audit_fourway(rt, spec)')
    for n in ast.walk(pre):
        if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('[CHECK PASS] COST5 only;'):
            n.value='[CHECK PASS] Current FULL COST11 four-way: legal reward/invalid convention, mixed CSV, '
    prefix=ast.parse('''from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import math, importlib, numpy as np
from cost11_fourway_checks import audit_fourway

def require(ok,msg):
    if not ok:raise RuntimeError(msg)
''').body
    return dump_tree(ast.Module(body=prefix+[found[n] for n in names],type_ignores=[]),PREFLIGHT_MODULE)


def transform_evaluator(source,case):
    names=('require','sha','safe','read_json','atomic_json','atomic_csv','digest',
        'canonical_split','evaluation_summary','save_report','evaluate_phase')
    tree=ast.parse(source);found={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    require(set(found)==set(names),'Frozen full evaluator interface changed.')
    fn=found['evaluate_phase']
    fn.body[:0]=ast.parse('''
require(phase in ('validation','test'), 'Only held-out evaluation allowed.')
require(spec['environment']['mask_mode']==EXPECTED_CASE==spec['fourway']['case'], 'Wrong evaluation mask.')
require(spec['beta']==spec['environment']['tradeoff_beta']==.5, 'Wrong beta.')
require(spec['environment']['step_budget_factor']==5. and spec['reward_definition']['cost_multiplier']==11., 'Not COST11 H5.')
require(spec['reward_definition']['success_terminal']==2. and spec['reward_definition']['failure_terminal']==0., 'Wrong event reward.')
''').body
    ix=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='chosen' for t in n.targets))
    fn.body[ix+1:ix+1]=ast.parse('''
expected=180 if phase=='validation' else 1400
require(len(chosen)==expected and len({r.map_id for r in chosen})==expected,'Wrong held-out map count.')
require(all(canonical_split(r.split)!='train' for r in chosen),'Training data in evaluation.')
if phase=='test':
    require(sum(canonical_split(r.split)=='validation' for r in chosen)==180,'Missing validation180.')
    require(sum(r.source=='osm' for r in chosen)==500,'Missing OSM500.')
''').body
    loop=next(n for n in fn.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(i, rec)')
    ix=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='rows.append')
    loop.body[ix:ix]=ast.parse('''
require(row['mask_mode']==EXPECTED_CASE, 'Cached evaluation has another mask.')
require(int(row['steps'])==int(row['valid_move_count'])+int(row['invalid_action_count']), 'Attempts/movements mixed.')
require(int(row['max_steps'])==int(math.ceil(5*int(row['active_cells']))), 'Horizon changed.')
''').body
    for n in ast.walk(found['save_report']):
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            n.value=n.value.replace('Relaxed B6 CNN3 / NoClock / ExternalCap —','FULL COST11 '+LABELS[case]+' —')
    prefix=ast.parse('''from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import csv, hashlib, html, json, math, os, time, numpy as np
''').body
    prefix+=ast.parse(f'RUN_ID={run_id(case)!r}\nVERSION={VERSION!r}\nEXPECTED_CASE={case!r}').body
    return dump_tree(ast.Module(body=prefix+[found[n] for n in names],type_ignores=[]),EVAL_MODULE)


def prepare_snapshot(ctx,dst,case):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    for name,h in ctx['protocol']['snapshot_hash'].items():
        p=child(ctx['snapshot'],name);require(sha(p)==h,'Source changed while copying: '+name)
        target=dst/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    for name in PACKAGE:
        require(not (dst/name).exists(),'Package/source collision: '+name)
        shutil.copy2(Path(__file__).parent/name,dst/name)
    jobs=((SOURCE_TRAINER+'.py',TRAINER+'.py',transform_trainer),
          (SOURCE_LOG+'.py',LOG+'.py',transform_logger),
          (SOURCE_EVAL,EVAL_MODULE,lambda s:transform_evaluator(s,case)),
          ('run_nrg1_c3nt_cost5.py',PREFLIGHT_MODULE,transform_preflight))
    for old,new,fun in jobs:
        require(not (dst/new).exists(),'Generated module collision.')
        a=(dst/old).read_text(encoding='utf-8-sig');b=fun(a);(dst/new).write_text(b,encoding='utf-8')
        (dst/(new+'.diff')).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile=old,tofile=new)),encoding='utf-8')
    protected={}
    for n in (SOURCE_CORE+'.py','coverage_core.py','core_ch4.py','core_ch4_bounded_b6.py',
              'nrg1_c3nt_cost5_features.py','resume_ch4.py','full_v2_utils.py'):
        require(sha(dst/n)==ctx['protocol']['snapshot_hash'][n],'Protected file modified: '+n)
        protected[n]=sha(dst/n)
    write(dst/'fourway_source_audit.json',dict(case=case,protected_original_bytes=protected,
        new_adapter=CORE,trainer_changes='three core imports only',logger_changes='accepted version only',
        invalid_action_rule='same original -1 + straight control time; no movement/heading change',
        no_warm_start=True,no_reward_ablation=True))
    return {str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def runtime(snapshot,case):
    snap=Path(snapshot).resolve();sys.dont_write_bytecode=True
    names=(TRAINER,CORE,LOG,SOURCE_CORE,'nrg1_c3nt_cost5_features','full_v2_utils',
        'coverage_core','core_ch4','core_ch4_bounded_b6','resume_ch4','evaluation_ch4')
    for name in names:
        if name in sys.modules:require(Path(sys.modules[name].__file__).resolve().parent==snap,'Fresh process required: '+name)
    sys.path.insert(0,str(snap));importlib.invalidate_caches()
    t=importlib.import_module(TRAINER);u=importlib.import_module('full_v2_utils');e=importlib.import_module('evaluation_ch4')
    c=importlib.import_module(CORE);f=importlib.import_module('nrg1_c3nt_cost5_features');log=importlib.import_module(LOG)
    for m in (t,u,e,c,f,log):require(Path(m.__file__).resolve().parent==snap,'Runtime escaped frozen source.')
    require(c.VERSION==VERSION and c.COST_MULTIPLIER==11. and c.NORMALIZATION_STEP_FACTOR==5.,'Wrong four-way adapter.')
    require(log.C3NT_VERSION==c.VERSION,'Logger/core mismatch.');log.install(t,u)
    entry=load_file('_cost11_fourway_eval',snap/EVAL_MODULE)
    pre=load_file('_cost11_fourway_preflight',snap/PREFLIGHT_MODULE)
    require(entry.EXPECTED_CASE==case and entry.RUN_ID==run_id(case),'Wrong generated evaluator.')
    return (t,u,e,c,f,entry),pre


def run_preflight(rt,pre,spec,settings,records,tmp):
    training=[r for r in records if str(r.split).lower()=='train' and r.source=='synthetic']
    require(training,'No training maps.')
    cfg=rt[0].ExperimentEnvConfig(**spec['environment']);pairs=[]
    seed=spec['fourway']['initialization_seed']
    for rec in training[:3]:
        a=rt[2].initial_condition(rec,replace(cfg,mask_mode='relaxed'),spec['initialization'],0,seed)
        b=rt[2].initial_condition(rec,cfg,spec['initialization'],0,seed)
        require(a==b,'Mask altered physical initialization/pair ID: '+rec.map_id)
        pairs.append(dict(map_id=rec.map_id,identical=True))
    rep=pre.preflight(rt,spec,settings,training,tmp)
    rep.update(fourway=spec['fourway'],initial_condition_mask_invariance=pairs,
        optimizer_updates_executed=0,training_maps_only=True)
    return rep

def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('check','train','validation','test','all','report'),default='check')
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--root',type=Path,default=Path('chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11'),help='Completed original FULL COST11 beta0.5 H5 root.')
    p.add_argument('--out-root',type=Path,default=Path('chapter4_C3NT_COST11_FOURWAY_seed42'))
    p.add_argument('--check-out',type=Path,default=Path('cost11_fourway_check_report'))
    g=p.add_mutually_exclusive_group();g.add_argument('--device',default=None);g.add_argument('--gpus',type=int,nargs='+')
    p.add_argument('--eval-device',default='cpu')
    p.add_argument('--baseline-targets',type=Path,help='Read-only original BCD/WLG validation180 CSV/ZIP; report only.')
    p.add_argument('--worker-case',choices=CASES,help=argparse.SUPPRESS)
    a=p.parse_args(argv);require(len(set(a.cases))==len(a.cases),'Duplicate mask.')
    if a.gpus:require(all(g>=0 for g in a.gpus) and len(set(a.gpus))==len(a.gpus),'Invalid GPU list.')
    return a

def worker_run(a):
    case=a.worker_case;root=abspath(a.root);parent=abspath(a.out_root);out=parent/case
    check=abspath(a.check_out)/case;suite=Path(__file__).resolve().parent
    require(not overlap(root,parent) and parent!=suite,'Output overlaps source/suite.')
    require(not overlap(root,check) and not overlap(parent,check) and check!=suite,'Unsafe check output.')
    ctx=source_context(root);spec,settings,ecfg=make_plan(ctx,case,a.device)
    print(f'[CASE] {LABELS[case]}; FULL COST11 beta=.5; coverage=1/N, complete=+2; H5; CNN261',flush=True)
    print('[INVALID] shared original rule: stay, heading preserved, straight control time, penalty=1.0; attempts and movements separate',flush=True)
    print(f'[TRAIN] Fresh weights or SAME-case recovery, 5M; device={settings["device"]}; original CNN261; no beta/clock input',flush=True)
    print('[EVAL] --phase all: validation180 then full1400(test1220 incl.OSM500 + validation180). '+str(ctx['counts']),flush=True)
    if a.phase=='check':
        with tempfile.TemporaryDirectory(prefix=case+'_check_') as tmp:
            snap=Path(tmp)/'source_snapshot';prepare_snapshot(ctx,snap,case);rt,pre=runtime(snap,case)
            records=rt[1].load_manifest(root/'manifest.csv',50,verify_files=False)
            rep=run_preflight(rt,pre,spec,settings,records,tmp)
            with output_lock(check/'.report.lock'):
                write(check/'checks.json',rep);write(check/'requested_changes.json',dict(spec=spec,settings=settings))
                (check/'model_summary.txt').write_text(rep['model_summary'],encoding='utf-8')
                for p in snap.glob('*.py.diff'):shutil.copy2(p,check/p.name)
                shutil.copy2(snap/'fourway_source_audit.json',check/'fourway_source_audit.json')
        print(f'[CHECK PASS] {case}; no optimizer update; {check}',flush=True);return
    require(a.phase in ('train','validation','test','all'),'Unsupported worker phase.')
    out.mkdir(parents=True,exist_ok=True)
    with output_lock(out/'.runner.lock'):
        request=dict(version=VERSION,case=case,source_root=str(root),source_run=str(ctx['run']),
            source_protocol_sha256=sha(root/'protocol.json'),source_manifest_sha256=sha(root/'manifest.csv'),
            source_model_sha256=ctx['source_model_sha256'],source_snapshot_hash=ctx['protocol']['snapshot_hash'],
            package_hash={n:sha(suite/n) for n in PACKAGE},spec=spec,settings=settings,evaluation=ecfg,
            eval_device=a.eval_device,validation_only=False)
        pp=out/'protocol.json';snap=out/'source_snapshot'
        if pp.is_file():
            p=read(pp);require(p['request']==safe(request),'Frozen mask/options changed; restore original code/options or use NEW output.')
            for n,h in p['snapshot_hash'].items():require(sha(child(snap,n))==h,'Frozen source changed: '+n)
            require(sha(out/'manifest.csv')==p['manifest_sha256'],'Frozen manifest changed.')
        else:
            require(a.phase in ('train','all'),'Evaluation-only requires existing trained fourway output.')
            require(not any(p.name!='.runner.lock' for p in out.iterdir()),'Unrecognized output; preserve it and select another output.')
            hashes=prepare_snapshot(ctx,snap,case);shutil.copy2(root/'manifest.csv',out/'manifest.csv')
            write(pp,dict(request=request,snapshot_hash=hashes,manifest_sha256=sha(out/'manifest.csv'),counts=ctx['counts']))
            write(out/'config.json',dict(environment=spec['environment'],training=settings,evaluation=ecfg,
                reward_definition=spec['reward_definition'],fourway=spec['fourway']))
        spec=deepcopy(spec);spec['nrg1_protocol_sha256']=sha(pp)
        rt,pre=runtime(snap,case);t,u,e,c,f,entry=rt
        write(out/'software_at_run.json',pre.software_check())
        records=u.load_manifest(out/'manifest.csv',50,verify_files=False)
        if a.phase in ('train','all'):
            with tempfile.TemporaryDirectory(prefix=case+'_preflight_') as tmp:
                write(out/'preflight.json',run_preflight(rt,pre,spec,settings,records,tmp))
            training=[r for r in records if str(r.split).lower()=='train' and r.source=='synthetic']
            result=t.train_run(spec,settings,training,out/'runs'/spec['run_id'])
            require(result.get('state')=='completed','Training not completed; no evaluation started.')
        if a.phase in ('train','all','validation'):
            entry.evaluate_phase(out,t,u,e,spec,ecfg,records,'validation',a.eval_device)
        if a.phase in ('all','test'):
            entry.evaluate_phase(out,t,u,e,spec,ecfg,records,'test',a.eval_device)
        from cost11_fourway_report import case_review
        case_review(out)
        print(f'[DONE] {case}; {a.phase}; review={out/"fourway_review.zip"}',flush=True)

def main(argv=None):
    a=arguments(argv)
    # All four conditions use independent worker processes and model weights.
    if a.worker_case is not None:
        require(not a.gpus,'Worker uses a single device.');worker_run(a);return
    root,out,check=abspath(a.root),abspath(a.out_root),abspath(a.check_out);suite=Path(__file__).resolve().parent
    require(not overlap(root,out) and out!=suite and not overlap(root,check) and not overlap(out,check) and check!=suite,'Unsafe paths.')
    targets=abspath(a.baseline_targets) if a.baseline_targets else None
    if targets:require(out not in targets.parents and check not in targets.parents,'Baseline cannot be an output artifact.')
    from cost11_fourway_report import combined_report
    if a.phase=='report':
        require(out.is_dir(),'No output to report.')
        with output_lock(out/'.launcher.lock'):combined_report(root,out,a.cases,targets)
        return
    ctx=source_context(root)
    devices=[f'cuda:{g}' for g in a.gpus] if a.gpus else [a.device or ctx['settings']['device']]
    assignment={b:devices[i%len(devices)] for i,b in enumerate(a.cases)}
    pending=list(a.cases);active={};failures=[];lockroot=check if a.phase=='check' else out
    with output_lock(lockroot/'.launcher.lock'):
        write(lockroot/'launch_plan.json',dict(version=VERSION,phase=a.phase,cases=a.cases,
            assignments={b:dev for b,dev in assignment.items()},training_steps_per_case=5_000_000,
            cap_and_normalizer='5N',validation_count=180,full_test_count=1400,
            test_requested=a.phase in ('all','test'),original_relaxed_reference_read_only=True))
        try:
            while pending or active:
                busy={dev for b,dev in active.values()}
                for case in list(pending):
                    dev=assignment[case]
                    if dev in busy or (a.phase=='check' and active):continue
                    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker-case',case,'--phase',a.phase,
                        '--root',str(root),'--out-root',str(out),'--check-out',str(check),'--device',dev,'--eval-device',a.eval_device]
                    proc=subprocess.Popen(cmd);active[proc]=(case,dev);busy.add(dev);pending.remove(case)
                    print(f'[LAUNCH] {case} {dev} pid={proc.pid}',flush=True)
                for proc,(case,dev) in list(active.items()):
                    code=proc.poll()
                    if code is not None:
                        print(f'[EXIT] {case} code={code}',flush=True)
                        if code:failures.append((case,code))
                        del active[proc]
                if active:time.sleep(.3)
        except KeyboardInterrupt:
            import signal
            for proc in active:
                if proc.poll() is None:proc.send_signal(signal.SIGINT)
            for proc in active:proc.wait()
            raise
        if a.phase!='check':
            combined_report(root,out,a.cases,targets)
            print('[REVIEW] '+str(out/'cost11_fourway_review.zip'),flush=True)
    require(not failures,f'Failed mask runs {failures}; preserve logs/recovery and resume SAME command.')
    if a.phase=='check':print('[CHECK ALL PASS] all requested mask probes; zero optimizer updates.',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('[STOP] Preserve logs/recovery; resume SAME command/code/devices.',file=sys.stderr);raise SystemExit(130)
    except Exception as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}',file=sys.stderr);raise

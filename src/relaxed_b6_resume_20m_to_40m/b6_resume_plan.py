"""Read-only import of the user's EXISTING COST11 B6 beta=.5 20M checkpoint.

No fresh model or alternative B/beta/reward is substituted. Native B124 PPO,
physics, CNN, sampler and recovery modules are imported byte-for-byte.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib, json, os, shutil, zipfile
from pathlib import Path
import b124_engine as E
import b124_native_support as U
import b6_plan as P

VERSION='cost11_B6_existing20M_to_total40M_CP5M_VAL1M_v1_20261004'
DEFAULT_OUT='chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42'
DEFAULT_SOURCE='chapter4_COST11_5M_to_20M_seed42/relaxed_b05'
SOURCE_STEP=20_004_864
TARGET_STEP=40_001_536
TARGET=40_000_000
STAGE=2
PACKAGE=Path(__file__).resolve().parent
EVIDENCE=PACKAGE/'evidence'


def input_source(root,explicit=None):
    p=Path(explicit).expanduser() if explicit is not None else root/DEFAULT_SOURCE
    if not p.is_absolute(): p=root/p
    p=p.resolve()
    if p.name=='run_spec.json': p=p.parent
    if (p/'run_spec.json').is_file():
        U.require(p.parent.name=='runs','--source-root run must be under runs/')
        return p.parent.parent,p
    U.require((p/'continuation_protocol.json').is_file(),
        'Existing B6 beta=.5 20M input not found: '+str(p)+
        '\nUse --source-root /actual/chapter4_COST11_5M_to_20M_seed42/relaxed_b05 .'
        '\nNo fresh training, COST5, beta=0 or 5M fallback is performed.')
    expected=U.read(EVIDENCE/'existing20M_run_spec.json')['run_id']
    run=p/'runs'/expected
    if not (run/'run_spec.json').is_file():
        candidates=[]
        for f in (p/'runs').glob('*/run_spec.json'):
            sp=U.read(f)
            if sp.get('beta')==.5 and sp.get('environment',{}).get('mask_mode')=='relaxed':
                candidates.append(f.parent)
        U.require(len(candidates)==1,'Need one matching original 20M run; candidates='+str(candidates))
        run=candidates[0]
    return p,run


def _identity(sp,st,rows):
    # Exact native trainers_ch4_c3nt_cost_sweep.run_identity algorithm.
    content=dict(spec=sp,settings=st,maps=[(r['map_id'],r['sha256']) for r in rows
                                        if r['source']=='synthetic' and r['split']=='train'])
    return hashlib.sha256(json.dumps(content,sort_keys=True,default=str).encode()).hexdigest()


def inspect_source(root,base,run,b124_plan,manifest):
    """Check source metadata, final-model bytes and sampler logs before copying."""
    required=['run_spec.json','training_settings.json','status.json','checkpoints.json']
    missing=[str(run/n) for n in required if not (run/n).is_file()]
    U.require(not missing,'Original 20M metadata missing: '+', '.join(missing))
    sp,st,status,cps=[U.read(run/n) for n in required]
    expected_sp=U.read(EVIDENCE/'existing20M_run_spec.json')
    expected_proto=U.read(EVIDENCE/'existing20M_continuation_protocol.json')
    proto=U.read(base/'continuation_protocol.json')
    U.require(sp==expected_sp,'Source run_spec differs from the previously uploaded B6 beta=.5 20M run')
    U.require(st==expected_proto['request']['settings'],'Source training settings differ from original 20M settings')
    U.require(proto==expected_proto,'Source continuation protocol changed')
    U.require(status.get('state')=='completed','Original 20M source is not complete; do not copy a live source')
    U.require(int(status.get('actual_timesteps',0))==SOURCE_STEP and int(status.get('requested_timesteps',0))==20_000_000,
              'Wrong source total: only the original completed 20M is imported')
    U.require(status.get('identity')==_identity(sp,st,manifest),
              'Source identity differs: source spec/settings or TRAIN900 map order differs')
    for k in ('episodes','ppo_epoch_counter'):
        U.require(isinstance(status.get(k),int) and status[k]>0,'Missing original counter '+k)
    env=deepcopy(b124_plan['template_spec']['environment']);env.update(mask_mode='relaxed',detour_budget_max=6)
    U.require(sp['environment']==env and sp['beta']==.5 and sp['seed']==42,'Source B/beta/physical config differs')
    U.require(proto['snapshot_hash']==b124_plan['runtime']['relaxed']['hashes'],
              'Original 20M and verified B124 runtime differ (reward/CNN/core/trainer)')
    ppo=deepcopy(b124_plan['template_settings']['ppo']);ppo.update(learning_rate_initial=5e-5,learning_rate_floor=5e-5)
    U.require(st['ppo']==ppo,'PPO hyperparameters differ from B124 continuation')
    for key in ('num_envs','map_sequence_seed','torch_threads','resume_every_timesteps','resume_keep_generations'):
        U.require(st[key]==b124_plan['template_settings'][key],'Sampler/recovery setting differs: '+key)
    # Manifest hash verifies original membership; compare each metadata field, not only count.
    candidates=[base/'manifest.csv',root/'chapter4_relaxedB6_C3NT_COST_SWEEP_seed42/cost11/manifest.csv']
    mf=next((f for f in candidates if f.is_file() and U.sha(f)==proto['source_manifest_sha256']),None)
    U.require(mf is not None,'Original 20M source manifest missing/changed. Tried '+str(candidates))
    mm=U.rows(mf)
    keys=('map_id','sha256','source','split','tier','pattern')
    U.require([tuple(r[k] for k in keys) for r in mm]==[tuple(r[k] for k in keys) for r in manifest],
              'Original20M and B124 map order/metadata are not identical')
    final=[c for c in cps if c.get('final')]
    U.require(len(final)==1 and int(final[0]['num_timesteps'])==SOURCE_STEP,'No unique registered 20M final model')
    receipt=U.read(EVIDENCE/'existing20M_receipt.json')
    U.require(final[0]['sha256']==receipt['source_final_sha256'],'Original final checkpoint registration changed')
    cp=deepcopy(final[0]);model=run/'last_model.zip'
    if not model.is_file():
        alternatives=[c for c in cps if not c.get('source_baseline') and not c.get('final')
                      and int(c.get('num_timesteps',0))==SOURCE_STEP]
        U.require(len(alternatives)==1,'20M final missing and no same-step registered checkpoint')
        cp=deepcopy(alternatives[0]);model=run/'checkpoints'/Path(cp['path']).name
        U.require(cp['sha256']==receipt['source_step20M_sha256'],'Alternate20M checkpoint registration differs')
    U.require(model.is_file() and not model.is_symlink() and U.sha(model)==cp['sha256'],
              'Original20M checkpoint missing/changed: '+str(model))
    # Read JSON only. Do not deserialize the checkpoint here.
    with zipfile.ZipFile(model) as z:
        for name in ('data','policy.pth','policy.optimizer.pth'):
            U.require(name in z.namelist() and z.getinfo(name).file_size>0,'Missing saved policy/Adam/data: '+name)
        data=json.loads(z.read('data'))
    U.require(int(data['num_timesteps'])==SOURCE_STEP and int(data['_n_updates'])==status['ppo_epoch_counter'],
              'Model counters differ from original completed status')
    U.require(data.get('seed')==42 and data.get('n_envs')==8,'Saved model seed/environment count differs')
    protected={str(base/'continuation_protocol.json'):U.sha(base/'continuation_protocol.json'),str(mf):U.sha(mf)}
    for name in required: protected[str(run/name)]=U.sha(run/name)
    logs=[]
    for rank in range(8):
        f=run/f'map_order_rank{rank}.csv'
        U.require(f.is_file() and not f.is_symlink(),'Missing original sampler log: '+str(f))
        rr=U.rows(f);U.require(bool(rr) and None not in rr[-1] and all(v is not None for v in rr[-1].values()),
                             'Truncated original sampler log: '+str(f))
        pos=[int(r['sequence_position']) for r in rr]
        U.require(pos==list(range(pos[0],pos[-1]+1)) and pos[0]>=1,'Non-contiguous original sampler log')
        U.require(all(int(r['rank'])==rank and int(r['episode'])==n for r,n in zip(rr,pos)),
                  'Sampler rank/episode positions differ')
        protected[str(f)]=U.sha(f);logs.append(f)
    software=run/'software.json'
    if software.is_file():
        d=U.read(software)
        for key in ('torch','numpy','gymnasium','stable-baselines3','sb3-contrib'):
            U.require(d.get(key)==b124_plan['software'].get(key),'Source software mismatch: '+key)
        protected[str(software)]=U.sha(software)
    protected[str(model)]=cp['sha256'];cp['path']=str(model)
    return dict(base=base,run=run,spec=sp,settings=st,status=status,checkpoint=cp,
                protected=protected,logs=logs,manifest=mf)


def _copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    h=U.sha(src)
    if dst.is_file():
        U.require(not dst.is_symlink() and U.sha(dst)==h,'Incomplete output contains changed file: '+str(dst))
        return
    U.require(not dst.exists(),'Output copy conflicts: '+str(dst))
    tmp=dst.with_name(dst.name+f'.writing.{os.getpid()}')
    try:
        shutil.copyfile(src,tmp);U.require(U.sha(tmp)==h,'Input changed while copying: '+str(src));os.replace(tmp,dst)
    finally:
        if tmp.exists():tmp.unlink()


def make_plan(root,out,gpu=None,source_root=None,b124_root=None):
    root,out=Path(root).resolve(),Path(out).resolve();codes=P.package_hashes()
    U.require(root.is_dir() and out!=root and out not in root.parents and not P.paths_overlap(out,PACKAGE),'Unsafe output')
    if (out/'plan.json').is_file():
        plan=U.read(out/'plan.json')
        U.require(plan.get('version')==VERSION and plan.get('package_hashes')==codes,'Different code/experiment; do not mix output')
        U.require(plan['out']==str(out) and plan['root']==str(root),'Resume path changed')
        if gpu is not None:U.require(plan['gpus']==[gpu],'Resume with the same recorded --gpu')
        if source_root is not None:
            b,r=input_source(root,source_root);U.require(str(r)==plan['existing_source']['original_run'],'Source changed on resume')
        if b124_root is not None:U.require(str(P.source_root(root,b124_root))==plan['b124_source_root'],'B124 reference changed')
        E.verify_plan(plan);return plan
    b124=P.source_root(root,b124_root)
    base,run=input_source(root,source_root)
    for p in (b124,base):U.require(not P.paths_overlap(out,p),'Output overlaps existing experiment: '+str(p))
    allowed={'.launcher.lock','_runtime','manifest.csv','reference_test_baselines.csv',
             'reference_validation_baselines.csv','source20M','initializing.json','audit'}
    if out.exists():U.require(not set(p.name for p in out.iterdir())-allowed,'Nonempty output from another run; choose new --out')
    template,mm,protected,baselines=P.inspect_inputs(root,b124)
    src=inspect_source(root,base,run,template,mm)
    gpu=0 if gpu is None else gpu;U.require(isinstance(gpu,int) and gpu>=0,'--gpu must be a nonnegative index')
    init=dict(version=VERSION,source_run=str(run),source_checkpoint_sha256=src['checkpoint']['sha256'],
              b124_root=str(b124),gpu=gpu,package_hashes=codes)
    U.freeze(out/'initializing.json',init)
    runtime={}
    for family,info in template['runtime'].items():
        dest=out/'_runtime'/family
        for name,h in info['hashes'].items():_copy(b124/'_runtime'/family/name,dest/name)
        runtime[family]=dict(path=str(dest),hashes=deepcopy(info['hashes']),version=info['version'])
    U.csv_write(out/'manifest.csv',mm)
    for name,f in baselines.items():_copy(f,out/name)
    imported=out/'source20M'; imported.mkdir(exist_ok=True)
    _copy(src['checkpoint']['path'],imported/'last_model.zip')
    for n in ('run_spec.json','training_settings.json','status.json','checkpoints.json'):_copy(run/n,imported/n)
    _copy(base/'continuation_protocol.json',imported/'continuation_protocol.json')
    for f in src['logs']:_copy(f,imported/f.name)
    original_guards={**protected,**src['protected']}
    for f in imported.iterdir():
        if f.is_file():protected[str(f)]=U.sha(f)
    # Protect immutable input metadata too; only snapshot checkpoint bytes are re-hashed during runtime.
    for name,h in original_guards.items():
        if not name.endswith('.zip'):protected[name]=h
    plan=deepcopy(template)
    plan.update(version=VERSION,root=str(root),out=str(out),Bs=[6],gpus=[gpu],runtime=runtime,
                manifest=str(out/'manifest.csv'),manifest_sha256=U.sha(out/'manifest.csv'),
                protected=protected,package_hashes=codes,b124_source_root=str(b124),active_stages=[STAGE],
                source_actual_step=SOURCE_STEP,source_nominal_m=20,target_total=TARGET,
                expected_validation_milestones=list(range(21,41)),
                stage_targets=[5_000_000,20_000_000,TARGET],raw_only=True,connection_refinement=False,
                existing_source=dict(original_base=str(base),original_run=str(run),snapshot_run=str(imported),
                    checkpoint_path=str(imported/'last_model.zip'),checkpoint_sha256=src['checkpoint']['sha256'],
                    actual_step=SOURCE_STEP,episodes=src['status']['episodes'],epoch_counter=src['status']['ppo_epoch_counter'],
                    source_training_identity=src['status']['identity']),
                initial_training='existing completed B6 beta=.5 20M policy+Adam; NO fresh 0M or B124 weight transfer',
                segment_boundaries='original0->5->20M retained; new20->40M. Policy/Adam/sampler progress continued; partial source episodes and global RNG restarted at the boundary. Later recovery is native full-state.',
                comparison_note='B1/2/4/6 have equal 40M budget and configured PPO settings, but B6 retains its own existing training history; not a re-created identical fresh run.',
                old_validation_note='Historical1M-20M results are kept in source outputs and are not re-evaluated. New validation starts21M.',
                source_file_hashes=original_guards)
    U.write(out/'audit/existing20M_import.json',dict(
        source_run=str(run),source_step=SOURCE_STEP,selected_checkpoint=src['checkpoint'],
        source_training_identity=src['status']['identity'],requested_total=TARGET,
        additional_nominal_steps=TARGET-SOURCE_STEP,final_update_step=TARGET_STEP,
        additional_optimizer_update_blocks=(TARGET_STEP-SOURCE_STEP)//E.BLOCK,
        source_weights_used=True,fresh_weights=False,settings_and_runtime_compatible=True,
        past_results_not_deleted=True,original_files_sha256=original_guards))
    U.write(out/'plan.json',plan);E.verify_plan(plan);return plan


def segment_paths(plan,B,stage):
    U.require(B==6,'This executable trains B6 only')
    name='continue_existing20M_to40M' if stage==STAGE else f'NOT_RUN_stage{stage}'
    base=Path(plan['out'])/'B6'/name
    return base,base/'runs'/'ppo_COST11_FULL_B6_beta05_seed42_existing20M_to40M'


def segment_spec(plan,B,stage):
    U.require(B==6 and stage==STAGE,'No fresh training or 5M stage is permitted')
    base,run=segment_paths(plan,B,stage);e=plan['existing_source']
    sp=U.read(Path(e['snapshot_run'])/'run_spec.json');st=U.read(Path(e['snapshot_run'])/'training_settings.json')
    for k in ('continuation','continuation_protocol_sha256','source_protocol_sha256','nrg1_protocol_sha256'):
        sp.pop(k,None);st.pop(k,None)
    sp.update(run_id=run.name,configuration_id=run.name,stage='existing_B6_20M_to40M_continuation',fresh_weights=False,source_weights_used=True)
    st.update(total_timesteps=TARGET,checkpoint_every_timesteps=5_000_000,device='cuda:0',retain_completed_resume=True)
    st['ppo']['learning_rate_initial']=st['ppo']['learning_rate_floor']=5e-5
    entry=dict(B=6,beta=.5,stage_index=STAGE,target=TARGET,source_model=e['checkpoint_path'],source_sha256=e['checkpoint_sha256'],
               source_step=e['actual_step'],source_run=e['snapshot_run'],source_episodes=e['episodes'],source_epoch_counter=e['epoch_counter'],
               initial_source='existing_original_B6_beta05_20M_policy_and_Adam',
               boundary_recovery='sampler advanced; unfinished episodes/global RNG restarted; subsequent native full-state recovery')
    sp['b124_protocol']=deepcopy(entry);st['b124_segment']=entry
    return sp,st


_NATIVE_EXTRACT=E.extract_train

def extract_train(source,continued):
    U.require(continued,'Fresh model creation is forbidden in the continuation-only package')
    text=_NATIVE_EXTRACT(source,True)
    before='            _b124_start(self.model, spec, settings, out, checkpoints, save_resume)\n'
    after=('            if int(self.model.num_timesteps) > int(settings["b124_segment"]["source_step"]):\n'
           '                _b124_start(self.model, spec, settings, out, checkpoints, save_resume)\n')
    U.require(text.count(before)==1,'Native start-hook shape changed')
    text=text.replace(before,after,1);compile(text,'continued20M_start_guard.py','exec')
    return text

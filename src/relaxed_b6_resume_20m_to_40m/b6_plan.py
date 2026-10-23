"""B6 plan built from the actual, frozen B124 plan supplied by the user.

This module changes orchestration, output paths and B, not PPO or the environment.
The imported b124_engine modules are kept byte-identical to the uploaded version.
"""
from __future__ import annotations
import hashlib, json, shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
import b124_engine as E
import b124_native_support as U

VERSION = 'cost11_B6_same_actual_B124_fresh40M_CP5M_VAL1M_v1'
DEFAULT_SOURCE = 'chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42'
DEFAULT_OUT = 'chapter4_COST11_B6_beta05_40M_CP5M_VAL1M_native_seed42'
PACKAGE = Path(__file__).resolve().parent
EVIDENCE = PACKAGE / 'evidence'


def evidence():
    receipt = U.read(EVIDENCE / 'source_provenance.json')
    U.require(U.sha(EVIDENCE / 'uploaded_B124_plan.json') == receipt['original_plan_sha256'],
              'Bundled original plan changed')
    for name, expected in receipt['unchanged_modules'].items():
        U.require(U.sha(PACKAGE/name) == expected, 'Original B124 module changed: '+name)
    return U.read(EVIDENCE/'uploaded_B124_plan.json'), receipt


def package_hashes():
    files = list(PACKAGE.glob('*.py')) + list(EVIDENCE.glob('*.json'))
    return {str(f.relative_to(PACKAGE)): U.sha(f) for f in sorted(files)}


def paths_overlap(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    return a == b or a in b.parents or b in a.parents


def source_root(root, explicit=None):
    p = Path(explicit).expanduser() if explicit is not None else root/DEFAULT_SOURCE
    if not p.is_absolute():
        p = root/p
    p = p.resolve()
    U.require((p/'plan.json').is_file(),
        'Actual B124 plan not found: '+str(p/'plan.json')+
        '\nUse --b124-root /actual/B124/output (the folder containing plan.json, manifest.csv, _runtime/).')
    return p


def verify_source_plan(source):
    original, receipt = evidence()
    plan = U.read(Path(source)/'plan.json')
    # The collected plan is immutable. Do not borrow a similarly named experiment.
    U.require(plan == original,
              'B124 plan differs from the uploaded final native plan. No fallback to another run.')
    U.require(plan['Bs'] == [1,2,4] and plan['beta'] == .5 and plan['seed'] == 42,
              'Wrong B124 control')
    U.require(plan['stage_targets'] == [5_000_000,20_000_000,40_000_000] and
              plan['checkpoint_interval'] == 5_000_000 and plan['validation_interval'] == 1_000_000,
              'B124 schedule changed')
    U.validate_spec(plan['template_spec'],plan['template_settings'],.5)
    return plan


def map_path(row, root):
    old = Path(row['path']).expanduser()
    candidates = [old]
    token = '/uav_chapter4_suite_v2/'
    if token in str(old):
        candidates.append(root/str(old).split(token,1)[1])
    mid = Path(row['map_id'])
    U.require(not mid.is_absolute() and '..' not in mid.parts,'Unsafe relative map_id')
    candidates += [root.parent/mid,root/mid]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError('Map missing; no map substitution: '+row['map_id']+'\nTried: '+str(candidates))


def inspect_inputs(root, source):
    """Read-only inspection before creating any training output."""
    sp = verify_source_plan(source)
    protected = {str(source/'plan.json'): U.sha(source/'plan.json')}
    mf = source/'manifest.csv'
    U.require(mf.is_file() and U.sha(mf)==sp['manifest_sha256'],'B124 manifest missing/changed: '+str(mf))
    mm = U.rows(mf)
    U.require(Counter((r['source'],r['split']) for r in mm)==
        {('synthetic','train'):900,('synthetic','validation'):180,('synthetic','test'):720,('osm','test'):500},
        'Wrong source partition. Expected 900/180/720/500.')
    for key in ('map_id','sha256'):
        U.require(len({r[key] for r in mm})==2300,'Duplicate or cross-split map '+key)
    train = [(r['map_id'],r['sha256']) for r in mm if r['source']=='synthetic' and r['split']=='train']
    content = dict(spec=sp['template_spec'],settings=sp['template_settings'],maps=train)
    identity = hashlib.sha256(json.dumps(content,sort_keys=True,default=str).encode()).hexdigest()
    U.require(identity==sp['template_training_identity'],'Training manifest order/identity changed')
    for r in mm:
        r['path']=str(map_path(r,root))
    protected[str(mf)]=U.sha(mf)
    for family in ('strict','relaxed'):
        for name,h in sp['runtime'][family]['hashes'].items():
            f=source/'_runtime'/family/name
            U.require(f.is_file() and U.sha(f)==h,'Native snapshot missing/changed: '+str(f))
    baselines={}
    for phase in ('validation','test'):
        name=f'reference_{phase}_baselines.csv';f=source/name
        U.require(f.is_file() and U.sha(f)==sp['baseline_hashes'][name],
                  'Original numerical baseline missing/changed: '+str(f))
        rr=U.rows(f);methods={r['method'] for r in rr}
        U.require(methods=={'bcd','wlg','ppo_b0_beta05_40m'},'Unexpected original reference methods')
        wanted={r['map_id'] for r in mm if r['split']==phase}
        for method in methods:
            sub=[r for r in rr if r['method']==method]
            U.require(len(sub)==len(wanted) and {r['map_id'] for r in sub}==wanted,
                      'Reference map membership differs: '+phase+'/'+method)
            U.require(all(U.truth(r['is_success']) for r in sub),'Failed original fixed reference')
        baselines[name]=f;protected[str(f)]=U.sha(f)
    return sp,mm,protected,baselines


def make_plan(root, out, gpu=None, b124_root=None):
    root,out=Path(root).resolve(),Path(out).resolve()
    U.require(root.is_dir(),'Suite root missing')
    U.require(out!=root and out not in root.parents,'Output must be a separate child/sibling, not suite or ancestor')
    U.require(not paths_overlap(out,PACKAGE),'Output overlaps executable package')
    code=package_hashes()
    if (out/'plan.json').is_file():
        p=U.read(out/'plan.json')
        U.require(p['version']==VERSION and p['package_hashes']==code,'Existing output belongs to other code. Use a new --out.')
        U.require(p['root']==str(root) and p['out']==str(out) and p['Bs']==[6], 'Resume root/output/B changed')
        if gpu is not None:
            U.require(p['gpus']==[gpu],'GPU selection changed. Re-run with the same --gpu as this plan.')
        if b124_root is not None:
            U.require(str(source_root(root,b124_root))==p['b124_source_root'],'B124 source changed on resume')
        E.verify_plan(p)
        return p
    source=source_root(root,b124_root)
    U.require(not paths_overlap(out,source),'B6 output must not overlap B1/B2/B4 input')
    U.require(not out.is_symlink(),'Symlink output refused')
    if out.exists():
        U.require(not any(f.name not in ('.launcher.lock',) for f in out.iterdir()),
                  'Nonempty output without plan. Use a new output; no files deleted.')
    source_plan,mm,protected,baselines=inspect_inputs(root,source)
    gpu=0 if gpu is None else gpu
    U.require(isinstance(gpu,int) and gpu>=0,'GPU index must be nonnegative')
    out.mkdir(parents=True,exist_ok=True)
    runtime={}
    for family,info in source_plan['runtime'].items():
        dest=out/'_runtime'/family;dest.mkdir(parents=True,exist_ok=True)
        for name,h in info['hashes'].items():
            shutil.copyfile(source/'_runtime'/family/name,dest/name)
            U.require(U.sha(dest/name)==h,'Snapshot copy failed')
        runtime[family]=dict(path=str(dest),hashes=deepcopy(info['hashes']),version=info['version'])
    U.csv_write(out/'manifest.csv',mm)
    for name,f in baselines.items():
        shutil.copyfile(f,out/name)
    # Match the actual initial parameter digest if its immutable receipt is present.
    initial=source/'audit/shared_initial_policy_seed42.json'
    initial_match='same architecture/seed/settings; source parameter receipt not available'
    if initial.is_file():
        record=U.read(initial)
        U.require(record.get('seed')==42 and record.get('source_weights_used') is False and
                  isinstance(record.get('tensor_sha256'),str) and len(record['tensor_sha256'])==64,
                  'Invalid B124 initial parameter receipt')
        U.freeze(out/'audit/shared_initial_policy_seed42.json',record)
        protected[str(initial)]=U.sha(initial)
        initial_match='initial parameter digest is frozen to actual B124 receipt before the first optimizer update'
    plan=deepcopy(source_plan)
    plan.update(version=VERSION,root=str(root),out=str(out),gpus=[gpu],Bs=[6],runtime=runtime,
        manifest=str(out/'manifest.csv'),manifest_sha256=U.sha(out/'manifest.csv'),
        protected=protected,package_hashes=code,b124_source_root=str(source),
        source_plan_sha256=U.sha(source/'plan.json'),source_plan_version=source_plan['version'],
        initial_training='fresh B6 weights at seed42; same 0->5->20->40M B124 schedule; no old B6/Strict weights',
        initial_parameter_crosscheck=initial_match,
        source_transfer='only immutable settings/runtime/manifest/numerical references copied; no B124 weights loaded')
    U.write(out/'plan.json',plan)
    U.write(out/'audit/B6_vs_uploaded_B124.json',dict(
        source=str(source),source_archive='B124_FOR_B6_SOURCE_20261003_162615_a78f3ba6.zip',
        actual_budget_changes=[1,2,4,6],B6=6,beta=.5,seed=42,fresh=True,
        same_template_spec=plan['template_spec']==source_plan['template_spec'],
        same_template_settings=plan['template_settings']==source_plan['template_settings'],
        same_stage_targets=plan['stage_targets']==source_plan['stage_targets'],
        same_software=plan['software']==source_plan['software'],
        checkpoint_interval=plan['checkpoint_interval'],validation_interval=plan['validation_interval'],
        raw_only=True,initial_parameter_crosscheck=initial_match,
        optimizer_and_physics_modules='copied byte-identically from actual B124',
        training_executed_by_plan_creation=False))
    E.verify_plan(plan)
    return plan

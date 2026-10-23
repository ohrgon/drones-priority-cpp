"""Read-only authentication and atomic result storage for the fixed B6 40M run."""
from __future__ import annotations
import csv, gzip, hashlib, io, json, math, os, sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
import b124_native_support as U
import ch5_common as C

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent
VERSION = 'B6_COST11_beta05_existing20M_to40M_common_refinement_v1_20261005'
STEP = 40_001_536
BUDGETS = (6,)
TRAIN_DEFAULT = 'chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42'
OUT_DEFAULT = 'refinement_b6_common'
FEATURE = 'nrg1_c3nt_cost5_features.py'
need, read, sha, safe, digest = U.require, U.read, U.sha, U.safe, U.digest
write, freeze, csvwrite, rows, lock = U.write, U.freeze, U.csv_write, U.rows, U.lock
flag, number = U.truth, U.number
near = C.near


def at(root, p):
    p = Path(p).expanduser()
    return (p if p.is_absolute() else Path(root)/p).resolve()


def relpaths(p, root, oldroot):
    p, root, oldroot = Path(p), Path(root), Path(oldroot)
    result = []
    try: result.append(root/p.relative_to(oldroot))
    except ValueError: pass
    token = '/uav_chapter4_suite_v2/'
    if token in str(p): result.append(root/str(p).split(token, 1)[1])
    if not p.is_absolute(): result.append(root/p)
    result.append(p)
    return list(dict.fromkeys(x.resolve() for x in result))


def protected_add(protected, p):
    p = Path(p).resolve()
    need(p.is_file(), 'Required input missing: '+str(p))
    h = sha(p)
    need(str(p) not in protected or protected[str(p)] == h, 'Input changed during inspection: '+str(p))
    protected[str(p)] = h
    return h


def code_hashes():
    paths = list(PACKAGE.glob('*.py')) + list((PACKAGE/'ch5_vendor').glob('*.py')) + list((PACKAGE/'native_original').glob('*.py'))
    paths += [p for p in (PACKAGE/'reference').iterdir() if p.is_file()]
    return {str(p.relative_to(PACKAGE)):sha(p) for p in sorted(paths)}


def gzrows(path):
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def domain(r):
    if r['source'] == 'synthetic' and r['split'] == 'validation': return 'Validation180'
    if r['source'] == 'synthetic' and r['split'] == 'test': return 'Synthetic720'
    if r['source'] == 'osm' and r['split'] == 'test': return 'OSM500'
    raise RuntimeError('Unsupported evaluation partition: '+str((r.get('source'),r.get('split'))))


def validation_identity(r):
    # The original native initializer's pair ID; map SHA, not file name, controls the seed.
    data = [r['map_sha256'], 'priority_mass', 0, 20260907,
            [int(r['start_r']), int(r['start_c'])], int(r['start_heading'])]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_expected(r, B, target):
    need(int(r['effective_B']) == B and number(r['beta']) == .5, 'Wrong B/beta in Raw source')
    need(int(r['checkpoint_step']) == STEP and int(r['nominal_m']) == 40, 'Not the requested 40M Raw source')
    need(r['checkpoint_sha256'] == target['sha256'], 'Raw source checkpoint hash mismatch')
    need(r['model'] == f'B{B}' and r['mask_mode'] == 'relaxed', 'Not the requested B6 experiment')
    need(r['nrg_version'] == 'nrg1_cnn3_noclock_externalcap_cost11_sweep_v1_20260926', 'Wrong native Raw version')
    need(number(r['c3nt_cost_multiplier']) == 11 and number(r['c3nt_normalization_step_factor']) == 5, 'Wrong COST11 normalization')
    need(int(r['detour_budget_max']) == B, 'Wrong Raw budget capacity')
    need(int(r['max_steps']) == math.ceil(5*int(r['active_cells'])), 'Wrong Raw horizon')
    need(flag(r['raw_only']) and not flag(r['connection_refinement']), 'Reference must be original Stage 1 Raw')
    need(flag(r['deterministic']) and r['initial_pair_id'] == validation_identity(r), 'Initial-condition identity mismatch')
    need(int(r['invalid_action_count']) == 0, 'Invalid original movement')
    near(r['time_straight_sec'], 16/11.462, 'Straight time')
    near(r['time_turn_sec'], 16/5.67, 'Turn time')
    near(r[C.T], int(r['move_straight_count'])*float(r['time_straight_sec'])+
         int(r['turn_count'])*float(r['time_turn_sec']), 'Raw time accounting')
    need(int(r['steps']) == int(r['move_straight_count'])+int(r['turn_count']), 'Move count')
    if flag(r['is_success']):
        need(number(r[C.LW]) is not None and 0 <= float(r[C.LW]) <= float(r[C.T])+2e-6, 'Missing successful Lw')
        near(r['coverage_pct'], 100, 'Successful coverage')
    else:
        need(number(r[C.LW]) is None and r['termination_reason'] == 'external_step_cap' and
             int(r['steps']) == int(r['max_steps']), 'Failed Raw must retain cap and undefined Lw')
    domain(r)


def reference_data():
    targets = read(PACKAGE/'reference/checkpoint_targets.json')
    rr = gzrows(PACKAGE/'reference/uploaded_B6_final40M_raw.csv.gz')
    grouped = {}
    for B in BUDGETS:
        rs = [r for r in rr if r['model'] == f'B{B}']
        need(len(rs) == 1400 and len({r['map_id'] for r in rs}) == 1400, 'Reference must contain 1400 unique maps per B')
        need(Counter(domain(r) for r in rs) == {'Validation180':180, 'Synthetic720':720, 'OSM500':500}, 'Reference partitions changed')
        for r in rs: validate_expected(r, B, targets[f'B{B}'])
        grouped[f'B{B}'] = {r['map_id']:r for r in rs}
    return targets, grouped


def compare_metrics(r, old):
    for k in ('map_id','map_sha256','initial_pair_id','checkpoint_sha256','source','split','pattern','tier'):
        need(str(r[k]) == str(old[k]), 'Raw identity differs: '+k)
    for k in ('effective_B','beta','checkpoint_step','nominal_m','start_r','start_c','start_heading',
              C.T,C.LW,'steps','max_steps','active_cells','coverage_pct','turn_count','move_straight_count',
              'revisit_steps','zero_transit_steps','path_length_m','time_straight_sec','time_turn_sec',
              'c3nt_cost_multiplier','nrg_detour_steps','nrg_detour_ratio','total_priority_mass',
              'covered_priority_ratio','priority_arrival_25_sec','priority_arrival_50_sec','priority_arrival_75_sec','priority_arrival_90_sec'):
        near(r.get(k), old.get(k), 'Same-checkpoint Raw reproduction '+k)
    for k in ('is_success','c3nt_terminated','c3nt_truncated'):
        need(flag(r[k]) == flag(old[k]), 'Raw termination/success changed: '+k)
    for k in ('termination_reason','c3nt_stop_kind'):
        need(r[k] == old[k], 'Raw stop kind changed: '+k)
    return dict(status='same_40M_Raw_matches_uploaded_original', compared=True,
                absolute_T_difference_sec=abs(float(r[C.T])-float(old[C.T])),
                absolute_Lw_difference_sec=abs(float(r[C.LW])-float(old[C.LW])) if flag(r['is_success']) else None)


BASE_REL = 'B6/continue_existing20M_to40M'
RUN_ID = 'ppo_COST11_FULL_B6_beta05_seed42_existing20M_to40M'


def expected_stage(plan):
    """Exactly reconstruct the source runner's metadata operation, not training."""
    sp=deepcopy(read(PACKAGE/'reference/existing20M_run_spec.json'))
    st=deepcopy(read(PACKAGE/'reference/existing20M_training_settings.json'))
    for k in ('continuation','continuation_protocol_sha256','source_protocol_sha256','nrg1_protocol_sha256'):
        sp.pop(k,None);st.pop(k,None)
    sp.update(run_id=RUN_ID,configuration_id=RUN_ID,stage='existing_B6_20M_to40M_continuation',fresh_weights=False,source_weights_used=True)
    st.update(total_timesteps=40_000_000,checkpoint_every_timesteps=5_000_000,device='cuda:0',retain_completed_resume=True)
    st['ppo']['learning_rate_initial']=st['ppo']['learning_rate_floor']=5e-5
    e=plan['existing_source']
    entry=dict(B=6,beta=.5,stage_index=2,target=40_000_000,source_model=e['checkpoint_path'],source_sha256=e['checkpoint_sha256'],
       source_step=e['actual_step'],source_run=e['snapshot_run'],source_episodes=e['episodes'],source_epoch_counter=e['epoch_counter'],
       initial_source='existing_original_B6_beta05_20M_policy_and_Adam',
       boundary_recovery='sampler advanced; unfinished episodes/global RNG restarted; subsequent native full-state recovery')
    sp['b124_protocol']=deepcopy(entry);st['b124_segment']=entry
    return sp,st


def validate_spec(sp,st,plan):
    es,et=expected_stage(plan)
    # Paths in the frozen source are permitted to retain the old mount location;
    # everything else, including the source-model hash, must match.
    for actual,want,key in ((sp,es,'b124_protocol'),(st,et,'b124_segment')):
        actual=deepcopy(actual);want=deepcopy(want)
        for obj in (actual,want):
            for field in ('source_model','source_run'):obj.get(key,{}).pop(field,None)
        need(actual==want,'Final B6 metadata differs from its actual existing20M continuation: '+key)
    # The imported B6 run has no encoder_ablation field. Do NOT fabricate one:
    # original feature-module SHA and model audit_cnn3_model validate the encoder.


def _under_training(value,training,old_training):
    p=Path(value)
    try:return (training/p.relative_to(old_training)).resolve()
    except ValueError:return p.resolve()


def _plan_matches(plan,original):
    for k in ('version','Bs','beta','seed','active_stages','stage_targets','target','target_total','source_actual_step',
       'source_nominal_m','optimizer_block','initial_seed','evaluation_seed','manifest_sha256','template_spec','template_settings',
       'software','checkpoint_interval','validation_interval','raw_only','connection_refinement'):
        need(plan.get(k)==original.get(k),'Source is not the uploaded B6 existing20M-to40M run: '+k)
    for k in ('version','hashes'):
        need(plan['runtime']['relaxed'][k]==original['runtime']['relaxed'][k],'B6 native runtime differs: '+k)
    for k in ('actual_step','checkpoint_sha256','episodes','epoch_counter','source_training_identity'):
        need(plan['existing_source'][k]==original['existing_source'][k],'Source 20M lineage changed: '+k)


def check_old_eval(training,scope,target,expected,protected):
    d=training/scope/'B6'/f'step_{STEP}'
    if not (d/'complete.json').is_file():
        return dict(path=str(d),status='using_bundled_uploaded_exact40M_Raw_metrics',native_complete_found=False)
    co=read(d/'complete.json');protected_add(protected,d/'complete.json')
    need(co['checkpoint_sha256']==target['sha256'] and int(co['checkpoint_step'])==STEP and co['scope']==scope,
         'Original final40M evaluation receipt mismatch')
    for n in ('per_map.csv','identity.json','summary.json'):
        need(n in co['files'] and (d/n).is_file() and sha(d/n)==co['files'][n],'Corrupt original evaluation file: '+str(d/n))
        protected_add(protected,d/n)
    ident=read(d/'identity.json');ip=ident['parameters']
    need(digest(ip)==ident['sha256']==co['identity'],'Original evaluation identity changed')
    need(ip['B']==6 and ip['checkpoint_sha256']==target['sha256'] and int(ip['checkpoint_step'])==STEP,
         'Original evaluation B/checkpoint differs')
    need(ip['initial_seed']==20260907 and ip['evaluation_seed']==42 and flag(ip['deterministic']), 'Different original evaluation seeds')
    wanted={k:v for k,v in expected.items() if (v['split']=='validation')==(scope=='validation')}
    rs=rows(d/'per_map.csv')
    need(len(rs)==len(wanted) and {r['map_id'] for r in rs}==set(wanted),'Original evaluation map set differs')
    need(set(tuple(x) for x in ip['records'])=={(r['map_id'],r['map_sha256']) for r in wanted.values()},'Original identity map hashes differ')
    for r in rs:compare_metrics(r,wanted[r['map_id']])
    return dict(path=str(d),status='native_receipt_matches_uploaded_Raw',native_complete_found=True,n_maps=len(rs),
                parameter_sha256=co.get('parameter_sha256'))


def source_context(root,training_root=None,scope='all'):
    root=Path(root).resolve();original=read(PACKAGE/'reference/uploaded_B6_plan.json')
    targets,expected_all=reference_data();target=targets['B6'];expected=expected_all['B6']
    candidates=[at(root,training_root)] if training_root else [root/TRAIN_DEFAULT]
    if not training_root:
        candidates += [p.parent for p in root.glob('*B6*EXISTING20M*/plan.json') if p.parent not in candidates]
    found=[];rejected=[]
    for base in candidates:
        if not (base/'plan.json').is_file():continue
        try:
            plan=read(base/'plan.json');_plan_matches(plan,original);found.append((base,plan))
        except Exception as e:rejected.append(dict(path=str(base),error=str(e)))
    need(len(found)==1,'Need the completed original B6 existing20M-to40M folder, not fresh B6, best/38M or 5M. '
         'Use --training-root PATH. Candidates='+str(candidates)+'; rejected='+str(rejected))
    training,plan=found[0];protected={};protected_add(protected,training/'plan.json');oldroot=Path(plan['root'])
    base=training/BASE_REL;run=base/'runs'/RUN_ID
    sp=read(run/'run_spec.json');st=read(run/'training_settings.json');status=read(run/'status.json');proto=read(base/'protocol.json')
    for p in (run/'run_spec.json',run/'training_settings.json',run/'status.json',base/'protocol.json'):protected_add(protected,p)
    validate_spec(sp,st,plan)
    need(status.get('state')=='completed' and int(status['actual_timesteps'])==STEP,'B6 training not complete at requested 40M')
    need(proto['spec']==sp and proto['settings']==st and proto['manifest_sha256']==plan['manifest_sha256'],'Stage protocol differs')
    need(proto['runtime']['hashes']==plan['runtime']['relaxed']['hashes'],'Stage native runtime differs')
    reg=read(run/'checkpoints.json');protected_add(protected,run/'checkpoints.json')
    registered=[c for c in reg if int(c.get('num_timesteps',-1))==STEP and c.get('sha256')==target['sha256'] and not c.get('source_baseline')]
    need(registered,'Exact uploaded40M checkpoint not registered; no earlier/best checkpoint substitution')
    cps=[run/'checkpoints'/f'step_{STEP}.zip',run/'last_model.zip']
    for c in registered:cps+=relpaths(c['path'],root,oldroot)
    cp=next((p for p in dict.fromkeys(cps) if p.is_file() and sha(p)==target['sha256']),None)
    need(cp is not None,'Registered B6 40M weights missing/changed: '+str(cps));protected_add(protected,cp)
    mfopts=[training/'manifest.csv']+relpaths(plan['manifest'],root,oldroot)
    mf=next((p for p in mfopts if p.is_file() and sha(p)==plan['manifest_sha256']),None)
    if mf:
        protected_add(protected,mf);mm=rows(mf)
        need(Counter((r['source'],r['split']) for r in mm)==C.V.EXPECTED_COUNTS,'Full manifest partitions changed')
        need(len({r['map_id'] for r in mm})==len(mm) and len({r['sha256'] for r in mm})==len(mm),'Manifest duplicates/cross-split hash overlap')
        em=[dict(r) for r in mm if r['split'] in ('validation','test')];manifest_kind='original_full_manifest_verified'
    else:
        mf=PACKAGE/'reference/evaluation_manifest1400.csv';protected_add(protected,mf);em=rows(mf)
        manifest_kind='bundled_original_evaluation_subset_matched_to_B6_1400_IDs_and_hashes'
    need(len(em)==1400 and len({r['map_id'] for r in em})==1400 and {r['map_id'] for r in em}==set(expected),'B6 evaluation membership changed')
    for r in em:
        old=expected[r['map_id']]
        for a,b in (('sha256','map_sha256'),('source','source'),('split','split'),('pattern','pattern'),('tier','tier')):
            need(str(r[a])==str(old[b]),'Map metadata mismatch: '+a+' '+r['map_id'])
    chosen=[r for r in em if scope=='all' or (r['split']=='validation')==(scope=='validation')]
    chosen.sort(key=lambda r:(0 if r['split']=='validation' else 1,0 if r['source']=='synthetic' else 1,str(r['tier']),r['pattern'],r['map_id']))
    need(len(chosen)=={'all':1400,'validation':180,'test':1220}[scope],'Wrong evaluation size')
    missing=[]
    for r in chosen:
        opts=relpaths(r['path'],root,oldroot)+[root.parent/r['map_id']]
        f=next((p for p in opts if p.is_file()),None)
        if f is None:missing.append(dict(map_id=r['map_id'],tried=[str(p) for p in opts]))
        else:r['path']=str(f)
    need(not missing,'Original evaluation maps missing ('+str(len(missing))+'): '+str(missing[:5]))
    rh=plan['runtime']['relaxed']['hashes'];needed=tuple(C.NATIVE_FILES)+(FEATURE,)
    snapopts=[training/'_runtime/relaxed']+relpaths(plan['runtime']['relaxed']['path'],root,oldroot)+[PACKAGE/'native_original']
    snap=next((p for p in snapopts if all((p/n).is_file() and sha(p/n)==rh[n] for n in needed)),None)
    need(snap is not None,'Exact hash-matched original B6 runtime missing/changed')
    for n in needed:protected_add(protected,snap/n)
    histories=[check_old_eval(training,sc,target,expected,protected) for sc in (['validation','test'] if scope=='all' else [scope])]
    parameter_digests={h['parameter_sha256'] for h in histories if h.get('parameter_sha256')}
    need(len(parameter_digests)<=1,'Validation/Test model parameter digests differ')
    job=dict(key='B6',B=6,beta=.5,checkpoint=dict(target,path=str(cp)),spec=sp,settings=st,software=plan['software'],
        snapshot=str(snap),sources={n:dict(path=str(snap/n),sha256=rh[n]) for n in C.NATIVE_FILES},
        native_version=plan['runtime']['relaxed']['version'],feature_sha256=rh[FEATURE],records=chosen,
        historical_rows={r['map_id']:expected[r['map_id']] for r in chosen},
        parameter_sha256=next(iter(parameter_digests)) if parameter_digests else None)
    return dict(version=VERSION,root=str(root),training_root=str(training),scope=scope,jobs={'B6':job},protected=protected,
        code_sha256=code_hashes(),source_eval_audit=histories,checkpoint_step=STEP,manifest_kind=manifest_kind,
        sample_partitions=dict(Counter(domain(r) for r in chosen)),
        refinement=dict(solver_sha256=sha(PACKAGE/'ch5_vendor/segment_graph_original.py'),
             common_driver_sha256=sha(PACKAGE/'ch5_common.py'),objective='T_only',fixed_first_visit_order=True,
             fixed_first_arrival_heading=True,step_constraint='each_connector_steps_le_raw_connector_steps',
             native_budget=6,failed_paths_preserved=True),
        no_training=True,no_previous_milestone_evaluation=True,evaluation_device='cpu')


def checked_member(directory,name):
    directory=Path(directory).resolve();p=(directory/name).resolve()
    need(not Path(name).is_absolute() and '..' not in Path(name).parts and directory in p.parents,'Unsafe relative member')
    return p


def save_npz(path,arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    need(not path.is_symlink(),'Symlink output refused')
    tmp=path.with_name(path.name+'.writing.'+str(os.getpid()))
    try:
        with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def load_npz(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def map_key(rec):return hashlib.sha256(rec['map_id'].encode()).hexdigest()[:24]


def receipt_read(directory,name,identity):
    directory=Path(directory);p=directory/name
    if not p.exists():return None
    d=read(p);need(d.get('identity')==identity,'Cache belongs to another protocol: '+str(p))
    for n,h in d.get('files',{}).items():
        f=checked_member(directory,n);need(f.is_file() and sha(f)==h,'Corrupt committed output (not a cache miss): '+str(f))
    return d

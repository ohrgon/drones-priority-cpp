"""Read-only source discovery and authenticated metadata for Strict40M VAL180.
Only the original, registered checkpoints are accepted. No training is invoked.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, os
from collections import Counter
from pathlib import Path
import ch5_common as C
from native_reference import strict50m_original as N

VERSION = 'strict40m_beta_refined_validation_v1_20261004'
CASES = {'strict_b0':0.0, 'strict_b05':0.5, 'strict_b1':1.0}
STEP = 40_001_536
NVAL = 180
TRAIN_DEFAULT = 'strict_beta_runs_seed42'
VAL_DEFAULT = 'cost11_strict_validation_20M_40M_every5M'
OUT_DEFAULT = 'chapter5_TABLE516_STRICT40M_BETA_REFINED_VAL180_v1'
FEATURE = 'nrg1_c3nt_cost5_features.py'
NEEDED = (*C.NATIVE_FILES, FEATURE)
PACKAGE = Path(__file__).resolve().parent
need, read, sha, write, atomic, freeze, digest = N.require, N.read, N.sha, N.write, N.write_text, N.freeze, N.digest
safe, flag = N.safe, N.truth
T,LW=C.T,C.LW


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        r=csv.DictReader(f)
        need(r.fieldnames and len(set(r.fieldnames))==len(r.fieldnames), 'Missing/duplicate CSV header: '+str(path))
        out=list(r)
    need(all(None not in r and all(v is not None for v in r.values()) for r in out), 'Malformed CSV: '+str(path))
    return out


def csvwrite(path, rr, fields=None):
    rr=[safe(r) for r in rr]; fields=fields or list(dict.fromkeys(k for r in rr for k in r)) or ['status']
    b=io.StringIO(newline=''); w=csv.DictWriter(b,fieldnames=fields); w.writeheader()
    for r in rr:
        w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    atomic(path,'\ufeff'+b.getvalue())


def near(a,b,label):
    C.near(a,b,label)


def at(root,p):
    p=Path(p).expanduser();return (p if p.is_absolute() else Path(root)/p).resolve()


def relpaths(value,root,original_root):
    """Preserve relative layout when a suite was moved; never search by mtime."""
    p=Path(str(value).replace('\\','/')).expanduser()
    if not p.is_absolute(): return [at(root,p)]
    choices=[]
    try: choices.append((Path(root)/p.relative_to(Path(original_root))).resolve())
    except ValueError: pass
    try: choices.append((Path(root).parent/p.relative_to(Path(original_root).parent)).resolve())
    except ValueError: pass
    choices.append(p.resolve())
    return list(dict.fromkeys(choices))


def file_with_hash(candidates,expected,label):
    checked=[]; hits=[]
    for p in dict.fromkeys(Path(p).resolve() for p in candidates):
        if p.is_file():
            h=sha(p);checked.append({'path':str(p),'sha256':h})
            if h==expected:hits.append(p)
    need(hits, f'{label} missing or changed; examined {checked or [str(p) for p in candidates]}')
    return hits[0]


def code_hashes():
    names=['run_table516_strict40m_refined.py','s40_io.py','s40_native.py','s40_report.py','ch5_common.py',
           'ch5_vendor/segment_graph_original.py','ch5_vendor/test8_original.py',
           'native_reference/strict50m_original.py','reference/checkpoint_targets.json',
           'reference/validation_membership.json','reference/Table_5_16_Strict40M_Raw_verified.csv']
    return {n:sha(PACKAGE/n) for n in names}


def protected_add(d,p):
    p=Path(p).resolve();d[str(p)]={'sha256':sha(p),'size':p.stat().st_size}


def verify_protected(d):
    for p,meta in d.items():
        need(Path(p).is_file() and sha(p)==meta['sha256'], 'Immutable input changed/missing: '+p)


def validate_row(row, job, rec):
    need(row.get('source')=='synthetic' and row.get('split')=='validation', 'Non-validation row refused')
    for k,w in [('map_id',rec['map_id']),('map_sha256',rec['sha256']),('model',job['key']),
                ('checkpoint_sha256',job['checkpoint']['sha256']),('pattern',rec['pattern']),('tier',rec['tier'])]:
        need(str(row.get(k))==str(w), f'Historical raw row {k} mismatch for '+rec['map_id'])
    need(int(row['checkpoint_step'])==STEP and int(row['nominal_m'])==40,'Wrong historical checkpoint step')
    near(row.get('beta'),job['beta'],'Historical beta');need(flag(row.get('deterministic')),'Not deterministic')
    N.validate_row(row,job)
    pair=hashlib.sha256(json.dumps([rec['sha256'],'priority_mass',0,20260907,
         [int(row['start_r']),int(row['start_c'])],int(row['start_heading'])],sort_keys=True).encode()).hexdigest()
    need(pair==row.get('initial_pair_id'),'Historical initial-condition digest differs')


def expected_identity(job,records):
    native=job['native_job']
    par=dict(version=N.VERSION,beta=job['beta'],checkpoint_sha256=job['checkpoint']['sha256'],checkpoint_step=STEP,
        environment=job['spec']['environment'],runtime_hashes={n:native['runtime_hashes'][n] for n in N.REQUIRED_RUNTIME},
        initial_seed=20260907,seed=42,deterministic=True,device='cpu',validation=[(r['map_id'],r['sha256']) for r in records])
    return {'sha256':digest(par),'parameters':safe(par)}


def historical_raw(job,records,bases,protected):
    """Verify full checkpoint results or native per-map JSON receipts. Metrics are NOT actions."""
    expected=expected_identity(job,records); byid={r['map_id']:r for r in records}; found={}; provenance=[]
    for parent in bases:
        base=Path(parent)/'validation'/job['key']/f'step_{STEP}'
        ip=base/'identity.json';co=base/'complete.json';rr=[]
        if co.exists():
            need(ip.is_file(),'Complete cache has no identity: '+str(base))
            need(read(ip)==expected,'Completed cache identity mismatch: '+str(base))
            d=read(co)
            need(d.get('identity')==expected['sha256'] and d.get('checkpoint_sha256')==job['checkpoint']['sha256'] and int(d.get('n_maps',-1))==NVAL,'Wrong complete receipt: '+str(base))
            need('per_map.csv' in d.get('files',{}),'Receipt lacks per_map.csv: '+str(base))
            for name,h in d['files'].items():
                need(Path(name).name==name and (base/name).is_file() and sha(base/name)==h,'Corrupt completed cache artifact: '+str(base/name))
                protected_add(protected,base/name)
            protected_add(protected,co);protected_add(protected,ip)
            rr=rows(base/'per_map.csv')
            need(len(rr)==NVAL and len({r['map_id'] for r in rr})==NVAL and {r['map_id'] for r in rr}==set(byid),'Wrong validation membership in cache')
        elif ip.exists():
            need(read(ip)==expected,'Partial cache identity mismatch: '+str(base)); protected_add(protected,ip)
            for rec in records:
                p=base/'maps'/(hashlib.sha256(rec['map_id'].encode()).hexdigest()+'.json')
                if not p.exists():continue
                d=read(p);row=d.get('row',{})
                need(d.get('identity')==expected['sha256'] and d.get('row_sha256')==digest(row),'Corrupt partial metric receipt: '+str(p))
                rr.append(row); protected_add(protected,p)
        if not rr:continue
        for r in rr:
            need(r.get('map_id') in byid,'Unexpected map in source');validate_row(r,job,byid[r['map_id']])
            if r['map_id'] in found:
                prior=found[r['map_id']]
                for k in (T,LW,'steps','turn_count','revisit_steps','path_length_m','start_r','start_c','start_heading'):
                    near(r[k],prior[k],'Duplicate raw source '+k)
            else:found[r['map_id']]=r
        provenance.append({'directory':str(base),'verified_rows':len(rr),'full_complete':co.exists(),
                           'stage':'Raw_metrics_only_no_action_trace'})
    return found,provenance


def source_context(root,training_root=None,validation_roots=None):
    root=Path(root).resolve(); targets=read(PACKAGE/'reference/checkpoint_targets.json')
    # The exact name is confirmed in the uploaded plan, with an explicit override and bounded discovery.
    candidates=[at(root,training_root)] if training_root else [root/TRAIN_DEFAULT]
    if not training_root:
        candidates += [p.parent for p in root.glob('*/plan.json') if p.parent not in candidates]
    checked=[];plans=[]
    for p in candidates:
        f=p/'plan.json'
        if not f.is_file():checked.append({'path':str(f),'reason':'missing'});continue
        try:d=read(f)
        except (OSError,ValueError) as e:checked.append({'path':str(f),'reason':str(e)});continue
        if d.get('version')==N.VERSION and set(d.get('jobs',{}))==set(CASES):plans.append((p,d))
    need(len(plans)==1,'Need one original Strict training plan. Use --training-root PATH. Candidates: '+str([str(x[0]) for x in plans] or checked))
    train,plan=plans[0];need(plan.get('launcher_sha256')==sha(PACKAGE/'native_reference/strict50m_original.py'),'Original launcher version changed')
    protected={};protected_add(protected,train/'plan.json');jobs={};master=None;mapchecked={}
    oldroot=plan['root']
    bases=[at(root,p) for p in (validation_roots or [])]+[root/VAL_DEFAULT,train]
    bases=list(dict.fromkeys(bases))
    summaries={r['model']:r for r in rows(PACKAGE/'reference/Table_5_16_Strict40M_Raw_verified.csv')}
    for key,beta in CASES.items():
        orig=plan['jobs'][key];need(orig['beta']==beta,'Plan beta mismatch');N.validate_spec(orig['spec'],orig['settings'],beta)
        env=orig['spec']['environment'];
        for k,v in C.V.PHYSICS.items():need(env.get(k)==v,'Physical setting changed: '+k)
        need(orig['spec'].get('remaining_step_observation') is False,'Wrong clock input')
        encoder=orig['spec'].get('encoder_ablation')
        if encoder is not None:need(encoder.get('merged_feature_dim')==261 and encoder.get('name')=='CNN3','Wrong CNN descriptor')
        # The beta0.5 continuation spec omits encoder_ablation; authenticate its
        # identical feature source and audit the loaded model rather than invent the field.
        need(orig['runtime_hashes'].get(FEATURE)=='57c8aca1afd9b530f01687e5ba4c34602be7f04d34e759468a3dbf9eba57598b','Wrong original CNN3 source hash')
        case=train/key
        proto=read(case/'continuation_protocol.json');protected_add(protected,case/'continuation_protocol.json')
        need(proto['version']==N.VERSION and proto['runtime_hashes']==orig['runtime_hashes'],'Continuation runtime changed')
        need(proto['spec']['environment']==env and proto['spec']['beta']==beta,'Continuation spec changed')
        run=case/'runs'/proto['spec']['run_id'];regpath=run/'checkpoints.json';registry=read(regpath)
        # This registry may grow while the user's training continues. Only the selected entry is frozen.
        cs=[r for r in registry if int(r.get('num_timesteps',-1))==STEP and not r.get('source_baseline')]
        need(cs,'Registered 40M checkpoint absent for '+key+'; no 20M/5M/last_model substitute')
        hashes={r['sha256'] for r in cs};need(len(hashes)==1 and next(iter(hashes))==targets[key]['sha256'],'Wrong/ambiguous 40M checkpoint for '+key)
        opts=[]
        for e in cs:
            name=Path(str(e['path']).replace('\\','/')).name
            opts += [run/'checkpoints'/name,run/name]+relpaths(e['path'],root,oldroot)
        cpfile=file_with_hash(opts,targets[key]['sha256'],'40M weight '+key);protected_add(protected,cpfile)
        snapopts=[case/'source_snapshot']+relpaths(orig['source_snapshot'],root,oldroot);snap=None
        for ss in dict.fromkeys(snapopts):
            if all((ss/n).is_file() and sha(ss/n)==orig['runtime_hashes'].get(n) for n in NEEDED):snap=ss;break
        need(snap is not None,'Native COST11/CNN snapshot missing or changed: '+str([str(p) for p in snapopts]))
        for n in NEEDED:protected_add(protected,snap/n)
        need(orig['runtime_hashes']['c3nt_cost_sweep_generated_core.py']=='bf80e7c13f8dd0c7f83a776725c01d5607031bb116588212ad7db032bcac52d5','Not original Strict COST11 engine')
        mfopts=relpaths(orig['manifest'],root,oldroot)+[case/'manifest.csv']
        mf=next((p for p in mfopts if p.is_file() and sha(p)==orig['manifest_sha256']),None)
        if mf is not None:
            manifest_mode='original_full_manifest_hash_verified'
            records_all=rows(mf);counts=Counter((r['source'],r['split']) for r in records_all)
            need(counts==C.V.EXPECTED_COUNTS,'Changed split membership: '+str(counts))
        else:
            # Existing native validation export is independently anchored to the
            # previously verified 180 ID/hash pairs below. No new split is drawn.
            alternatives=[p/'report'/'validation_manifest180.csv' for p in bases]
            mf=next((p for p in alternatives if p.is_file()),None)
            need(mf is not None,'Neither original manifest nor exported VAL180 manifest is available: '+str(mfopts+alternatives))
            manifest_mode='native_exported_VAL180_membership_verified'
            records_all=rows(mf)
            need(len(records_all)==NVAL and all(r['source']=='synthetic' and r['split']=='validation' for r in records_all),'Not the native VAL180 export')
        protected_add(protected,mf)
        for field in ('map_id','sha256'):
            need(len({r[field] for r in records_all})==len(records_all),'Duplicate manifest '+field)
        records=sorted([dict(r) for r in records_all if r['source']=='synthetic' and r['split']=='validation'],key=lambda r:(r['source'],str(r['tier']),r['pattern'],r['map_id']))
        need(len(records)==NVAL,'Expected 180 synthetic validation maps')
        need([[r['map_id'],r['sha256']] for r in records]==read(PACKAGE/'reference/validation_membership.json'),'Not the previously verified VAL180 membership')
        signature=[(r['map_id'],r['sha256'],r['tier'],r['pattern']) for r in records]
        if master is None:master=signature
        need(master==signature,'Different validation sets across beta')
        for r in records:
            # Map IDs may contain the string "test"; the manifest split, not the filename, defines VAL.
            opts=relpaths(r['path'],root,oldroot)
            actual=next((p for p in opts if p.is_file()),None)
            need(actual is not None,'Original validation array missing: '+str(opts))
            r['path']=str(actual)
        cp=dict(targets[key],path=str(cpfile)); sources={n:{'path':str(snap/n),'sha256':orig['runtime_hashes'][n]} for n in C.NATIVE_FILES}
        j=dict(key=key,beta=beta,checkpoint=cp,spec=orig['spec'],settings=orig['settings'],software=orig['software'],native_job=orig,
            snapshot=str(snap),sources=sources,native_version=N.CORE_VERSION,feature_sha256=orig['runtime_hashes'][FEATURE],
            records=records,reference_summary=summaries[key],manifest_path=str(mf),manifest_validation_mode=manifest_mode)
        rr,provenance=historical_raw(j,records,bases,protected);j.update(historical_rows=rr,historical_provenance=provenance)
        jobs[key]=j
    return {'version':VERSION,'root':str(root),'training_root':str(train),'validation_roots':[str(p) for p in bases],
            'jobs':jobs,'protected':protected,'code_sha256':code_hashes(),'evaluation':{'n_validation':NVAL,'initial_seed':20260907,'seed':42,'deterministic':True,'device':'cpu'},
            'scope':'Strict COST11 FULL beta0/0.5/1.0 40M; VAL180 only',
            'refinement':{'engine':'uploaded_SegmentGraph_shortest_Astar','fixed_first_visit_order':True,'fixed_first_arrival_heading':True,'objective':'T_only',
                          'step_constraint':'each_connector_steps_le_raw_connector_steps','effective_B':0},
            'optimizer_updates':0,'train_rollouts':0,'test_rollouts':0,'osm_rollouts':0,'nominal_m':40,'actual_step':STEP}


def checked_member(directory,name):
    directory=Path(directory).resolve();p=(directory/name).resolve()
    need(not Path(name).is_absolute() and '..' not in Path(name).parts and directory in p.parents,'Unsafe relative file member')
    return p


def save_npz(path,arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
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
    directory=Path(directory);path=directory/name
    if not path.exists():return None
    co=read(path);need(co.get('identity')==identity,'Output identity changed: '+str(path))
    for n,h in co.get('files',{}).items():
        p=checked_member(directory,n);need(p.is_file() and sha(p)==h,'Corrupt committed output (not a cache miss): '+str(p))
    return co

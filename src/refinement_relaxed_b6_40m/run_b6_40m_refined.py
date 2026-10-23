#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B6 FULL beta0.5 COST11 fixed40M -> same B124 connection refinement.
Default scope: Synthetic Validation180 + Synthetic Test720 + OSM Test500.
CPU only, no training. Reconstruct absent Raw actions from the exact completed
40M model; verify against uploaded Raw records; commit Raw before refinement.
Previously completed B0/B1/B2/B4/classical results are reused as references.
"""
from __future__ import annotations
import argparse,os,signal,subprocess,sys,time,traceback,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import b6ref_io as I
import b6ref_native as W
import b6ref_report as R


def verify_package():
    p=I.PACKAGE/'PACKAGE_SHA256.json'
    if p.exists():
        for n,h in I.read(p).items():I.need((I.PACKAGE/n).is_file() and I.sha(I.PACKAGE/n)==h,'Packaged file changed: '+n)


def protocol_load(out):
    d=I.read(Path(out)/'protocol.json')
    I.need(I.digest(d['parameters'])==d['identity'],'Output protocol changed')
    I.need(d['parameters']['code_sha256']==I.code_hashes(),'Output from another code version; do not mix')
    return d['parameters'],d['identity']


def launch(ctx,identity,workers,cached_only=False):
    out=Path(ctx['out']);logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    n=min(workers,len(ctx['jobs']['B6']['records']));active=[];errors=[];last=0.
    try:
        for i in range(n):
            lp=logs/f'B6_part{i+1}.log';f=lp.open('a',encoding='utf-8',buffering=1)
            cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--_worker',str(i),'--_nworkers',str(n),'--_out',str(out)]
            if cached_only:cmd.append('--cached-only')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
            p=subprocess.Popen(cmd,cwd=ctx['root'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((i,p,f,lp));print('[CPU WORKER]',i+1,'/'+str(n),'log='+str(lp),flush=True)
        while active:
            for item in active[:]:
                i,p,f,lp=item;rc=p.poll()
                if rc is None:continue
                f.close();active.remove(item)
                if rc:errors.append(dict(worker=i+1,exit_code=rc,log=str(lp)));print('[WORKER ERROR]',i+1,'see',lp,flush=True)
                else:print('[WORKER DONE]',i+1,flush=True)
            if time.monotonic()-last>30 and active:
                for i,p,f,lp in active:
                    try:
                        with lp.open('rb') as r:r.seek(max(0,lp.stat().st_size-4096));tail=r.read().decode('utf-8','replace').splitlines()
                        line=next((s for s in reversed(tail) if '[REFINE]' in s or '[MODEL LOADED]' in s),None)
                        if line:print(line,flush=True)
                    except OSError:pass
                last=time.monotonic()
            if active:time.sleep(.3)
    except BaseException:
        for i,p,f,lp in active:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGINT)
                except ProcessLookupError:pass
        deadline=time.monotonic()+8
        for i,p,f,lp in active:
            try:p.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
            f.close()
        raise
    return errors


def diagnostics(root,error):
    d=root/'chapter5_B6_refinement_input_diagnostics'/time.strftime('%Y%m%d_%H%M%S');d.mkdir(parents=True,exist_ok=True)
    I.write(d/'input_error.json',dict(status='INPUT_MISSING_OR_INCONSISTENT',root=str(root),error=str(error),traceback=traceback.format_exc(),
        directories=sorted(p.name for p in root.iterdir() if p.is_dir()),evaluation_started=False))
    z=d/'B6_REFINEMENT_INPUT_DIAGNOSTIC.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as a:a.write(d/'input_error.json','input_error.json')
    print('[INPUT DIAGNOSTIC]',z,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--training-root',type=Path,help='Completed original B6 existing20M-to40M output containing plan.json')
    p.add_argument('--out',type=Path)
    p.add_argument('--phase',choices=('all','check','report'),default='all')
    p.add_argument('--scope',choices=('all','validation','test'),default='all')
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=3,help='CPU shards of the ONE exact40M model; no GPU')
    p.add_argument('--cached-only',action='store_true',help='Only reuse this output\'s committed Raw/Refined; fail on missing Raw trace')
    p.add_argument('--_worker',type=int,help=argparse.SUPPRESS)
    p.add_argument('--_nworkers',type=int,default=1,help=argparse.SUPPRESS)
    p.add_argument('--_out',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args();verify_package()
    if a._worker is not None:
        ctx,identity=protocol_load(a._out)
        try:W.worker(ctx,identity,a._worker,a._nworkers,a.cached_only);return 0
        except KeyboardInterrupt:print('[PAUSED] Committed Raw/Refined preserved.',flush=True);return 130
        except Exception as e:
            traceback.print_exc();I.write(Path(ctx['out'])/'logs'/f'B6_part{a._worker+1}_error.json',dict(error=str(e),traceback=traceback.format_exc()));return 1
    root=a.root.expanduser().resolve();I.need(root.is_dir(),'Suite root missing')
    suffix='' if a.scope=='all' else '_'+a.scope
    out=I.at(root,a.out or (I.OUT_DEFAULT+suffix));I.need(out!=root and out not in root.parents,'Output cannot overwrite/contain suite')
    if a.phase=='report':
        ctx,identity=protocol_load(out)
        with I.lock(out/'.run.lock'):audit=R.build_report(ctx,identity)
        return 0 if audit['status']=='COMPLETE' else 2
    try:
        ctx=I.source_context(root,a.training_root,a.scope);src=Path(ctx['training_root'])
        I.need(out!=src and src not in out.parents and out not in src.parents,'Output overlaps original B6 training source')
        for f in ctx['protected']:I.need(out not in Path(f).parents,'Output contains protected input')
        refs,prov=R.baseline_pairs(ctx)
        I.need(set(refs)=={'B0','B1','B2','B4','BCD','WLG','STC'} if a.scope!='validation' else set(refs)=={'B0','B1','B2','B4'},'Previous reference set incomplete')
        ctx['out']=str(out);identity=I.digest(ctx)
    except Exception as e:diagnostics(root,e);raise
    j=ctx['jobs']['B6']
    print('[SCOPE] B6 beta=.5 COST11 FULL fixed40M actual40,001,536; CPU only; no training.',flush=True)
    print('[PARTITIONS]',ctx['sample_partitions'],'Validation is not added to final Test averages.',flush=True)
    print('[CHECKPOINT]',j['checkpoint']['path'],flush=True)
    print('[SHA256]',j['checkpoint']['sha256'],flush=True)
    print('[PLAN]',len(j['records']),'Raw/Refined pairs. Missing actions -> same-checkpoint CPU inference -> Raw metric match -> commit -> refine.',flush=True)
    print('[REUSE] B0/B1/B2/B4 and classical previously verified Raw/Refined rows; no re-inference/re-optimization.',flush=True)
    print('[OUTPUT]',out,flush=True)
    if a.phase=='check':
        print('[CHECK PASS] Files/metadata/hashes/evaluation subsets/prior references checked. No model inference or refinement yet.',flush=True);return 0
    out.mkdir(parents=True,exist_ok=True)
    with I.lock(out/'.run.lock'):
        I.freeze(out/'protocol.json',dict(identity=identity,parameters=ctx))
        I.csvwrite(out/'evaluation_manifest.csv',j['records'])
        I.write(out/'input_audit.json',dict(identity=identity,input_checks_passed=True,scope=a.scope,checkpoint_step=I.STEP,
            checkpoint_sha256=j['checkpoint']['sha256'],source_eval_audit=ctx['source_eval_audit'],no_training=True,original_outputs_read_only=True))
        errors=[];paused=False
        try:errors=launch(ctx,identity,a.workers,a.cached_only)
        except KeyboardInterrupt:paused=True;errors=[dict(reason='user_interrupted')];print('[PAUSED] Rerun same command to reuse committed Raw/Refined.',flush=True)
        audit=R.build_report(ctx,identity,errors)
        return 130 if paused else 0 if audit['status']=='COMPLETE' and not errors else 2


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('[PAUSED]',file=sys.stderr);sys.exit(130)
    except Exception as e:traceback.print_exc();print('[ERROR]',type(e).__name__,str(e),file=sys.stderr);sys.exit(1)

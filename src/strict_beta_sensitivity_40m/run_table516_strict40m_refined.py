#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict COST11 β=0/0.5/1.0 · 40M · Validation180 · RAW -> common REFINED.

No training, no Test/OSM. The former native validation saved metric JSONs, not
primitive actions. Missing action traces are reconstructed with the SAME 40M
checkpoint on CPU, matched against those metrics, saved, then refined. Subsequent
runs reuse committed raw/refined maps. No raw values are labelled Refined until
the common engine and native replay pass for that map.
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time, traceback, zipfile
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_VISIBLE_DEVICES']=''
import s40_io as I
import s40_native as W
import s40_report as R


def verify_package():
    p=I.PACKAGE/'PACKAGE_SHA256.json'
    if not p.exists():return # development/tests; release always includes the manifest
    for n,h in I.read(p).items():I.need(I.sha(I.PACKAGE/n)==h,'Packaged file changed: '+n)


def protocol_load(out):
    doc=I.read(Path(out)/'protocol.json');I.need(I.digest(doc['parameters'])==doc['identity'],'Changed protocol.json')
    I.need(doc['parameters']['code_sha256']==I.code_hashes(),'Output created by other code; do not mix versions')
    return doc['parameters'],doc['identity']


def launch(context,identity,workers,cached_only=False):
    out=Path(context['out']);logs=out/'logs';logs.mkdir(exist_ok=True)
    pending=list(context['jobs']);active=[];errors=[];last=0
    try:
        while pending or active:
            while pending and len(active)<workers:
                key=pending.pop(0);lp=logs/(key+'.log');f=lp.open('a',encoding='utf-8',buffering=1)
                cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--_worker',key,'--_out',str(out)]
                if cached_only:cmd.append('--cached-only')
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                p=subprocess.Popen(cmd,cwd=context['root'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((key,p,f,lp));print('[CPU WORKER]',key,'log='+str(lp),flush=True)
            for item in active[:]:
                key,p,f,lp=item;rc=p.poll()
                if rc is None:continue
                f.close();active.remove(item)
                if rc:errors.append({'model':key,'exit_code':rc,'log':str(lp)});print('[WORKER ERROR]',key,'see',lp,flush=True)
                else:print('[WORKER DONE]',key,flush=True)
            if time.monotonic()-last>30 and active:
                for key,p,f,lp in active:
                    try:
                        with lp.open('rb') as r:r.seek(max(0,lp.stat().st_size-4096));tail=r.read().decode('utf-8','replace').splitlines()
                        line=next((s for s in reversed(tail) if '[VAL-REFINE]' in s),None)
                        if line:print(line,flush=True)
                    except OSError:pass
                last=time.monotonic()
            if active:time.sleep(.25)
    except BaseException:
        for key,p,f,lp in active:
            if p.poll() is None:
                try:os.killpg(p.pid,signal.SIGINT)
                except ProcessLookupError:pass
        deadline=time.monotonic()+5
        for key,p,f,lp in active:
            try:p.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
            f.close()
        raise
    return errors


def diagnostics(root,out,error):
    out.mkdir(parents=True,exist_ok=True)
    d={'status':'INPUT_MISSING_OR_INCONSISTENT','error':str(error),'traceback':traceback.format_exc(),
       'root':str(root),'no_training':True,'no_evaluation_started_by_preflight':True,
       'root_top_level_directories':sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []}
    p=out/'input_error.json';I.write(p,d)
    z=out/'TABLE516_INPUT_DIAGNOSTIC.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:archive.write(p,p.name)
    print('[INPUT DIAGNOSTIC]',z,flush=True)


def main():
    a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--training-root',type=Path,help='Original Strict 50M continuation folder containing plan.json (only its registered 40M checkpoints are used)')
    a.add_argument('--validation-root',type=Path,action='append',help='Additional saved native validation results root')
    a.add_argument('--out',type=Path)
    a.add_argument('--phase',choices=['all','check','report'],default='all')
    a.add_argument('--workers',type=int,choices=[1,2,3,4],default=3,help='At most 3 CPU processes: one per beta')
    a.add_argument('--cached-only',action='store_true',help='Forbid weight loading and missing-trace inference; only process this output\'s saved Raw traces')
    a.add_argument('--_worker',choices=list(I.CASES),help=argparse.SUPPRESS)
    a.add_argument('--_out',type=Path,help=argparse.SUPPRESS)
    args=a.parse_args();verify_package()
    if args._worker:
        context,identity=protocol_load(args._out)
        try:
            W.worker(context,args._worker,identity,args.cached_only)
            return 0
        except KeyboardInterrupt:
            print('[PAUSED] Raw/Refined commits are preserved.',flush=True);return 130
        except Exception as e:
            traceback.print_exc();I.write(Path(context['out'])/'logs'/(args._worker+'_error.json'),{'error':str(e),'traceback':traceback.format_exc()});return 1
    root=args.root.expanduser().resolve();out=I.at(root,args.out or I.OUT_DEFAULT)
    I.need(root.is_dir() and out!=root,'Invalid root/output')
    if args.phase=='report':
        context,identity=protocol_load(out)
        with I.N.lock(out/'.run.lock'):
            audit=R.build_report(context,identity)
        return 0 if audit['status']=='COMPLETE' else 2
    try:
        context=I.source_context(root,args.training_root,args.validation_root)
        for d in [Path(context['training_root']),*[Path(p) for p in context['validation_roots']]]:
            I.need(out!=d and d not in out.parents and out not in d.parents,'Output overlaps input root: '+str(d))
        for p in context['protected']:I.need(out not in Path(p).parents,'Output contains input file')
        context['out']=str(out); identity=I.digest(context)
    except Exception as e:
        diagnostics(root,out,e);raise
    print('[SCOPE] Strict COST11 FULL; beta=0/0.5/1; 40M (actual 40,001,536); Validation180 only',flush=True)
    print('[NO TRAIN / NO TEST / NO OSM]',flush=True)
    print('[TRAINING SOURCE READ ONLY]',context['training_root'],flush=True)
    for key,j in context['jobs'].items():
        print('[CHECKPOINT]',key,j['checkpoint']['path'],flush=True)
        print('[RAW REFERENCE]',key,str(len(j['historical_rows']))+'/180 verified metric rows; original validator does not save actions',flush=True)
    print('[PLAN] 3 beta x 180 maps = 540 pairs. Missing primitive traces require same-checkpoint CPU inference, NOT training.',flush=True)
    if args.phase=='check':
        print('[CHECK PASS] Registered weights / hashes / native snapshots / validation membership. No model deserialization, inference or refinement executed.',flush=True);return 0
    out.mkdir(parents=True,exist_ok=True)
    with I.N.lock(out/'.run.lock'):
        I.freeze(out/'protocol.json',{'identity':identity,'parameters':context})
        I.csvwrite(out/'validation_manifest.csv',context['jobs']['strict_b0']['records'])
        I.write(out/'input_audit.json',{'input_checks_passed':True,'identity':identity,'checkpoint_step':I.STEP,
             'raw_reference_count':{k:len(j['historical_rows']) for k,j in context['jobs'].items()},'source_roots':context['validation_roots'],
             'three_beta_same_membership':True,'refinement_not_yet_assumed_complete':True})
        errors=[];paused=False
        try:errors=launch(context,identity,min(3,args.workers),args.cached_only)
        except KeyboardInterrupt:
            paused=True;errors.append({'reason':'user_interrupted'});print('[PAUSED] Re-run the same command. Completed raw traces and refinements remain.',flush=True)
        audit=R.build_report(context,identity,errors)
        return 130 if paused else 0 if audit['status']=='COMPLETE' and not errors else 2


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('[PAUSED]',file=sys.stderr);sys.exit(130)
    except Exception as e:
        traceback.print_exc();print('[ERROR]',type(e).__name__,e,file=sys.stderr);sys.exit(1)

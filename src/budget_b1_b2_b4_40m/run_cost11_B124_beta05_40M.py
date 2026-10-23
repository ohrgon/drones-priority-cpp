#!/usr/bin/env python3
"""B=1/2/4, beta=.5: fresh native COST11 PPO ->40M, CP5M / VAL1M.
Run inside the existing uav_chapter4_suite_v2, using its original Python environment.
This launcher verifies frozen native sources and never substitutes a generic engine.
"""
from __future__ import annotations
import argparse, os, signal, subprocess, sys
from pathlib import Path
import b124_engine as E
import b124_native_support as U

DEFAULT_OUT = 'chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42'


def check(plan):
    E.verify_plan(plan)
    # B=0 on both native implementations must be behaviorally identical on the
    # deterministic preflight trace; positive B must exercise all budget transitions.
    E.run_worker(plan, 'trace', 0, family='strict')
    E.run_worker(plan, 'trace', 0, family='relaxed')
    a=U.read(Path(plan['out'])/'audit/preflight_strict_B0.json')
    b=U.read(Path(plan['out'])/'audit/preflight_relaxed_B0.json')
    U.require(a['trace']==b['trace'], 'Native Strict and relaxed(B0) are not equivalent on preflight trace')
    for B in (1,2,4):E.run_worker(plan, 'trace', B)
    for B,gpu in zip((1,2,4),plan['gpus']):E.run_worker(plan,'gpu_check',B,gpu=gpu)
    U.write(Path(plan['out'])/'audit/preflight_complete.json',dict(passed=True,
            B0_trace_equivalent=True,positive_Bs=[1,2,4],optimizer_updates=0,
            checkpoint_interval=5_000_000,validation_interval=1_000_000))
    print('[CHECK PASS] B semantics, native runtime, original software and three assigned GPUs',flush=True)


def coordinator(plan,B,mode):
    gpu=plan['gpus'][(1,2,4).index(B)]
    with U.lock(Path(plan['out'])/f'B{B}'/'.coordinator.lock'):
        if mode=='train':
            for stage in range(3):E.run_worker(plan,'train_segment',B,stage,gpu=gpu)
        elif mode=='validate':
            for stage in range(3):E.pending(plan,B,stage)
        elif mode=='test':
            _,run=E.segment_paths(plan,B,2)
            E.final_checkpoint(run)  # completion guard, not selection on test performance
            cps=[x for x in E.milestones(plan,B,2) if x['nominal_m']==40]
            U.require(len(cps)==1,'Fixed40M permanent checkpoint is missing')
            E.run_worker(plan,'eval',B,2,cps[0]['path'],scope='test')
        else:raise ValueError(mode)


def launch(plan,mode):
    logs=Path(plan['out'])/'logs';logs.mkdir(exist_ok=True)
    jobs=[]
    try:
        for B,gpu in zip((1,2,4),plan['gpus']):
            env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    CUDA_VISIBLE_DEVICES=str(gpu) if mode=='train' else '',MPLBACKEND='Agg')
            command=[sys.executable,str(Path(__file__).resolve()),'--_worker','coordinator',
                     '--_plan',str(Path(plan['out'])/'plan.json'),'--B',str(B),'--mode',mode]
            path=logs/f'B{B}_{mode}.log';f=path.open('a',encoding='utf-8',buffering=1)
            process=subprocess.Popen(command,cwd=plan['root'],env=env,stdout=f,
                        stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((B,process,f,path))
            print(f'[LAUNCH {mode.upper()}] B={B} beta=.5 GPU={gpu if mode=="train" else "CPU"} '
                  f'log={path}',flush=True)
        failures=[]
        for B,process,f,path in jobs:
            rc=process.wait();f.close()
            print(f'[DONE {mode.upper()}] B={B} exit={rc}',flush=True)
            if rc:failures.append(dict(B=B,exit=rc,log=str(path)))
        U.require(not failures,'Workers failed; committed recovery and caches retained: '+str(failures))
    except KeyboardInterrupt:
        for _,process,_,_ in jobs:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for _,process,f,_ in jobs:
            process.wait();f.close()
        raise
    finally:
        for _,process,f,_ in jobs:
            if process.poll() is not None and not f.closed:f.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--gpus',nargs=3,type=int,default=[1,2,3],metavar=('B1_GPU','B2_GPU','B4_GPU'))
    parser.add_argument('--phase',choices=['all','check','train','validate','test','report'],default='all')
    parser.add_argument('--_worker',choices=['trace','gpu_check','train_segment','eval','coordinator'],help=argparse.SUPPRESS)
    parser.add_argument('--_plan',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--B',type=int,choices=[0,1,2,4],help=argparse.SUPPRESS)
    parser.add_argument('--stage',type=int,choices=[0,1,2],default=0,help=argparse.SUPPRESS)
    parser.add_argument('--scope',choices=['validation','test'],default='validation',help=argparse.SUPPRESS)
    parser.add_argument('--family',choices=['strict','relaxed'],default='relaxed',help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=['train','validate','test'],help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        U.require(args._plan is not None and args.B is not None,'Missing worker plan/B')
        plan=U.read(args._plan);E.verify_plan(plan)
        if args._worker=='trace':E.trace_preflight(plan,args.B,args.family)
        elif args._worker=='gpu_check':
            rt=E.runtime(plan,args.B,train=True)
            U.require(rt.cfg.detour_budget_max==args.B and rt.cfg.mask_mode=='relaxed','Incorrect assigned B')
            print(f'[GPU CHECK PASS] B={args.B}; visible CUDA device=0; optimizer updates=0',flush=True)
        elif args._worker=='train_segment':E.train_segment(plan,args.B,args.stage)
        elif args._worker=='eval':
            U.require(args.checkpoint is not None,'Missing checkpoint')
            from b124_evaluation import evaluate
            evaluate(plan,args.B,args.stage,args.checkpoint,args.scope)
        else:
            U.require(args.mode is not None,'Missing coordinator mode');coordinator(plan,args.B,args.mode)
        return 0
    root=args.root.expanduser().resolve()
    U.require(root.is_dir(),'Suite root not found: '+str(root))
    out=args.out.expanduser().resolve() if args.out else root/DEFAULT_OUT
    U.require(len(set(args.gpus))==3 and min(args.gpus)>=0,'Use three distinct nonnegative GPU IDs')
    with U.lock(out/'.launcher.lock'):
        plan=E.make_plan(root,out,args.gpus);E.verify_plan(plan)
        print('[NATIVE FIX] uploaded core verified; complete reset/step budget audits parameterized; legacy failed output not reused',flush=True)
        print('[PLAN] B=1,2,4; beta=.5; fresh weights; seed42; COST11 FULL H5 CNN3; TOTAL40M each',flush=True)
        print('[STAGES] 0->5M LR=max(5e-5,2.5e-4*progress_over_5M); 5->20->40M LR=5e-5',flush=True)
        print('[SAVE] permanent: 5M,10M,...,40M; recovery: every50k after native completed optimizer update',flush=True)
        print('[VAL] every1M, synthetic180, CPU; temporary non5M weights deleted after successful evaluation',flush=True)
        print('[TEST] fixed40M only; synthetic720 and OSM500 separate; B0/BCD/WLG numerical caches reused',flush=True)
        if args.phase in ('all','check','train'):check(plan)
        if args.phase in ('all','train'):launch(plan,'train')
        if args.phase=='validate':launch(plan,'validate')
        if args.phase in ('all','test'):launch(plan,'test')
        if args.phase!='check':
            from b124_report import build_report
            build_report(plan,bundle=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[PAUSED] Re-run the same command. No reset/deletion is needed; native committed recovery is reused.',file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

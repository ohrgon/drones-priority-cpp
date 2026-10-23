#!/usr/bin/env python3
"""B=6, beta=.5, COST11: continue the ORIGINAL completed 20M to total40M.

Default physical GPU=0. No B1/B2/B4 or old B6 model/result is modified.
Training/recovery/evaluation use the byte-identical native B124 modules.
"""
from __future__ import annotations
import argparse, os, signal, subprocess, sys
from pathlib import Path
import b124_engine as E
import b124_native_support as U
import b6_resume_plan as P
import b6_plan as O

ENTRY=Path(__file__).resolve()


def worker_command(plan, mode, B, stage=0, checkpoint=None, scope='validation', family='relaxed'):
    U.require(B in (0,6),'This package runs B6 only (B0 is a preflight control)')
    cmd=[sys.executable,str(ENTRY),'--_worker',mode,'--_plan',str(Path(plan['out'])/'plan.json'),
         '--B',str(B),'--stage',str(stage),'--scope',scope,'--family',family]
    if checkpoint is not None:
        cmd += ['--checkpoint',str(checkpoint)]
    return cmd


def run_worker(plan,mode,B,stage=0,checkpoint=None,scope='validation',family='relaxed',gpu=None):
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=str(gpu) if gpu is not None else '',MPLBACKEND='Agg')
    p=subprocess.run(worker_command(plan,mode,B,stage,checkpoint,scope,family),cwd=plan['root'],env=env)
    U.require(p.returncode==0,f'{mode} failed (B={B}, stage={stage}); committed recovery/caches retained')


def install_orchestration():
    """Route worker/report callbacks only. Native learning and core source stay unchanged."""
    E.VERSION=P.VERSION
    E.KEYS=('B6',)
    E.segment_paths=P.segment_paths
    E.segment_spec=P.segment_spec
    E.extract_train=P.extract_train
    E.run_worker=run_worker
    import b124_report as old_report
    import b6_resume_report
    old_report.build_report=b6_resume_report.build_report


def check(plan):
    E.verify_plan(plan)
    E.run_worker(plan,'trace',6)
    E.run_worker(plan,'gpu_check',6,gpu=plan['gpus'][0])
    U.write(Path(plan['out'])/'audit/preflight_complete.json',dict(
        passed=True,source_step=P.SOURCE_STEP,source_weights_used=True,fresh_weights=False,
        positive_Bs=[6],optimizer_updates=0,GPU=plan['gpus'][0],
        checkpoint_interval=5_000_000,validation_interval=1_000_000))
    print('[CHECK PASS] existing20M policy/Adam, native B6 runtime, sampler, software and GPU; no updates',flush=True)


def source_model_check(p):
    # Runs in its own selected-GPU subprocess. This only verifies; no gradient update.
    rt=E.runtime(p,6,train=True)
    from stable_baselines3.common.save_util import load_from_zip_file
    from sb3_contrib import MaskablePPO
    e=p['existing_source'];cp=e['checkpoint_path']
    U.require(U.sha(cp)==e['checkpoint_sha256'],'Imported source checkpoint changed')
    _,params,_=load_from_zip_file(cp,device='cpu',load_data=False)
    U.require(params and 'policy' in params and 'policy.optimizer' in params,'Missing policy/Adam')
    expected=U.tensor_digest(params)
    model=MaskablePPO.load(cp,device='cpu')
    U.require(U.tensor_digest(model.get_parameters())==expected,'Source policy/Adam round trip differs')
    U.require(model.num_timesteps==e['actual_step'] and model._n_updates==e['epoch_counter'],'Source counters differ')
    U.require(bool(model.policy.optimizer.state_dict()['state']),'Source has no learned Adam moments')
    sp,st=P.segment_spec(p,6,2);E.assert_ppo(model,sp,st)
    rt.features.audit_cnn3_model(model);rt.core.audit_model(model)
    orders=U.rebuild_orders(rt,dict(settings=U.read(Path(e['snapshot_run'])/'training_settings.json'),source_run=e['snapshot_run']))
    U.write(Path(p['out'])/'audit/source_model_preflight.json',dict(
        source_step=e['actual_step'],checkpoint_sha256=e['checkpoint_sha256'],
        tensor_digest=expected,policy_Adam_exact=True,learning_rate=5e-5,
        sampler_positions=[x['episode'] for x in orders],optimizer_updates=0))
    print('[SOURCE MODEL PASS] existing B6 beta=.5 step20,004,864; policy+Adam+sampler verified',flush=True)
    del model


def coordinator(plan,mode):
    with U.lock(Path(plan['out'])/'B6/.coordinator.lock'):
        if mode=='train':
            for stage in plan['active_stages']:
                E.run_worker(plan,'train_segment',6,stage,gpu=plan['gpus'][0])
        elif mode=='validate':
            for stage in plan['active_stages']:
                E.pending(plan,6,stage)
        elif mode=='test':
            _,run=E.segment_paths(plan,6,2)
            E.final_checkpoint(run)
            cps=[x for x in E.milestones(plan,6,2) if x['nominal_m']==40]
            U.require(len(cps)==1,'Registered fixed 40M checkpoint missing; no earlier model is substituted')
            E.run_worker(plan,'eval',6,2,cps[0]['path'],scope='test')
        else:
            raise ValueError(mode)


def launch(plan,mode):
    logs=Path(plan['out'])/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'B6_{mode}.log'
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',MPLBACKEND='Agg',
               CUDA_VISIBLE_DEVICES=str(plan['gpus'][0]) if mode=='train' else '')
    command=worker_command(plan,'coordinator',6)+['--mode',mode]
    with path.open('a',encoding='utf-8',buffering=1) as log:
        process=subprocess.Popen(command,cwd=plan['root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'[LAUNCH {mode.upper()}] B=6 beta=.5 GPU={plan["gpus"][0] if mode=="train" else "CPU"} log={path}',flush=True)
        try:
            rc=process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            process.wait()
            raise
    print(f'[DONE {mode.upper()}] B=6 exit={rc}',flush=True)
    U.require(rc==0,'Worker failed. Do not delete output; inspect '+str(path))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,help='Separate B6 output; never use the B124 output')
    parser.add_argument('--source-root',type=Path,help='Existing original20M experiment root or run directory')
    parser.add_argument('--b124-root',type=Path,help='Original B124 output with plan.json/manifest.csv/_runtime')
    parser.add_argument('--gpu',type=int,default=None,help='Physical CUDA device, default 0 for a new run; resume keeps recorded device')
    parser.add_argument('--phase',choices=['all','check','train','validate','test','report'],default='all')
    parser.add_argument('--_worker',choices=['trace','gpu_check','train_segment','eval','coordinator'],help=argparse.SUPPRESS)
    parser.add_argument('--_plan',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--B',type=int,choices=[0,6],help=argparse.SUPPRESS)
    parser.add_argument('--stage',type=int,choices=[0,1,2],default=0,help=argparse.SUPPRESS)
    parser.add_argument('--scope',choices=['validation','test'],default='validation',help=argparse.SUPPRESS)
    parser.add_argument('--family',choices=['strict','relaxed'],default='relaxed',help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--mode',choices=['train','validate','test'],help=argparse.SUPPRESS)
    a=parser.parse_args(argv)
    install_orchestration()
    if a._worker:
        U.require(a._plan is not None and a.B is not None,'Missing worker arguments')
        p=U.read(a._plan)
        U.require(p['version']==P.VERSION and p['Bs']==[6], 'Wrong worker plan')
        E.verify_plan(p)
        if a._worker=='trace':E.trace_preflight(p,a.B,a.family)
        elif a._worker=='gpu_check':
            source_model_check(p)
        elif a._worker=='train_segment':
            U.require(a.B==6,'No B0 training permitted')
            E.train_segment(p,6,a.stage)
        elif a._worker=='eval':
            U.require(a.B==6 and a.checkpoint is not None,'B6 evaluation checkpoint required')
            from b124_evaluation import evaluate
            evaluate(p,6,a.stage,a.checkpoint,a.scope)
        else:
            U.require(a.mode is not None and a.B==6,'B6 coordinator mode required')
            coordinator(p,a.mode)
        return 0
    root=a.root.expanduser().resolve()
    out=(a.out.expanduser().resolve() if a.out else root/P.DEFAULT_OUT)
    U.require(root.is_dir(),'Suite root missing: '+str(root))
    U.require(out!=root and out not in root.parents and not O.paths_overlap(out,ENTRY.parent),
              'Unsafe output path')
    # Reject source-overlap before even placing a launcher lock in that tree.
    if (out/'plan.json').is_file():
        existing=U.read(out/'plan.json')
        U.require(existing.get('version')==P.VERSION and existing.get('Bs')==[6],
                  'Refusing to lock another experiment output. Use a separate B6 --out.')
    else:
        src=O.source_root(root,a.b124_root)
        old,_=P.input_source(root,a.source_root)
        U.require(not O.paths_overlap(out,src) and not O.paths_overlap(out,old),'Output overlaps a source experiment')
    with U.lock(out/'.launcher.lock'):
        p=P.make_plan(root,out,a.gpu,a.source_root,a.b124_root)
        E.verify_plan(p)
        print('[PLAN] B6 beta=.5 COST11 FULL, seed42, CNN3/no-clock, H=ceil(5N)',flush=True)
        print('[START] EXISTING B6 beta=.5 at 20,004,864; no 0M/5M retraining; source='+p['existing_source']['original_run'],flush=True)
        print('[SCHEDULE] 20,004,864 -> TOTAL 40,000,000 (final update40,001,536); LR=5e-5; old20M is INCLUDED, not +40M',flush=True)
        print('[SAVE] new permanent CP25/30/35/40M; validation21..40M on synthetic180/CPU; native full-state recovery',flush=True)
        print('[TEST] --phase all includes fixed40M Raw Test720/OSM500; --phase train excludes Test',flush=True)
        print('[PROTECTED] B1/B2/B4 and old B6 sources remain read-only; output='+str(out),flush=True)
        if a.phase in ('all','check','train'):check(p)
        if a.phase in ('all','train'):launch(p,'train')
        if a.phase=='validate':launch(p,'validate')
        if a.phase in ('all','test'):launch(p,'test')
        if a.phase!='check':
            from b6_resume_report import build_report
            build_report(p,bundle=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[PAUSED] Re-run the same command; committed recovery/caches retained.',file=sys.stderr)
        raise SystemExit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

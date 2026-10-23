"""Local tests. Synthetic fixtures are NEVER manuscript results."""
import copy,importlib.util,json,sys,unittest,tempfile,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import run_kappa1_fresh5m as R
E=R.E

def fake_own(other,failures=0):
    rr=[dict(r) for r in other if int(float(r['cost']))==3]
    for r in rr:r['cost']=1;r['experiment_origin']='test_fixture_not_results'
    for r in [r for r in rr if E.flag(r['is_success'])][:failures]:r['is_success']=False
    return rr

def runtime():
    job=dict(snapshot=str(P/'native'),spec=R.make_request('cpu')['spec'],
             hashes={n:E.sha(P/'native'/n) for n in E.FILES})
    return E.load_native(job),job

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.other,_,cls.old,cls.oldmaps=R.refs()
    def test_existing_eight_complete(self):
        self.assertEqual(len(self.other),1440);self.assertEqual(len(self.oldmaps),167)
    def test_existing_k11_values(self):
        r=next(r for r in self.old if r['kappa']==11)
        self.assertAlmostEqual(r['Raw_T_sec'],1864.89,places=2)
        self.assertAlmostEqual(r['Refined_Lw_sec'],726.76,places=2)
    def test_fresh_parameters(self):
        q=R.make_request('cuda:0')
        self.assertFalse(q['old_checkpoint_required']);self.assertFalse(q['warm_start']);self.assertFalse(q['test_enabled'])
        self.assertEqual(q['settings']['total_timesteps'],5000000)
        self.assertEqual(q['settings']['ppo']['n_steps'],1024)
        self.assertEqual(q['settings']['checkpoint_every_timesteps'],200000)
        self.assertEqual(q['spec']['environment']['detour_budget_max'],6)
        self.assertEqual(q['spec']['reward_definition']['cost_multiplier'],1)
    def test_training_and_eval_membership(self):
        rr=E.rows(P/'reference/train_validation_manifest.csv')
        self.assertEqual(len(rr),1080)
        self.assertEqual({r['split'] for r in rr},{'train','validation'})
        va={r['map_id']:r['sha256'] for r in rr if r['split']=='validation'}
        self.assertEqual(va,{r['map_id']:r['map_sha256'] for r in self.other})
    def test_new_common_recomputed(self):
        a,m=R.nine_table(fake_own(self.other,5),self.other)
        self.assertLess(len(m),167);self.assertEqual(len(a),9)
        self.assertEqual({r['common_n'] for r in a},{len(m)})
        self.assertEqual(a[0]['origin'],'fresh_rerun')
        self.assertTrue(all(r['origin']=='existing_saved_result' for r in a[1:]))
    def test_no_165_success_requirement(self):
        a,m=R.nine_table(fake_own(self.other),self.other)
        self.assertEqual(a[0]['success_n'],170);self.assertEqual(len(m),167)
    def test_zero_common_is_not_zero_time(self):
        own=fake_own(self.other)
        for r in own:r['is_success']=False
        a,m=R.nine_table(own,self.other)
        self.assertEqual(len(m),0);self.assertTrue(all(r['Refined_T_sec'] is None for r in a))
    def test_duplicate_rejected(self):
        own=fake_own(self.other);own[-1]=dict(own[0])
        with self.assertRaises(RuntimeError):R.nine_table(own,self.other)
    def test_map_hash_rejected(self):
        own=fake_own(self.other);own[0]['map_sha256']='0'*64
        with self.assertRaises(RuntimeError):R.nine_table(own,self.other)
    def test_initial_pair_rejected(self):
        own=fake_own(self.other);own[0]['initial_pair_id']='bad'
        with self.assertRaises(RuntimeError):R.nine_table(own,self.other)
    def test_core_coefficient_one(self):
        rt,job=runtime()
        parts=rt.core.reward_parts(new_count=1,n_active=10,mass_before=10,total_mass=20,dt=2,time_cap=100,beta=.5,completed=False,exhausted=False)
        self.assertAlmostEqual(parts['coverage'],.1)
        self.assertAlmostEqual(parts['mission_time'],-.01)
        self.assertAlmostEqual(parts['latency'],-.005)
        self.assertEqual(parts['terminal'],0)
    def test_core_reward_success_cap(self):
        rt,job=runtime()
        args=dict(new_count=1,n_active=2,mass_before=1,total_mass=2,dt=1,time_cap=20,beta=.5,exhausted=True)
        self.assertEqual(rt.core.reward_parts(completed=True,**args)['terminal'],2)
        self.assertEqual(rt.core.reward_parts(completed=False,**args)['terminal'],0)
    def test_native_trace_refinement_and_success(self):
        rt,job=runtime();a=np.zeros((50,50),np.uint8);a[1:4,1:4]=1
        rec=dict(map_id='fixture',sha256=rt.utils.array_hash(a),pattern='fixture',tier='fixture',path='not-used')
        job.update(evaluation=dict(seed=42,initial_seed=20260907),checkpoint_step=5005312,checkpoint_sha256='test-only')
        ic=E.initial(rt,rec,a,20260907)
        # Deterministic local progress choice, NOT a trained policy.
        def decision(c,obs,mask):
            boundary,strict=rt.core.reference_masks(c)
            return int(np.flatnonzero(strict)[0])
        raw,tr=E.run_episode(rt,job,rec,a,ic,decision)
        refined,tr2,audit,segs=E.C.common_refine(rt,raw,tr)
        self.assertEqual(raw['is_success'],1);self.assertEqual(refined['is_success'],1)
        self.assertLessEqual(refined[E.T],raw[E.T]+1e-6)
        self.assertTrue(audit['order_preserved']);self.assertTrue(audit['arrival_heading_preserved'])
    def test_actual_connector_shortening(self):
        rt,job=runtime();a=np.zeros((50,50),np.uint8);a[1,1]=1;a[1,4]=1
        rec=dict(map_id='fixture_detour',sha256=rt.utils.array_hash(a),pattern='fixture',tier='fixture',path='not-used')
        job.update(evaluation=dict(seed=42,initial_seed=20260907),checkpoint_step=5005312,checkpoint_sha256='test-only')
        ic=dict(start=[1,1],heading=1,pair_id='test-only');seq=iter([1,3,1,1,1])
        raw,tr=E.run_episode(rt,job,rec,a,ic,lambda c,o,m:next(seq))
        ref,tr2,audit,segs=E.C.common_refine(rt,raw,tr)
        self.assertTrue(audit['path_changed']);self.assertLess(ref[E.T],raw[E.T])
        self.assertEqual(list(tr2['actions']),[1,1,1])
    def test_failed_trajectory_unchanged(self):
        rt,job=runtime();a=np.zeros((50,50),np.uint8);a[1,1]=1;a[1,48]=1
        rec=dict(map_id='fixture_failure',sha256=rt.utils.array_hash(a),pattern='fixture',tier='fixture',path='not-used')
        job.update(evaluation=dict(seed=42,initial_seed=20260907),checkpoint_step=5005312,checkpoint_sha256='test-only')
        ic=dict(start=[1,1],heading=1,pair_id='test-only')
        raw,tr=E.run_episode(rt,job,rec,a,ic,lambda c,o,m:1)
        ref,tr2,audit,segs=E.C.common_refine(rt,raw,tr)
        self.assertEqual(raw['is_success'],0);self.assertEqual(ref['is_success'],0)
        self.assertTrue(np.array_equal(tr['actions'],tr2['actions']))
    def test_saved_raw_resume_no_model_inference(self):
        rt,job=runtime();a=np.zeros((50,50),np.uint8);a[1:4,1:4]=1
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);np.save(root/'map.npy',a)
            rec=dict(map_id='test_saved_raw',sha256=rt.utils.array_hash(a),pattern='fixture',tier='fixture',path=str(root/'map.npy'))
            job.update(evaluation=dict(seed=42,initial_seed=20260907),checkpoint_step=5005312,checkpoint_sha256='test-only')
            ic=E.initial(rt,rec,a,20260907)
            def decision(c,o,m):return int(np.flatnonzero(rt.core.reference_masks(c)[1])[0])
            raw,tr=E.run_episode(rt,job,rec,a,ic,decision)
            d=R.mapdir(root,rec);d.mkdir(parents=True)
            E.write(d/'raw_metrics.json',raw);R.save_npz(d/'raw_trace.npz',tr)
            E.write(d/'raw_complete.json',dict(identity='i',files={n:E.sha(d/n) for n in ('raw_trace.npz','raw_metrics.json')}))
            R.worker_init(job)
            old=E.load_model
            def forbidden(*a,**k):raise AssertionError('Should not infer a committed raw trace')
            E.load_model=forbidden
            try:
                first=R.process((rec,str(root),'i'));second=R.process((rec,str(root),'i'))
            finally:E.load_model=old
            self.assertEqual(first,second);self.assertTrue((d/'complete.json').exists())
    def test_cache_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);E.write(d/'x.json',{'x':1});E.write(d/'complete.json',dict(identity='i',files={'x.json':E.sha(d/'x.json')}))
            self.assertTrue(R.committed(d,'complete.json','i'))
            E.write(d/'x.json',{'x':2})
            with self.assertRaises(RuntimeError):R.committed(d,'complete.json','i')
    def test_raw_commit_preserved_separately(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);R.save_npz(d/'raw_trace.npz',{'positions':np.array([[0,0],[0,1]])})
            E.write(d/'raw_metrics.json',{'T':1})
            E.write(d/'raw_complete.json',dict(identity='i',files={n:E.sha(d/n) for n in ('raw_trace.npz','raw_metrics.json')}))
            self.assertIsNone(R.committed(d,'complete.json','i'))
            self.assertIsNotNone(R.committed(d,'raw_complete.json','i'))
    def test_native_trainer_generation(self):
        spec=importlib.util.spec_from_file_location('gen_original',P/'native/run_nrg1_c3nt.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.transform_trainer((P/'reference/upstream_trainers_ch4_bounded_b6_nrg1.py').read_text()),
                         (P/'native/trainers_ch4_bounded_b6_nrg1_c3nt.py').read_text())
    def test_refine_engine_unchanged(self):
        # Explicit known native reward SHA already required by old evaluator.
        self.assertEqual(E.sha(P/'native/nrg1_c3nt_core.py'),E.NATIVE_HASH)
    def test_report_without_learning(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'report').mkdir()
            a=R.report(out)
            self.assertEqual(a['status'],'PARTIAL');self.assertEqual(a['kappa1_maps'],0)
            self.assertTrue((out/'TABLE514_KAPPA1_FRESH5M_REVIEW.zip').exists())

if __name__=='__main__':unittest.main(verbosity=2)

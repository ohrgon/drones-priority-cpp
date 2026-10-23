"""Unit tests use temporary synthetic fixtures, never new research data."""
from __future__ import annotations
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

P = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(P))
import kappa_extension as X
import run_kappa5_15_20 as R
E, F = X.E, X.F
sys.path.insert(0, str(X.UP/'native'))
import core_ch4_bounded_b6 as B
import core_ch4 as I
import full_v2_utils as U


def fixture_runtime(k, directory):
    core_path = directory/f'core{k}.py'; core_path.write_text(X.scaled_core(k)[0])
    name = f'fixture_core_{k}_{abs(hash(str(directory)))}'
    spec = importlib.util.spec_from_file_location(name, core_path)
    core = importlib.util.module_from_spec(spec); sys.modules[name]=core; spec.loader.exec_module(core)
    cfg = B.ExperimentEnvConfig(**X.make_request(k,'cpu')['spec']['environment'])
    return SimpleNamespace(core=core, cfg=cfg, factory=core.wrap_factory(B.make_core), initial=I,
                           utils=U, bounded=B, moves=E.C.V.MOVE_TUPLES, snapshot=directory)


def fake_cost(eight,k):
    """Non-paper fixture based on the reference schema; never exported outside a tmpdir."""
    rr = [dict(r) for r in eight if int(float(r['cost'])) == 3]
    for r in rr:
        r.update(cost=str(k), split='validation', checkpoint_step=str(X.FINAL_STEP),
                 checkpoint_sha256='a'*64,
                 experiment_origin=('fresh_rerun_not_recovered_historical_checkpoint' if k==1 else 'TEST_FIXTURE'))
        if not E.flag(r['is_success']):
            r['witness_time_sec']=r[E.T]; r['witness_steps']=r['steps']
    return rr


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eight,_,cls.old,cls.maps=F.refs()
        cls.one=fake_cost(cls.eight,1)
        cls.new=[r for k in X.ADDED for r in fake_cost(cls.eight,k)]

    def test_01_fixed_cost_set(self):
        self.assertEqual(X.ALL_COSTS,(1,3,5,7,9,10,11,12,13,14,15,20))
        self.assertEqual(len(self.eight),1440)
        X.check_rows(self.eight,X.EXISTING[1:])

    def test_02_original_hyperparameters(self):
        ref=E.read(X.UP/'reference/training_settings.json')
        for k in X.ADDED:
            r=X.make_request(k,'cuda:0')
            self.assertEqual(r['settings'],ref)
            self.assertEqual(r['spec']['environment'],E.read(X.UP/'reference/fresh_spec.json')['environment'])
            self.assertFalse(r['warm_start']);self.assertFalse(r['test_enabled']);self.assertFalse(r['osm_enabled'])
            self.assertEqual(r['spec']['reward_definition']['cost_multiplier'],k)
            self.assertEqual(r['settings']['total_timesteps'],5000000)

    def test_03_only_time_costs_scale(self):
        kwargs=dict(new_count=1,n_active=10,mass_before=10,total_mass=20,dt=2,time_cap=100,beta=.5,completed=True,exhausted=True)
        with tempfile.TemporaryDirectory() as td:
            for k in X.ADDED:
                rt=fixture_runtime(k,Path(td));p=rt.core.reward_parts(**kwargs)
                self.assertAlmostEqual(p['coverage'],.1);self.assertEqual(p['terminal'],2)
                self.assertAlmostEqual(p['latency'],-.005*k);self.assertAlmostEqual(p['mission_time'],-.01*k)
                self.assertEqual(p['invalid'],0)

    def test_04_terminal_failure_and_singleton(self):
        with tempfile.TemporaryDirectory() as td:
            for k in X.ADDED:
                rt=fixture_runtime(k,Path(td));a=np.zeros((50,50),np.uint8);a[2,2]=5
                c=rt.factory(rt.cfg);c.reset(a,start=(2,2),heading=1,seed=42)
                o,r,t,tr,info=c.step(0)
                self.assertEqual(r,2);self.assertTrue(t);self.assertFalse(tr);self.assertEqual(c.steps,0)
                self.assertLess(info['nrg_return_identity_error'],1e-6)

    def test_05_same_transitions_masks_and_scaled_return(self):
        with tempfile.TemporaryDirectory() as td:
            runtimes=[fixture_runtime(k,Path(td)) for k in X.ADDED]
            a=np.zeros((50,50),np.uint8);a[1:5,1:5]=1
            cores=[rt.factory(rt.cfg) for rt in runtimes]
            for c in cores:c.reset(a,start=(1,1),heading=1,seed=42)
            while not cores[0]._done:
                action=int(np.flatnonzero(runtimes[0].core.reference_masks(cores[0])[1])[0])
                masks=[c.action_masks().copy() for c in cores]
                self.assertTrue(all(np.array_equal(masks[0],m) for m in masks))
                results=[c.step(action) for c in cores]
                for j in range(1,len(cores)):
                    self.assertEqual(cores[j].drone_pos,cores[0].drone_pos)
                    self.assertEqual(cores[j].elapsed_time_sec,cores[0].elapsed_time_sec)
                    for key in results[0][0]:self.assertTrue(np.array_equal(results[0][0][key],results[j][0][key]))
            for c,k in zip(cores,X.ADDED):
                info=c.info(True);c.assert_consistent()
                self.assertEqual(info['c3nt_cost_multiplier'],k)
                self.assertLess(info['nrg_return_identity_error'],1e-6)

    def test_06_refinement_shorten_order_heading_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            for k in X.ADDED:
                rt=fixture_runtime(k,Path(td));a=np.zeros((50,50),np.uint8);a[1,1]=1;a[1,4]=1
                rec=dict(map_id='fixture',sha256=U.array_hash(a),pattern='fixture',tier='fixture')
                job=dict(checkpoint_step=X.FINAL_STEP,checkpoint_sha256='fixture',evaluation=dict(seed=42,initial_seed=1))
                ic=dict(start=[1,1],heading=1,pair_id='fixture'); seq=iter([1,3,1,1,1])
                raw,tr=E.run_episode(rt,job,rec,a,ic,lambda c,o,m:next(seq))
                ref,tr2,audit,segs=E.C.common_refine(rt,raw,tr)
                self.assertLess(ref[E.T],raw[E.T]);self.assertEqual(list(tr2['actions']),[1,1,1])
                self.assertTrue(audit['arrival_heading_preserved']);self.assertTrue(audit['order_preserved'])

    def test_07_failed_paths_never_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            for k in X.ADDED:
                rt=fixture_runtime(k,Path(td));a=np.zeros((50,50),np.uint8);a[1,1]=1;a[1,48]=1
                rec=dict(map_id='fixture',sha256=U.array_hash(a),pattern='fixture',tier='fixture')
                job=dict(checkpoint_step=X.FINAL_STEP,checkpoint_sha256='fixture',evaluation=dict(seed=42,initial_seed=1))
                raw,tr=E.run_episode(rt,job,rec,a,dict(start=[1,1],heading=1,pair_id='fixture'),lambda c,o,m:1)
                ref,tr2,audit,segs=E.C.common_refine(rt,raw,tr)
                self.assertFalse(E.flag(raw['is_success']));self.assertIsNone(ref[E.LW])
                self.assertTrue(np.array_equal(tr['actions'],tr2['actions']))
                self.assertEqual(raw['reward_sum_terminal'],0)

    def test_08_native_trainer_and_engine_unchanged(self):
        expected=E.read(X.UP/'PACKAGE_SHA256.json')
        for name in ('native/trainers_ch4_bounded_b6_nrg1_c3nt.py','native/nrg1_c3nt_features.py','engine/ch5_common.py',
                     'native/resume_ch4.py','native/coverage_core.py','engine/ch5_vendor/segment_graph_original.py'):
            self.assertEqual(E.sha(X.UP/name),expected[name])

    def test_09_private_snapshot_configured(self):
        with tempfile.TemporaryDirectory() as td:
            for k in X.ADDED:
                out=Path(td)/str(k);out.mkdir();h=X.install_snapshot(out,k)
                for name in ('trainers_ch4_bounded_b6_nrg1_c3nt.py','nrg1_c3nt_features.py','core_ch4_bounded_b6.py'):
                    self.assertEqual(h[name],E.sha(X.UP/'native'/name))
                self.assertIn(X.native_version(k),(out/'source_snapshot/run_nrg1_c3nt_logfix.py').read_text())

    def test_10_twelve_common_recomputed(self):
        rows=copy.deepcopy(self.one+self.eight+self.new)
        for r in rows:
            if int(float(r['cost']))==20 and E.flag(r['is_success']):
                r.update(is_success=False, original_Lw_sec=None,witness_Lw_sec=None, witness_time_sec=r[E.T]);break
        tab,common=X.table(rows,X.ALL_COSTS)
        self.assertEqual(len(tab),12);self.assertLess(len(common),167)
        self.assertEqual({r['common_n'] for r in tab},{len(common)})
        self.assertTrue(all(r['origin']=='existing_saved_result' for r in tab if r['kappa'] in X.EXISTING))

    def test_11_empty_common_is_undefined_not_zero(self):
        rows=copy.deepcopy(self.one+self.eight+self.new)
        for r in rows:
            if int(float(r['cost']))==20:
                r.update(is_success=False,original_Lw_sec=None,witness_Lw_sec=None,witness_time_sec=r[E.T])
        tab,common=X.table(rows,X.ALL_COSTS)
        self.assertFalse(common);self.assertTrue(all(r['Raw_T_sec'] is None and r['Refined_Lw_sec'] is None for r in tab))

    def test_12_duplicate_rejected(self):
        rows=copy.deepcopy(self.one);rows[-1]=dict(rows[0])
        with self.assertRaises(RuntimeError):R.validated_existing(rows)

    def test_13_membership_and_pair_rejected(self):
        for key in ('map_sha256','initial_pair_id'):
            rows=copy.deepcopy(self.one);rows[0][key]='bad'
            with self.assertRaises(RuntimeError):R.validated_existing(rows)

    def test_14_changed_old_eight_rejected(self):
        rows=copy.deepcopy(self.one+self.eight)
        rows[-1][E.T]=str(float(rows[-1][E.T])+1)
        with self.assertRaises(RuntimeError):R.validated_existing(rows)

    def test_15_test_and_wrong_checkpoint_rejected(self):
        for key,v in (('split','test'),('source','osm'),('checkpoint_step','40001536')):
            rows=copy.deepcopy(self.one);rows[0][key]=v
            with self.assertRaises(RuntimeError):R.validated_existing(rows)

    def test_16_reference_csv_zip_and_freeze(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();base=root/'old';(base/'report').mkdir(parents=True)
            E.csvwrite(base/'report/per_map_nine_costs.csv',self.one+self.eight)
            with zipfile.ZipFile(root/'old.zip','w') as z:z.write(base/'report/per_map_nine_costs.csv','report/per_map_nine_costs.csv')
            a,_=R.existing_payload(base);b,_=R.existing_payload(root/'old.zip');self.assertEqual(a,b)
            one,meta=R.freeze_existing(root,out,base)
            one2,meta2=R.freeze_existing(root,out,None);self.assertEqual(one,one2);self.assertEqual(meta,meta2)

    def test_17_missing_kappa1_does_not_fabricate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir();r,m=R.freeze_existing(root,out,None)
            self.assertEqual(r,[]);self.assertEqual(m['status'],'KAPPA1_REFERENCE_MISSING')

    def test_18_raw_complete_cache_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);E.write(d/'x.json',{'a':1});E.write(d/'raw_complete.json',dict(identity='i',files={'x.json':E.sha(d/'x.json')}))
            self.assertTrue(F.committed(d,'raw_complete.json','i'));E.write(d/'x.json',{'a':2})
            with self.assertRaises(RuntimeError):F.committed(d,'raw_complete.json','i')

    def test_19_spawned_cpu_refinement_all_costs(self):
        for k in X.ADDED:
            p=subprocess.run([sys.executable,'-B',str(P/'tests/worker_probe.py'),'--kappa',str(k)],
                             capture_output=True,text=True,timeout=40)
            self.assertEqual(p.returncode,0,p.stdout+'\n'+p.stderr)
            self.assertIn(f'PASS COST{k}',p.stdout)

    def test_20_no_numeric_target_reproduction(self):
        tab,common=X.table(self.one+self.eight+self.new,X.ALL_COSTS)
        self.assertEqual(next(r['success_n'] for r in tab if r['kappa']==1),170)
        self.assertEqual(len(common),167)  # fixture-derived, never a fixed constraint


    def test_21_reference_one_only_equals_wide_nine(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'out';out.mkdir()
            E.csvwrite(root/'one.csv',self.one)
            E.csvwrite(root/'nine.csv',self.one+self.eight)
            one,meta=R.freeze_existing(root,out,root/'nine.csv')
            one2,meta2=R.freeze_existing(root,out,root/'one.csv')
            self.assertEqual(one,one2)

    def test_22_report_complete_and_partial(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);E.write(root/'plan.json',{'test_fixture':True})
            plan={'out':str(root)}
            def fixture_case(plan,k):return fake_cost(self.eight,k)
            with patch.object(R,'complete_case',fixture_case), patch.object(R,'make_review',lambda p,o: None):
                a=R.build_report(plan,self.one,{'test_fixture':True})
                self.assertEqual(a['status'],'COMPLETE')
                self.assertEqual(a['total_per_map_rows'],2160)
                self.assertEqual(len(E.rows(root/'report/table_5_14b_twelve_costs_RAW_REFINED.csv')),12)
                self.assertEqual(len(E.rows(root/'report/per_map_added_costs.csv')),540)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);E.write(root/'plan.json',{'test_fixture':True});plan={'out':str(root)}
            with patch.object(R,'complete_case',lambda p,k:[]):
                a=R.build_report(plan,[],{'status':'missing'})
                self.assertEqual(a['status'],'PARTIAL')
                self.assertFalse((root/'report/table_5_14b_twelve_costs_RAW_REFINED.csv').exists())
                with zipfile.ZipFile(root/'TABLE514_KAPPA5_15_20_REVIEW.zip') as z:
                    self.assertIsNone(z.testzip())


if __name__=='__main__':unittest.main(verbosity=2)

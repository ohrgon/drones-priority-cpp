"""Tiny synthetic integration fixture. NO trained model, NO manuscript results."""
from __future__ import annotations
import argparse
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(P))
import kappa_extension as X
E, F = X.E, X.F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--kappa', type=int, required=True); args = ap.parse_args()
    k = args.kappa; X.configure(k)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td); hashes = X.install_snapshot(out, k)
        job = dict(snapshot=str(out / 'source_snapshot'), spec=X.make_request(k, 'cpu')['spec'], hashes=hashes,
                   kappa=k, native_version=X.native_version(k), evaluation=dict(seed=42, initial_seed=20260907),
                   checkpoint_step=X.FINAL_STEP, checkpoint_sha256='a'*64)
        rt = E.load_native(job)
        a = np.zeros((50, 50), np.uint8); a[1:4, 1:4] = 1
        np.save(out / 'fixture.npy', a)
        rec = dict(map_id='TEST_FIXTURE_NOT_A_PAPER_RESULT', sha256=rt.utils.array_hash(a),
                   pattern='fixture', tier='fixture', path=str(out/'fixture.npy'))
        ic = E.initial(rt, rec, a, job['evaluation']['initial_seed'])
        raw, trace = E.run_episode(rt, job, rec, a, ic,
                decision=lambda c, o, m: int(np.flatnonzero(rt.core.reference_masks(c)[1])[0]))
        assert raw['c3nt_cost_multiplier'] == k
        assert raw['method'] == f'relaxed_kappa{k}'
        d = F.mapdir(out, rec); d.mkdir(parents=True)
        E.write(d/'raw_metrics.json', raw); F.save_npz(d/'raw_trace.npz', trace)
        E.write(d/'raw_complete.json', dict(identity='fixture', files={n:E.sha(d/n) for n in ('raw_metrics.json','raw_trace.npz')}))
        # Spawn is deliberate: verify that each CPU worker receives the correct cost
        # BEFORE resolving the original runner task and loading its native module.
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=X.refinement_worker_init, initargs=(job,)) as pool:
            result = pool.submit(F.process, (rec, str(out), 'fixture')).result(timeout=30)
        assert result['cost'] == k and result['is_success'] is True
        assert result['witness_time_sec'] <= result['mission_time_sec'] + 1e-6
        # A second call must use committed results, without any model present.
        X.refinement_worker_init(job)
        result2 = F.process((rec, str(out), 'fixture'))
        assert result == result2
        assert E.read(d/'refinement_audit.json')['arrival_heading_preserved'] is True
        print(f'PASS COST{k}: native reward, spawned CPU refinement, raw cache reuse, completed cache reuse')


if __name__ == '__main__':
    main()

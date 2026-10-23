"""Read-only optional B1/B2/B4/B6 comparison at the SAME checkpoint and Raw stage."""
from __future__ import annotations
from pathlib import Path
import b124_native_support as U
from b124_report import summarize


def append_budget_comparison(plan, available, report):
    source=Path(plan['b124_source_root']);outrows=[];status=[]
    for (phase,B,m),b6 in sorted(available.items()):
        if B!=6:
            continue
        step=int(b6[0]['checkpoint_step'])
        allrows={6:b6}; errors=[]
        for budget in (1,2,4):
            base=source/phase/f'B{budget}'/f'step_{step}'
            complete=base/'complete.json'
            if not complete.is_file():
                errors.append(f'B{budget}: no committed {phase} {m}M result')
                continue
            try:
                c=U.read(complete)
                U.require(c['scope']==phase and int(c['checkpoint_step'])==step and int(c['nominal_m'])==m,
                          'checkpoint identity mismatch')
                for name,h in c['files'].items():
                    U.require(U.sha(base/name)==h,'completed file changed: '+name)
                rr=U.rows(base/'per_map.csv')
                U.require(len(rr)==(180 if phase=='validation' else 1220),'incomplete map count')
                d={r['map_id']:r for r in rr}
                U.require(len(d)==len(rr)==len(b6) and set(d)=={r['map_id'] for r in b6},'map membership mismatch')
                for r in b6:
                    other=d[r['map_id']]
                    U.require(int(other['effective_B'])==budget and U.near(other['beta'],.5), 'wrong budget/beta')
                    U.require(int(other['checkpoint_step'])==step and U.truth(other['raw_only']) and
                              not U.truth(other['connection_refinement']),'stage mismatch')
                    for k in ('map_sha256','initial_pair_id','source','split','pattern','tier'):
                        U.require(str(other[k])==str(r[k]),'mismatched '+k)
                allrows[budget]=rr
            except (RuntimeError,ValueError,KeyError,OSError) as exc:
                errors.append(f'B{budget}: '+str(exc))
        status.append(dict(phase=phase,nominal_m=m,actual_step=step,
            completed_Bs=sorted(allrows),comparison_ready=len(allrows)==4,
            messages=errors,source_results_not_modified=True))
        if len(allrows)!=4:
            continue
        ds={b:{r['map_id']:r for r in rr} for b,rr in allrows.items()}
        common=set(ds[6])
        for d in ds.values():
            common &= {i for i,r in d.items() if U.truth(r['is_success'])}
        scopes=[('VAL_syn180',set(ds[6]))] if phase=='validation' else [
            ('Test_syn720',{i for i,r in ds[6].items() if r['source']=='synthetic'}),
            ('Test_osm500',{i for i,r in ds[6].items() if r['source']=='osm'}),
            ('Combined_Test1220_auxiliary',set(ds[6]))]
        for scope,ids in scopes:
            common_here=ids & common
            for budget in (1,2,4,6):
                d=ds[budget];s=summarize([d[i] for i in sorted(common_here)])
                s['time_mean_population']='B1_B2_B4_B6_common_success_same_actual_checkpoint_Raw'
                outrows.append(dict(scope=scope,nominal_m=m,actual_step=step,B=budget,beta=.5,
                    evaluated_population_N=len(ids),own_success_S=sum(U.truth(d[i]['is_success']) for i in ids),
                    common_map_ids=sorted(common_here),**s))
    U.csv_write(report/'B1_B2_B4_B6_same_checkpoint_common_success.csv',outrows)
    U.write(report/'B1_B2_B4_B6_comparison_status.json',dict(
        rule='only committed equal-step Raw evaluations with matching map/initial pair; no missing results fabricated',
        pairs=status,source_root=str(source),source_is_read_only=True))

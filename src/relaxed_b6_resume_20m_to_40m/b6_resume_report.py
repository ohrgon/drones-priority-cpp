"""Numerical reports. Failure rates and own-/common-success denominators stay explicit.
No test-based checkpoint ranking or automatic winner selection is performed.
"""
from __future__ import annotations
from collections import defaultdict
import html, json, os, zipfile
from pathlib import Path
import b124_native_support as U
# Training/evaluation stay in the unchanged b124 modules; only report indexing is B6-specific.

METRICS = [U.T, U.LW, 'path_length_m', 'turn_count', 'revisit_steps', 'zero_transit_steps',
           'priority_arrival_25_sec', 'priority_arrival_50_sec', 'priority_arrival_75_sec',
           'priority_arrival_90_sec', 'masked_entropy_mean', 'multiaction_normalized_entropy_mean']
DIAGNOSTICS = ['learning_rate', 'approx_kl', 'clip_fraction', 'entropy_loss',
               'explained_variance', 'value_loss', 'policy_gradient_loss']


def summarize(rows, target=None):
    good = [r for r in rows if U.truth(r['is_success'])]
    n = len(rows); target = n if target is None else target
    result = dict(N_target=target, N_evaluated=n, success_S=len(good),
                  success_pct=100*len(good)/target if target else None, time_mean_n=len(good),
                  time_mean_population='own_success', coverage_pct=U.mean(r.get('coverage_pct') for r in rows))
    for key in METRICS:
        result[key] = U.mean(r.get(key) for r in good)
    t, lw = result[U.T], result[U.LW]
    result['Lw_over_T_ratio_of_means'] = lw/t if t and lw is not None else None
    result['mean_mapwise_Lw_over_T'] = U.mean(float(r[U.LW])/float(r[U.T]) for r in good
                                             if U.number(r.get(U.T)) not in (None, 0))
    decisions = sum(int(r.get('decision_steps') or 0) for r in rows)
    result['action_total_steps'] = decisions
    result['action_map_mean_population'] = 'all evaluated prefixes including failures'
    result['allowed_actions_map_mean'] = U.mean(r.get('allowed_actions_mean') for r in rows)
    total = sum(int(r.get('allowed_actions_sum') or 0) for r in rows)
    result['allowed_actions_pooled_step_mean'] = total/decisions if decisions else None
    for k in range(1,5):
        count = sum(int(r.get(f'allowed_actions_k{k}_count') or 0) for r in rows)
        result[f'allowed_actions_k{k}_count'] = count
        result[f'allowed_actions_k{k}_ratio'] = count/decisions if decisions else None
    result['chosen_strict_step_ratio'] = (sum(float(r.get('chosen_strict_ratio') or 0) * int(r.get('decision_steps') or 0)
                                              for r in rows)/decisions if decisions else None)
    return result


def comparisons(rows, baselines):
    counts = dict(A=0, B=0, C=0, D=0, AB=0)
    good_ids = []
    for row in rows:
        key = row['map_id']
        bcd, wlg = baselines['bcd'][key], baselines['wlg'][key]
        for base in (bcd,wlg):
            U.require(base['map_sha256'] == row['map_sha256'] and base['initial_pair_id'] == row['initial_pair_id'],
                      'Unpaired baseline comparison: '+key)
            U.require(U.truth(base['is_success']), 'Baseline failure requires an explicit paired denominator')
        if not U.truth(row['is_success']):
            continue
        good_ids.append(key)
        t,lw = float(row[U.T]),float(row[U.LW])
        tb,lb = float(bcd[U.T]),float(bcd[U.LW]);tw,lwlg = float(wlg[U.T]),float(wlg[U.LW])
        A=t<tb and lw<lb;B=t<tw and lw<lwlg
        counts['A']+=int(A);counts['B']+=int(B)
        counts['C']+=int(t<tw and lw<lb);counts['D']+=int(t<tb and lw<lwlg)
        counts['AB']+=int(A and B)
    result = dict(criteria_denominator_N=len(rows), **counts)
    for name,n in counts.items():
        result[name+'_pct_of_all'] = 100*n/len(rows) if rows else None
    good = [r for r in rows if U.truth(r['is_success'])]
    for method in ('bcd','wlg','ppo_b0_beta05_40m'):
        bases = [baselines[method][i] for i in good_ids]
        result[method+'_paired_mean_n'] = len(bases)
        for metric in (U.T,U.LW):
            base = U.mean(r[metric] for r in bases); ours=U.mean(r[metric] for r in good)
            result[method+'_'+metric] = base
            result['delta_pct_vs_'+method+'_'+metric] = 100*(ours-base)/base if base and ours is not None else None
    return result


def scopes(rows, phase):
    if phase == 'validation':
        yield 'VAL_syn180', rows
    else:
        yield 'Test_syn720', [r for r in rows if r['source']=='synthetic']
        yield 'Test_osm500', [r for r in rows if r['source']=='osm']
        yield 'Combined_Test1220_auxiliary', rows


def build_report(plan, bundle=True):
    out=Path(plan['out']); report=out/'report';report.mkdir(parents=True,exist_ok=True)
    with U.lock(out/'.report.lock', blocking=True):
        baselines={}
        for phase in ('validation','test'):
            data=U.rows(out/f'reference_{phase}_baselines.csv')
            baselines[phase]={}
            for row in data:
                baselines[phase].setdefault(row['method'],{})[row['map_id']]=row
        summary=[]; strata=[]; raw=[]; available={}; diagnostics=[]; inventory=[]
        for phase in ('validation','test'):
            for B in (6,):
                for done in sorted((out/phase/f'B{B}').glob('step_*/complete.json')):
                    completion=U.read(done);base=done.parent
                    for name,h in completion['files'].items():
                        U.require(U.sha(base/name)==h, 'Completed report input changed: '+str(base/name))
                    rows=U.rows(base/'per_map.csv');m=int(completion['nominal_m'])
                    U.require(len(rows)==(180 if phase=='validation' else 1220), 'Incomplete input was marked complete')
                    available[(phase,B,m)]=rows;raw.extend(dict(evaluation_phase=phase,**r) for r in rows)
                    for scope,rr in scopes(rows,phase):
                        meta=dict(scope=scope,detour_B=B,beta=.5,nominal_m=m,checkpoint_step=completion['checkpoint_step'])
                        summary.append(dict(**meta,**summarize(rr),**comparisons(rr,baselines[phase])))
                        groups=[]
                        if scope!='Combined_Test1220_auxiliary':
                            for grouping,keys in [('tier',['tier']),('category',['pattern']),('category_x_tier',['pattern','tier'])]:
                                by=defaultdict(list)
                                for row in rr:by[tuple(row[k] for k in keys)].append(row)
                                for vals,items in sorted(by.items()):
                                    strata.append(dict(**meta,grouping=grouping,**dict(zip(keys,vals)),
                                                       **summarize(items),**comparisons(items,baselines[phase])))
        # B6 vs immutable B0/BCD/WLG numerical references, on common-success maps.
        common=[]
        for phase, m in sorted({(phase,m) for phase,B,m in available}):
            if not all((phase,B,m) in available for B in (6,)):continue
            maps={B:{r['map_id']:r for r in available[phase,B,m]} for B in (6,)}
            good=set(maps[6])
            for d in maps.values():good &= {i for i,r in d.items() if U.truth(r['is_success'])}
            partitions=[('VAL_syn180',good)] if phase=='validation' else [
                ('Test_syn720',{i for i in good if maps[6][i]['source']=='synthetic'}),
                ('Test_osm500',{i for i in good if maps[6][i]['source']=='osm'}),
                ('Combined_Test1220_auxiliary',good)]
            for scope,ids in partitions:
                for B,d in maps.items():
                    rr=[d[i] for i in sorted(ids)]
                    s=summarize(rr);s['time_mean_population']='B6_B0_BCD_WLG_common_success'
                    common.append(dict(scope=scope,nominal_m=m,method=f'B{B}',**s))
                for method in ('ppo_b0_beta05_40m','bcd','wlg'):
                    rr=[baselines[phase][method][i] for i in sorted(ids)]
                    s=summarize(rr);s['time_mean_population']='B6_B0_BCD_WLG_common_success'
                    common.append(dict(scope=scope,nominal_m=m,method=method,**s))
        from b124_engine import segment_paths, milestones
        for B in (6,):
            for stage in plan.get('active_stages',[2]):
                _,run=segment_paths(plan,B,stage)
                for cp in milestones(plan,B,stage):
                    inventory.append(dict(B=B,nominal_m=cp['nominal_m'],actual_step=cp['num_timesteps'],
                                          stage=stage,path=cp['path'],sha256=cp['sha256']))
                log=run/'training_updates.csv'
                if log.is_file():
                    # A worker may be appending. A partial final line is not reported as a complete update.
                    for row in U.rows(log):
                        if None in row or any(row.get(k) is None for k in row):continue
                        diagnostics.append(dict(B=B,stage=stage,**row))
        summary.sort(key=lambda r:({'VAL_syn180':0,'Test_syn720':1,'Test_osm500':2,'Combined_Test1220_auxiliary':3}[r['scope']],r['nominal_m'],r['detour_B']))
        for name,rr in [('checkpoint_summary.csv',summary),('category_tier_summary.csv',strata),
                        ('common_success_comparison.csv',common),('training_diagnostics.csv',diagnostics),
                        ('permanent_checkpoint_inventory.csv',inventory)]:U.csv_write(report/name,rr)
        expected_val=[(B,m) for B in (6,) for m in plan['expected_validation_milestones']]
        missing_val=[{'B':B,'nominal_m':m} for B,m in expected_val if ('validation',B,m) not in available]
        missing_test=[B for B in (6,) if ('test',B,40) not in available]
        U.write(report/'status.json',dict(validation_complete_count=len(expected_val)-len(missing_val),
                validation_target=len(expected_val),source_step=plan['source_actual_step'],historical_1M_20M_not_repeated=True,missing_validation=missing_val,missing_fixed40M_test=missing_test,
                permanent_checkpoint_interval=5_000_000,validation_interval=1_000_000,
                primary_checkpoint='fixed40M; no automatic test-based selection',
                state='COMPLETE' if not missing_val and not missing_test else 'PARTIAL'))
        def table(rr,columns):
            text='<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr></thead><tbody>'
            for row in rr:
                cells=[]
                for c in columns:
                    v=row.get(c)
                    if isinstance(v,float):v=f'{v:,.4f}'
                    cells.append('<td>'+html.escape('—' if v is None else str(v))+'</td>')
                text+='<tr>'+''.join(cells)+'</tr>'
            return text+'</tbody></table></div>'
        columns=['scope','detour_B','nominal_m','success_S','N_evaluated','time_mean_n',U.T,U.LW,'Lw_over_T_ratio_of_means','A','B','C','D','AB']
        title='COST11 · B=6 · β=0.5 · 40M (existing20M → total40M) · checkpoint 5M / validation 1M'
        body='<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+title+'</title><style>body{font:15px system-ui;line-height:1.7;margin:30px auto;max-width:1500px;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #ccc;padding:8px;text-align:right}th{background:#eee}.scroll{overflow:auto}code{background:#eee}</style><h1>'+title+'</h1>'
        body+=f'<p>완료된 Validation {len(expected_val)-len(missing_val)}/{len(expected_val)} new checkpoint (21–40M). Fixed40M Test {1-len(missing_test)}/1 model.</p>'
        body+='<p>기존 B6·β=0.5·COST11 20M의 정책 및 Adam 상태를 불러와 총40M까지 이어 학습한다. 0–20M은 다시 학습하지 않는다. 추가 구간 LR=5e-5, 영구 checkpoint25/30/35/40M, 새 Validation21–40M이다. 경계에서 지도 선택 순서는 보존하고 진행 중 episode와 전역 RNG는 재시작한다. 이후 중단 복구는 native full-state이다. B124와 하이퍼파라미터·학습량은 맞추되 B6의 기존 학습 이력까지 새 실행과 동일하다고 주장하지 않는다.</p>'
        body+='<p>표의 시간 평균은 성공 경로만 사용하며 n을 함께 표시한다. baseline 감소율은 해당 행의 동일 성공 지도에 다시 맞춘 평균이다. A=BCD 두 지표, B=WLG 두 지표, C=T(WLG)+Lw(BCD), D=T(BCD)+Lw(WLG), AB=A∩B. 모든 조건은 strict &lt;이며 실패는 달성에 포함하지 않는다. action 통계는 실패 prefix까지 포함한다.</p>'
        body+='<p>Test_syn720과 Test_osm500을 분리하고 Combined1220은 보조로만 제시한다. 이전 Test 관찰 이력이 있으므로 blind held-out 일반화 검증으로 과장하지 않는다. Stage1 Raw이며 연결 개선은 수행하지 않는다.</p>'
        body+='<p>참고 baseline은 B124에 사용한 기존 Raw 수치이다. 5.1의 all-five Refined 기준선 결과를 갱신하는 표가 아니다. B0는 고정40M reference이며 B6 중간 checkpoint와 같은 학습량이라고 해석하지 않는다.</p>'
        body+='<h2>완료된 checkpoint</h2>'+table(summary,columns)
        body+='<h2>파일</h2><p>checkpoint_summary.csv · category_tier_summary.csv · common_success_comparison.csv · training_diagnostics.csv · permanent_checkpoint_inventory.csv · status.json</p></html>'
        U.write_text(report/'index.html',body)
        from b6_budget_report import append_budget_comparison
        append_budget_comparison(plan,available,report)
        if bundle:
            U.csv_write(report/'all_per_map.csv',raw)
            dst=report/'UPLOAD_B6_EXISTING20M_TO40M_results.zip';tmp=dst.with_name(dst.name+'.writing')
            with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
                for f in report.iterdir():
                    if f.is_file() and f.suffix in ('.csv','.json','.html'):z.write(f,'report/'+f.name)
                z.write(out/'plan.json','plan.json')
                for f in sorted((out/'source20M').glob('*.json')):z.write(f,'source20M/'+f.name)
                package=Path(__file__).resolve().parent
                for f in sorted((package/'evidence').glob('*.json')):
                    z.write(f,'source_evidence/'+f.name)
                for f in sorted(package.glob('*.py')):
                    z.write(f,'code/'+f.name)
                for name in ('reference_validation_baselines.csv','reference_test_baselines.csv'):z.write(out/name,name)
                for f in sorted((out/'audit').glob('*.json')):z.write(f,'audit/'+f.name)
            os.replace(tmp,dst)
        print('[REPORT]',report/'index.html',flush=True)

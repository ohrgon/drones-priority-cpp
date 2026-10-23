"""Compute Table5.16 only from completed RAW/REFINED pairs on the same VAL set."""
from __future__ import annotations
from collections import Counter
import html, itertools, json, math, os, zipfile
from pathlib import Path
import s40_io as I
import s40_native as W
T,LW=I.T,I.LW


def avg(rr,k):
    vals=[I.N.number(r.get(k)) for r in rr]
    return math.fsum(v for v in vals if v is not None)/len(vals) if vals and all(v is not None for v in vals) else None


def reduction(before,after):return 100*(before-after)/before if before not in (None,0) and after is not None else None


def outcome(dt,dl,eps=1e-6):
    if abs(dt)<=eps or abs(dl)<=eps:return 'tie_in_one_or_both'
    if dt>0 and dl>0:return 'A_shorter_T_and_Lw'
    if dt<0 and dl<0:return 'B_shorter_T_and_Lw'
    return 'A_shorter_T_B_shorter_Lw' if dt>0 else 'B_shorter_T_A_shorter_Lw'


def render_table(rr,fields):
    def val(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:,.4f}'
        return html.escape(str(v))
    return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+val(r.get(k))+'</td>' for k in fields)+'</tr>' for r in rr)+'</tbody></table>'


def build_report(context,identity,errors=None):
    out=Path(context['out']);report=out/'report';report.mkdir(parents=True,exist_ok=True)
    allrows=[];allaudit=[];status=[];raws={};refs={};commiterrors=[];historical=[]
    for key,job in context['jobs'].items():
        rr={};ff={};success=0
        for rec in job['records']:
            d=out/'methods'/key/'maps'/I.map_key(rec)
            try:
                result=W.verify_result(d,identity,rec)
            except Exception as e:commiterrors.append({'key':key,'map_id':rec['map_id'],'error':str(e)});continue
            if result is None:continue
            raw,ref,audit=result
            for r in (raw,ref):
                I.need(r.get('source')=='synthetic' and r.get('split')=='validation' and r['checkpoint_sha256']==job['checkpoint']['sha256'] and int(r['checkpoint_step'])==I.STEP,'Wrong completed table row')
            rr[rec['map_id']]=raw;ff[rec['map_id']]=ref
            allrows += [dict(raw,model=key,table_stage='raw'),dict(ref,model=key,table_stage='refined')]
            allaudit.append(audit)
        done=len(rr)==len(job['records'])
        status.append({'model':key,'beta':job['beta'],'checkpoint_step':I.STEP,'checkpoint_sha256':job['checkpoint']['sha256'],
            'target_maps':len(job['records']),'completed_raw_refined_pairs':len(rr),'status':'COMPLETE' if done else 'PARTIAL'})
        raws[key]=rr;refs[key]=ff
        # Raw results must reproduce the stored same-checkpoint baseline, not merely be plausible.
        if done and job.get('reference_summary'):
            good=[r for r in rr.values() if I.flag(r['is_success'])];refsum=job['reference_summary']
            delta={}
            for k in (T,LW,'coverage_pct','path_length_m','turn_count','revisit_steps'):
                measured=avg(list(rr.values()) if k=='coverage_pct' else good,k);expected=I.N.number(refsum.get(k))
                try:I.near(measured,expected,'40M Raw summary reproduction '+key+' '+k);ok=True
                except RuntimeError:ok=False
                delta[k]={'new_raw':measured,'previous_raw':expected,'match':ok}
            successok=len(good)==int(refsum['success_n'])
            historical.append({'model':key,'success_matches':successok,'fields':delta,'all_match':successok and all(v['match'] for v in delta.values())})
    full=all(s['status']=='COMPLETE' for s in status) and len(status)==3 and not commiterrors
    baselinematch=all(h['all_match'] for h in historical)
    full=full and baselinematch
    common=set.intersection(*[{mid for mid,r in raws[key].items() if I.flag(r['is_success']) and I.flag(refs[key][mid]['is_success'])} for key in context['jobs']]) if full else set()
    if full:
        for mid in common:
            checks={(raws[k][mid]['map_sha256'],raws[k][mid]['initial_pair_id'],float(raws[k][mid]['time_straight_sec']),float(raws[k][mid]['time_turn_sec'])) for k in raws}
            I.need(len(checks)==1,'Cross-beta map/initial/time mismatch')
    table=[];own=[]
    for key,job in context['jobs'].items():
        raw=list(raws[key].values());good=[r for r in raw if I.flag(r['is_success'])]
        own.append({'model':key,'beta':job['beta'],'completed':len(raw),'target':len(job['records']),'success_n_observed':len(good),'note':'Partial rows are not final validation results' if len(raw)!=len(job['records']) else 'own-success raw',T:avg(good,T),LW:avg(good,LW)})
        rg=[raws[key][mid] for mid in sorted(common)];fg=[refs[key][mid] for mid in sorted(common)]
        rt,rl,ft,fl=avg(rg,T),avg(rg,LW),avg(fg,T),avg(fg,LW)
        ar=[a for a in allaudit if a['model']==key and a['map_id'] in common]
        table.append({'beta':job['beta'],'model':key,'nominal_m':40,'checkpoint_step':I.STEP,'N_validation':len(job['records']),
            'success_n':len(good) if full else None,'success_pct':100*len(good)/len(job['records']) if full else None,
            'common_success_n':len(common) if full else None,'raw_T_sec':rt,'raw_Lw_sec':rl,'refined_T_sec':ft,'refined_Lw_sec':fl,
            'T_reduction_pct':reduction(rt,ft),'Lw_reduction_pct':reduction(rl,fl),
            'raw_Lw_over_T_pct':100*rl/rt if rt else None,'refined_Lw_over_T_pct':100*fl/ft if ft else None,
            'changed_paths_on_common_set':sum(a['path_changed'] for a in ar) if full else None,
            'changed_paths_on_own_success':sum(a['path_changed'] for a in allaudit if a['model']==key) if full else None,
            'status':'COMPLETE' if full else 'PENDING_COMPLETE_COMPARISON'})
    paired=[]
    if full:
        for a,b in itertools.combinations(context['jobs'],2):
            ids=sorted(common);count=Counter();dr=[];dl=[]
            for mid in ids:
                ra,rb=refs[a][mid],refs[b][mid];dt=float(rb[T])-float(ra[T]);dw=float(rb[LW])-float(ra[LW]);dr.append(dt);dl.append(dw);count[outcome(dt,dw)]+=1
            paired.append({'A':a,'B':b,'common_n':len(ids),'mean_delta_T_B_minus_A':math.fsum(dr)/len(dr) if dr else None,
                 'mean_delta_Lw_B_minus_A':math.fsum(dl)/len(dl) if dl else None,**{k:count[k] for k in ['A_shorter_T_and_Lw','B_shorter_T_and_Lw','A_shorter_T_B_shorter_Lw','B_shorter_T_A_shorter_Lw','tie_in_one_or_both']}})
    I.csvwrite(report/'table_5_16_strict40m_raw_refined.csv',table)
    I.csvwrite(report/'all_betas_validation_raw_refined_per_map.csv',allrows)
    I.csvwrite(report/'refinement_audit_per_map.csv',allaudit)
    I.csvwrite(report/'checkpoint_status.csv',status);I.csvwrite(report/'own_success_raw_reference.csv',own)
    I.csvwrite(report/'beta_pairwise_refined.csv',paired)
    first=next(iter(context['jobs']))
    I.csvwrite(report/'common_success_maps.csv',[{k:raws[first][mid][k] for k in ('map_id','map_sha256','initial_pair_id','source','split','tier','pattern')} for mid in sorted(common)])
    I.write(report/'raw_reproduction_audit.json',historical)
    I.verify_protected(context['protected'])
    audit={'status':'COMPLETE' if full else 'PARTIAL','requested_pairs':sum(len(j['records']) for j in context['jobs'].values()),'completed_pairs':sum(len(x) for x in raws.values()),
        'common_success_n':len(common) if full else None,'all_source_hashes_unchanged':True,'raw_reference_means_match':baselinematch,
        'same_solver_as_chapter5':True,'new_training_updates':0,'test_or_osm_rollouts':0,'checkpoint_step':I.STEP,
        'raw_before_refinement_cached_separately':True,'success_and_order_heading_preserved':all(a.get('order_preserved') and a.get('arrival_heading_preserved') and a.get('success_preserved') for a in allaudit),
        'commit_errors':commiterrors,'worker_errors':errors or [],'not_a_global_beta_optimum_claim':True}
    I.write(report/'audit.json',audit)
    def f(v):return '산출 대기' if v is None else f'{v:,.2f}' if isinstance(v,float) else str(v)
    md=['# Table 5.16. Strict 40M β sensitivity after common refinement','',
        'COST11 · FULL · Strict(B=0) · seed42 · 실제 40,001,536 steps · Synthetic Validation 180. Train/Test/OSM 평가는 포함하지 않는다.',
        '',f"**상태: {audit['status']} · 완료 {audit['completed_pairs']}/{audit['requested_pairs']} Raw/Refined 쌍**",'',
        '| β | 성공/180 | 공통 n | Raw T (초) | Raw Lw (초) | Refined T (초) | Refined Lw (초) | T 감소(%) | Lw 감소(%) | 변경 경로(개) |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        s=f"{r['success_n']}/180" if r['success_n'] is not None else '산출 대기'
        md.append('| '+' | '.join([f(r['beta']),s]+[f(r[k]) for k in ('common_success_n','raw_T_sec','raw_Lw_sec','refined_T_sec','refined_Lw_sec','T_reduction_pct','Lw_reduction_pct','changed_paths_on_common_set')])+' |')
    md += ['', '주: 시간 평균과 감소율은 세 β가 모두 성공한 동일 map 집합에서 계산한다. 성공률은 전체 180개 기준이며 Raw/Refined 성공 여부는 동일하다. 감소율은 평균의 감소율이며 지도별 감소율의 평균이 아니다.',
        '', '최초 방문순서와 각 최초 도착 heading을 유지하며, 구간별 이동 횟수가 늘어나지 않는 T-only 연결 개선을 실제 적용했다. 개선 0은 적용 결과이며 미실행을 뜻하지 않는다. 단일 seed·세 β의 비교를 전역 최적값의 입증으로 해석하지 않는다.',
        '', '이 표는 Validation 설정 분석이다. 5.1의 Test720/OSM500 결과와 합산하거나 해당 표를 자동 대체하지 않는다.']
    I.atomic(report/'Table_5_16_KO.md','\n'.join(md)+'\n')
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Strict40M β Refined</title><style>body{font:15px sans-serif;line-height:1.65;max-width:1250px;margin:35px auto;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;border:1px solid #bbb;text-align:right}th:first-child,td:first-child{text-align:left}pre{white-space:pre-wrap}.wide{overflow:auto}</style><h1>Strict40M β sensitivity — Validation Raw / Refined</h1>'
    page+='<p><b>'+audit['status']+'</b> · COST11 · β=0/0.5/1 · Validation only · no training · no Test/OSM</p><div class="wide">'
    page+=render_table(table,['beta','success_n','N_validation','common_success_n','raw_T_sec','raw_Lw_sec','refined_T_sec','refined_Lw_sec','T_reduction_pct','Lw_reduction_pct','changed_paths_on_common_set'])+'</div>'
    page+='<h2>기록과 검증</h2>'+render_table(status,['model','completed_raw_refined_pairs','target_maps','status'])
    page+='<pre>'+html.escape(json.dumps(audit,ensure_ascii=False,indent=2))+'</pre></html>';I.atomic(out/'index.html',page)
    make_bundle(context,identity,raws,refs,allaudit)
    print('[REPORT]',out/'index.html',flush=True);print('[STATUS]',audit['status'],str(audit['completed_pairs'])+'/'+str(audit['requested_pairs']),flush=True)
    print('[ZIP]',out/'TABLE516_STRICT40M_BETA_REFINED_REVIEW.zip',flush=True)
    return audit


def make_bundle(context,identity,raws,refs,audits):
    out=Path(context['out']);zp=out/'TABLE516_STRICT40M_BETA_REFINED_REVIEW.zip';tmp=zp.with_name(zp.name+'.writing')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted((out/'report').glob('*')):
            if p.is_file():z.write(p,'report/'+p.name)
        for n in ('protocol.json','input_audit.json','index.html','validation_manifest.csv'):
            if (out/n).is_file():z.write(out/n,n)
        for key,j in context['jobs'].items():
            for n in ('complete.json','raw_per_map.csv','refined_per_map.csv','audit_per_map.csv'):
                p=out/'methods'/key/n
                if p.is_file():z.write(p,'methods/'+key+'/'+n)
            z.writestr('provenance/'+key+'.json',json.dumps(I.safe({k:j[k] for k in ('checkpoint','spec','settings','software','sources','feature_sha256','historical_provenance','reference_summary')}),ensure_ascii=False,indent=2))
            # Two full path pairs per beta: first changed and first unchanged when available.
            candidates=[a for a in audits if a['model']==key];picked=[]
            for flag in (True,False):
                opt=next((a for a in candidates if a['path_changed']==flag),None)
                if opt:picked.append(opt['map_id'])
            for mid in list(raws[key]):
                if len(picked)>=2:break
                if mid not in picked:picked.append(mid)
            for mid in picked:
                d=out/'methods'/key/'maps'/I.map_key({'map_id':mid})
                for p in sorted(d.glob('*')):
                    if p.is_file() and '.writing' not in p.name:z.write(p,'samples/'+key+'/'+d.name+'/'+p.name)
        for p in sorted((out/'logs').glob('*.log')) if (out/'logs').is_dir() else []:
            b=p.read_bytes();z.writestr('logs/'+p.name,b[-2_000_000:])
        for n in context['code_sha256']:z.write(I.PACKAGE/n,'code/'+n)
        for n in ('README_KO.md','VALIDATION.md'):
            if (I.PACKAGE/n).is_file():z.write(I.PACKAGE/n,'code/'+n)
    with zipfile.ZipFile(tmp) as z:I.need(z.testzip() is None,'Review ZIP CRC failure')
    os.replace(tmp,zp)

"""Reports use matched successful maps; failures and missing outputs stay distinct."""
from __future__ import annotations
import itertools, html, math, time, zipfile
from collections import Counter,defaultdict
from pathlib import Path
import b6ref_io as I
import b6ref_native as W
import ch5_common as C
T,LW=C.T,C.LW
METRICS=(T,LW,'turn_count','revisit_steps','zero_transit_steps','path_length_m','nrg_detour_ratio',
         'allowed_actions_mean','single_action_mask_ratio','priority_arrival_25_sec','priority_arrival_50_sec','priority_arrival_75_sec','priority_arrival_90_sec')


def avg(rows,key):
    xs=[I.number(r.get(key)) for r in rows]
    return math.fsum(v for v in xs if v is not None)/sum(v is not None for v in xs) if any(v is not None for v in xs) else None


def outcome(dt,dl):
    # dt=B-A and dl=B-A. Counts refer to maps, not seconds or percentages.
    eps=1e-6
    if abs(dt)<=eps or abs(dl)<=eps:return 'tie_in_one_or_both_metrics'
    if dt>0 and dl>0:return 'A_shorter_T_and_Lw'
    if dt<0 and dl<0:return 'B_shorter_T_and_Lw'
    if dt>0:return 'A_shorter_T_B_shorter_Lw'
    return 'B_shorter_T_A_shorter_Lw'


def membership(context):
    rs=next(iter(context['jobs'].values()))['records']
    ids={k:{r['map_id'] for r in rs if I.domain(r)==k} for k in ('Validation180','Synthetic720','OSM500')}
    if ids['Synthetic720'] or ids['OSM500']:ids['CombinedTest1220']=ids['Synthetic720']|ids['OSM500']
    return {k:v for k,v in ids.items() if v}


def baseline_pairs(context):
    if context.get('fixture_only_not_paper_results'):return {},[]
    rr=I.gzrows(I.PACKAGE/'reference/previous_B0124_classical_Raw_Refined.csv.gz')
    ids={r['map_id'] for r in next(iter(context['jobs'].values()))['records']}
    anchor=next(iter(context['jobs'].values()))['historical_rows'];pairs=defaultdict(lambda:{'raw':{},'refined':{}})
    for r in rr:
        if r['map_id'] not in ids:continue
        mid=r['map_id'];old=anchor[mid]
        for k in ('map_sha256','initial_pair_id','source','split','pattern','tier'):
            I.need(str(r[k])==str(old[k]),'Prior Refined baseline mismatches B124 map/initial: '+k+' '+mid)
        for k in ('start_r','start_c','start_heading','time_straight_sec','time_turn_sec','active_cells'):
            I.near(r.get(k),old.get(k),'Prior baseline physical setting '+k)
        if r['method'] in ('B0','B1','B2','B4','B6'):
            I.need(int(r['checkpoint_step'])==I.STEP and I.number(r['beta'])==.5 and int(r['effective_B'])==int(r['method'][1:]),'Not a verified beta=.5 40M budget reference')
        st=r['reference_stage'];I.need(st in ('raw','refined'),'Unknown prior reference stage');I.need(mid not in pairs[r['method']][st],'Duplicate reference baseline row')
        pairs[r['method']][st][mid]=r
    for method,pair in pairs.items():
        I.need(set(pair['raw'])==set(pair['refined']),'Unpaired prior Raw/Refined baseline: '+method)
        for mid,r in pair['raw'].items():
            f=pair['refined'][mid]
            I.need(I.flag(r['is_success'])==I.flag(f['is_success']),'Prior baseline success changed')
            I.need(float(f[T])<=float(r[T])+2e-6,'Prior baseline time increased')
            if I.flag(r['is_success']):I.need(float(f[LW])<=float(r[LW])+2e-6,'Prior baseline latency increased')
    return dict(pairs),I.read(I.PACKAGE/'reference/PROVENANCE.json')


def table_html(rows):
    if not rows:return '<p>자료 없음 / 아직 완료되지 않음.</p>'
    fields=list(rows[0]);body='<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'
    for r in rows:
        body+='<tr>'+''.join('<td>'+html.escape('—' if r.get(k) is None else f'{r[k]:.6g}' if isinstance(r.get(k),float) else str(r.get(k,'')))+'</td>' for k in fields)+'</tr>'
    return body+'</tbody></table>'


def build_report(context,identity,worker_errors=None):
    out=Path(context['out']);rep=out/'report';rep.mkdir(parents=True,exist_ok=True)
    pairs={};auditrows=[];states=[];integrity=[];completed=0;raw_only=0
    domains=membership(context);rawrows=[];refrows=[]
    for key,j in context['jobs'].items():
        raw={};ref={};na=0
        for rec in j['records']:
            d=out/'methods'/key/'maps'/I.map_key(rec)
            try:
                rs=W.verify_result(d,identity,rec)
                if rs is not None:
                    r,f,a=rs;raw[rec['map_id']]=r;ref[rec['map_id']]=f;auditrows.append(a);completed+=1;na+=1
                    I.compare_metrics(r,j['historical_rows'][rec['map_id']])
                else:
                    rc=I.receipt_read(d,'raw_complete.json',identity)
                    if rc is not None:
                        r=I.read(d/'raw_metrics.json');I.compare_metrics(r,j['historical_rows'][rec['map_id']]);raw[rec['map_id']]=r;raw_only+=1
            except Exception as exc:
                integrity.append({'method':key,'map_id':rec['map_id'],'error':str(exc)})
        pairs[key]={'raw':raw,'refined':ref}
        states.append(dict(method=key,N_expected=len(j['records']),raw_saved=len(raw),refined_completed=len(ref),
            status='COMPLETE' if len(ref)==len(j['records']) and not any(e['method']==key for e in integrity) else 'ERROR' if any(e['method']==key for e in integrity) else 'PARTIAL'))
        rawrows+=list(raw.values());refrows+=list(ref.values())
    baseline,baseline_prov=baseline_pairs(context);pairs.update(baseline)
    expected=sum(len(j['records']) for j in context['jobs'].values())
    for p,h in context['protected'].items():
        if not Path(p).is_file() or I.sha(p)!=h:integrity.append({'input_file':p,'error':'source_modified_or_missing'})
    state='ERROR' if integrity or worker_errors else 'COMPLETE' if completed==expected else 'PARTIAL'
    # Explicit per-domain counts; means never include unsuccessful elapsed times.
    own=[];effects=[]
    for dn,ids in domains.items():
        for method,pair in pairs.items():
            full=ids<=set(pair['raw']) and ids<=set(pair['refined'])
            if not full:
                if method in context['jobs']:
                    own.append(dict(domain=dn,method=method,stage='refined',status='PARTIAL',N_total=len(ids),n_processed=len(ids&set(pair['refined'])),success_n=None,success_pct=None,mean_success_n=None,mean_T_sec=None,mean_Lw_sec=None))
                continue
            for stage in ('raw','refined'):
                rr=[pair[stage][i] for i in sorted(ids)];succ=[r for r in rr if I.flag(r['is_success'])]
                o=dict(domain=dn,method=method,stage=stage,status='COMPLETE',N_total=len(ids),n_processed=len(rr),success_n=len(succ),
                    success_pct=100*len(succ)/len(ids),mean_success_n=len(succ),mean_T_sec=avg(succ,T),mean_Lw_sec=avg(succ,LW),
                    coverage_all_maps_pct=avg(rr,'coverage_pct'))
                o.update({k:avg(succ,k) for k in METRICS});own.append(o)
            if method in context['jobs']:
                s={i for i in ids if I.flag(pair['raw'][i]['is_success'])};r=[pair['raw'][i] for i in sorted(s)];f=[pair['refined'][i] for i in sorted(s)]
                ar,af,lr,lf=avg(r,T),avg(f,T),avg(r,LW),avg(f,LW)
                e=dict(domain=dn,method=method,N_total=len(ids),success_n=len(s),common_raw_refined_n=len(s),
                    raw_T_sec=ar,refined_T_sec=af,raw_Lw_sec=lr,refined_Lw_sec=lf,
                    T_reduction_pct=100*(ar-af)/ar if ar else None,Lw_reduction_pct=100*(lr-lf)/lr if lr else None,
                    changed_paths_on_success=sum(I.flag(pair['refined'][i]['path_changed']) for i in s),
                    failures_preserved=len(ids)-len(s))
                effects.append(e)
    common=[];common_ids=[]
    groups=[('B0_B6',['B0','B6']),('B0_B1_B2_B4_B6',['B0','B1','B2','B4','B6']),
            ('B0_B1_B2_B4_B6_BCD_WLG',['B0','B1','B2','B4','B6','BCD','WLG']),
            ('B0_B1_B2_B4_B6_BCD_WLG_STC',['B0','B1','B2','B4','B6','BCD','WLG','STC']),
            ('MAIN5_B0_B6_BCD_WLG_STC',['B0','B6','BCD','WLG','STC'])]
    # all stages use exactly the same method/group membership for direct Raw/Refined comparisons.
    for dn,ids in domains.items():
        for gn,methods in groups:
            if not all(m in pairs and ids<=set(pairs[m]['refined']) and ids<=set(pairs[m]['raw']) for m in methods):continue
            ok={i for i in ids if all(I.flag(pairs[m]['refined'][i]['is_success']) for m in methods)}
            common_ids.extend(dict(domain=dn,comparison_group=gn,map_id=i) for i in sorted(ok))
            for st in ('raw','refined'):
                for m in methods:
                    rr=[pairs[m][st][i] for i in sorted(ok)];s=sum(I.flag(pairs[m][st][i]['is_success']) for i in ids)
                    row=dict(domain=dn,comparison_group=gn,method=m,stage=st,N_total=len(ids),success_n=s,success_pct=100*s/len(ids),common_success_n=len(ok),
                        T_sec=avg(rr,T),Lw_sec=avg(rr,LW),Lw_over_T_pct=100*avg(rr,LW)/avg(rr,T) if rr and avg(rr,T) else None)
                    row.update({k:avg(rr,k) for k in METRICS});common.append(row)
    paired=[];paired_maps=[];tierpatterns=[]
    for dn,ids in domains.items():
        if dn=='CombinedTest1220':subgroups=[('all','all',ids)]
        else:
            subgroups=[('all','all',ids)]
            recs={r['map_id']:r for r in next(iter(context['jobs'].values()))['records']}
            if dn!='OSM500':
                for field in ('tier','pattern'):
                    for val in sorted({recs[i][field] for i in ids}):
                        subgroups.append((field,val,{i for i in ids if recs[i][field]==val}))
        for group,val,groupids in subgroups:
            bm=[m for m in ('B0','B1','B2','B4','B6') if m in pairs and groupids<=set(pairs[m]['refined'])]
            if len(bm)==5:
                ok={i for i in groupids if all(I.flag(pairs[m]['refined'][i]['is_success']) for m in bm)}
                for m in bm:
                    rr=[pairs[m]['refined'][i] for i in sorted(ok)]
                    tierpatterns.append(dict(domain=dn,group=group,group_value=val,method=m,compared_methods=','.join(bm),N_total=len(groupids),
                        success_n=sum(I.flag(pairs[m]['refined'][i]['is_success']) for i in groupids),common_success_n=len(ok),T_sec=avg(rr,T),Lw_sec=avg(rr,LW)))
            # B0/B124 head-to-head and model-free baselines (domain totals only).
            pm=bm+([m for m in ('BCD','WLG','STC') if m in pairs and groupids<=set(pairs[m]['refined'])] if group=='all' else [])
            for a,b in itertools.combinations(pm,2):
                if a not in ('B0','B1','B2','B4','B6') and b not in ('B0','B1','B2','B4','B6'):continue
                ra,rb=pairs[a]['refined'],pairs[b]['refined'];ok={i for i in groupids if I.flag(ra[i]['is_success']) and I.flag(rb[i]['is_success'])}
                counts=Counter();
                for mid in sorted(ok):
                    dt=float(rb[mid][T])-float(ra[mid][T]);dl=float(rb[mid][LW])-float(ra[mid][LW]);oc=outcome(dt,dl);counts[oc]+=1
                    if group=='all' and dn!='CombinedTest1220':paired_maps.append(dict(domain=dn,A=a,B=b,map_id=mid,delta_T_B_minus_A_sec=dt,delta_Lw_B_minus_A_sec=dl,outcome=oc))
                ar,br=[ra[i] for i in sorted(ok)],[rb[i] for i in sorted(ok)]
                q=dict(domain=dn,group=group,group_value=val,A=a,B=b,N_total=len(groupids),common_success_n=len(ok),
                    A_only_success_n=sum(I.flag(ra[i]['is_success']) and not I.flag(rb[i]['is_success']) for i in groupids),
                    B_only_success_n=sum(I.flag(rb[i]['is_success']) and not I.flag(ra[i]['is_success']) for i in groupids),
                    both_failed_n=sum(not I.flag(ra[i]['is_success']) and not I.flag(rb[i]['is_success']) for i in groupids),
                    A_T_sec=avg(ar,T),B_T_sec=avg(br,T),A_Lw_sec=avg(ar,LW),B_Lw_sec=avg(br,LW))
                for oc in ('A_shorter_T_and_Lw','B_shorter_T_and_Lw','A_shorter_T_B_shorter_Lw','B_shorter_T_A_shorter_Lw','tie_in_one_or_both_metrics'):
                    q[oc+'_maps']=counts[oc]
                I.need(sum(counts.values())==len(ok),'Paired outcome accounting')
                paired.append(q)
    criteria=[]
    for dn,ids in domains.items():
        if not all(m in pairs and ids<=set(pairs[m]['refined']) for m in ('BCD','WLG')):continue
        for m in ('B0','B1','B2','B4','B6'):
            if m not in pairs or not ids<=set(pairs[m]['refined']):continue
            counts=Counter()
            for i in ids:
                p=pairs[m]['refined'][i];bd=pairs['BCD']['refined'][i];wl=pairs['WLG']['refined'][i]
                if not all(I.flag(r['is_success']) for r in (p,bd,wl)):continue
                pt,pl=float(p[T]),float(p[LW]);eps=1e-6
                for label,t,l in [('A',bd[T],bd[LW]),('B',wl[T],wl[LW]),('C',wl[T],bd[LW]),('D',bd[T],wl[LW])]:
                    counts[label]+=int(pt<float(t)-eps and pl<float(l)-eps)
            criteria.append(dict(domain=dn,method=m,N_total=len(ids),baseline_stage='BCD_Refined_WLG_Refined',**{k+'_maps':counts[k] for k in 'ABCD'},**{k+'_pct':100*counts[k]/len(ids) for k in 'ABCD'}))
    I.csvwrite(rep/'method_status.csv',states);I.csvwrite(rep/'all_B6_raw_refined_per_map.csv',rawrows+refrows)
    I.csvwrite(rep/'refinement_audit_per_map.csv',auditrows);I.csvwrite(rep/'overall_own_success.csv',own)
    I.csvwrite(rep/'raw_vs_refined_B6.csv',effects);I.csvwrite(rep/'common_success_raw_refined.csv',common)
    I.csvwrite(rep/'common_success_map_ids.csv',common_ids);I.csvwrite(rep/'paired_refined_summary.csv',paired)
    I.csvwrite(rep/'paired_refined_map_outcomes.csv',paired_maps);I.csvwrite(rep/'tier_pattern_refined_common.csv',tierpatterns)
    I.csvwrite(rep/'criteria_ABCD_REFINED_baselines.csv',criteria)
    for dn,file in [('Validation180','budget_validation180_refined.csv'),('Synthetic720','budget_synthetic720_refined.csv'),('OSM500','budget_osm500_refined.csv'),('CombinedTest1220','budget_combined1220_refined.csv')]:
        I.csvwrite(rep/file,[r for r in common if r['domain']==dn and r['stage']=='refined'])
    I.csvwrite(rep/'reused_reference_raw_refined_per_map.csv',[r for p in baseline.values() for st in ('raw','refined') for r in p[st].values()])
    audit=dict(status=state,version=I.VERSION,protocol_identity=identity,expected_pairs=expected,completed_pairs=completed,
        raw_only_committed_pairs=raw_only,counts_include_validation_but_no_ALL1400_performance_mean=True,
        original_raw_reproduction='each completed pair matched uploaded B6 exact40M per-map metrics; unprocessed maps are not claimed verified',
        source_files_unchanged=not any('input_file' in e for e in integrity),integrity_errors=integrity,worker_errors=worker_errors or [],
        checkpoint_step=I.STEP,optimizer_updates=0,training=False,baseline_policy_reinference=False,
        previous_refinement_sources=baseline_prov,method_status=states,
        verification_scope='Server native inference/replay for new B6; original uploaded prior Refined baselines reused as per-map data, not reoptimized',
        common_engine_budget_extension='No new solver change: B124 common driver, SegmentGraph A*, native guard and support copied byte-identically')
    I.write(rep/'audit.json',audit)
    # Method summaries are rebuilt from authenticated per-map receipts, not a
    # worker's most recent CSV. They are stable when --workers changes on resume.
    base=out/'methods'/'B6';base.mkdir(parents=True,exist_ok=True)
    I.csvwrite(base/'raw_per_map.csv',rawrows);I.csvwrite(base/'refined_per_map.csv',refrows);I.csvwrite(base/'audit_per_map.csv',auditrows)
    if state=='COMPLETE':
        I.write(base/'complete.json',dict(identity=identity,key='B6',n_maps=completed,checkpoint_step=I.STEP,
            checkpoint_sha256=context['jobs']['B6']['checkpoint']['sha256'],
            files={n:I.sha(base/n) for n in ('raw_per_map.csv','refined_per_map.csv','audit_per_map.csv')}))
    elif (base/'complete.json').exists():
        # Retain evidence of an earlier completion without leaving a stale success marker.
        (base/'complete.json').replace(base/'previous_complete_receipt.json')
    lines=['# B6·β=0.5·40M 연결 개선 및 B=0·1·2·4·6 비교','',f'상태: **{state}**. B6 완료 {completed}/{expected} 경로쌍.',
        '', 'B6만 새로 처리한다. 기존20M→총40M 이어학습의 정확한 40,001,536-step checkpoint를 사용하며, 5M·20M·38M이나 새 초기화 B6와 혼합하지 않는다.',
        '새 학습은 없다. 저장 행동이 없는 map은 같은 checkpoint의 CPU masked argmax로 경로를 생성하고 기존 Raw와 대조한 뒤 별도 확정한다. 확정한 Raw와 완료 Refined는 재실행 시 재사용한다.',
        '원래 B=6 마스크, 최초 방문순서 및 최초 도착 heading을 유지하고 각 연결구간의 이동 횟수를 늘리지 않는 T-only 연결 개선이다. B124에 사용한 공통 엔진을 그대로 적용한다. 실패 경로는 그대로 보존한다.',
        'Validation 180 / Synthetic Test 720 / OSM Test 500은 별도 집합이다. Combined Test는 720+500이며 Validation을 더하지 않는다. 성공 수는 전체 평가집합, 시간 평균은 명시된 성공집합을 사용한다.',
        '이전 B0·B1·B2·B4의 40M과 BCD·WLG·STC의 Raw/Refined map별 결과를 재사용한다. 참조 모델은 새로 추론하거나 재최적화하지 않는다. Classical의 Validation Refined는 없으므로 그 비교표를 생성하지 않는다.',
        '공통 평균은 B6까지 포함한 교집합에서 모든 방법을 다시 계산한다. comparison_group과 common_success_n을 반드시 함께 읽는다. 전체 성공 수와 공통 성공 수, 자체 성공 평균은 다르다.',
        '표의 A/B 또는 B0/B6 두 지표 우세 횟수는 초가 아니라 지도 수(개)이다. 한 방법의 T가 짧고 다른 방법의 Lw가 짧으면 상충관계로 나눈다. 동률 허용오차는 1e-6초다.',
        'A–D 달성은 같은 지도에서 BCD-Refined·WLG-Refined 기준과 비교한다. 원래 Raw 기준선 달성 횟수를 복사하지 않는다. A–D는 중복 가능하므로 합산하지 않는다.',
        '이번 결과도 단일 seed·개발 중 Test 검토 이력의 설정 비교이다. B6는 기존 학습 이력을 이어받았으며 전역 최적 B, 통계적 유의성 또는 새로운 untouched Test를 주장하지 않는다.',
        '기존 KMZ manifest는 이 코드가 수정하지 않는다. 경로는 methods/B6/maps/<id>/result.json과 raw_trace.npz/refined_trace.npz에 저장한다. KMZ 생성·기체 호환성 검증은 수행하지 않는다.',
        '', '주요 파일: raw_vs_refined_B6.csv, common_success_raw_refined.csv, budget_*_refined.csv, paired_refined_summary.csv, tier_pattern_refined_common.csv, criteria_ABCD_REFINED_baselines.csv.',
        '', '## B6 Raw → Refined', table_html(effects), '', '## 공통 성공집합 Refined',table_html([r for r in common if r['stage']=='refined'])]
    I.U.write_text(rep/'B6_Refinement_KO.md','\n'.join(lines)+'\n')
    ht='<!doctype html><html lang="ko"><meta charset="utf-8"><title>B6 40M refinement</title><style>body{font:15px system-ui;margin:30px;line-height:1.6}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #aaa;padding:5px}.scroll{overflow:auto}</style>'
    ht+=f'<h1>B6 40M Raw → Refined: {state}</h1><p>{completed}/{expected} pairs. CPU only. Validation / Synthetic Test / OSM are separate.</p>'
    for title,rr in [('완료 상태',states),('B6 개선 전후',effects),('공통 성공집합 Refined',[r for r in common if r['stage']=='refined']),('지도 수 기반 직접 비교',[r for r in paired if r['group']=='all']),('A–D (Refined 기준선)',criteria)]:
        ht+='<h2>'+title+'</h2><div class="scroll">'+table_html(rr)+'</div>'
    I.U.write_text(out/'index.html',ht+'</html>')
    bundle(out,context)
    print('[REPORT]',state,f'{completed}/{expected}','[ZIP]',out/'B6_40M_REFINED_REVIEW.zip',flush=True)
    return audit


def bundle(out,context):
    out=Path(out);target=out/'B6_40M_REFINED_REVIEW.zip';tmp=target.with_suffix('.writing.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n in ('protocol.json','input_audit.json','evaluation_manifest.csv','index.html'):
            p=out/n
            if p.is_file():z.write(p,n)
        for p in sorted((out/'report').glob('*')):
            if p.is_file():z.write(p,str(p.relative_to(out)))
        for key,j in context['jobs'].items():
            base=out/'methods'/key
            for n in ('complete.json','raw_per_map.csv','refined_per_map.csv','audit_per_map.csv','budget_guard_audit.json'):
                if (base/n).is_file():z.write(base/n,str((base/n).relative_to(out)))
            # Up to two maps per domain, preferring one changed path and one failure.
            bins=defaultdict(list)
            for rec in j['records']:
                p=base/'maps'/I.map_key(rec)
                if (p/'complete.json').is_file():
                    a=I.read(p/'refinement_audit.json');r=I.read(p/'raw_metrics.json')
                    bins[I.domain(rec)].append((not bool(a['path_changed']),I.flag(r['is_success']),p))
            for domain,items in bins.items():
                items.sort();chosen=[]
                for predicate in (lambda x:not x[0],lambda x:not x[1],lambda x:True):
                    found=next((x[2] for x in items if predicate(x) and x[2] not in chosen),None)
                    if found is not None and len(chosen)<2:chosen.append(found)
                for p in chosen:
                    for f in sorted(p.iterdir()):
                        if f.is_file() and '.writing.' not in f.name and not f.name.startswith('.'):z.write(f,str(Path('samples')/key/domain/p.name/f.name))
            err=out/'logs'/(key+'_error.json')
            if err.is_file():z.write(err,str(err.relative_to(out)))
        for n in ('reference/PROVENANCE.json','reference/checkpoint_targets.json','PACKAGE_SHA256.json'):
            if (I.PACKAGE/n).is_file():z.write(I.PACKAGE/n,'code/'+n)
        for p in list(I.PACKAGE.glob('*.py'))+list((I.PACKAGE/'ch5_vendor').glob('*.py')):
            z.write(p,'code/'+str(p.relative_to(I.PACKAGE)))
        for p in sorted((out/'logs').glob('*_error.json')):
            z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(tmp) as z:I.need(z.testzip() is None,'ZIP CRC failed')
    tmp.replace(target)

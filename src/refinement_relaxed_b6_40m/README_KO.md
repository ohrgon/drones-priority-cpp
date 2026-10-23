# B6·β=0.5·40M 연결 개선 + B=0·1·2·4·6 비교

## 1. 이번에 처리하는 것

완료된 **COST11·FULL·B=6·β=0.5·seed=42**의 정확한 **40,001,536-step** 모델을 사용한다.
기존 20,004,864-step 모델을 총 40M까지 이어학습한 출력이며, 새 가중치에서 시작한 다른 B6 모델이나
5M·20M·38M·best checkpoint를 대신 사용하지 않는다.

기본 입력 폴더:

```text
chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42/
```

정확한 가중치 SHA-256:

```text
eea33db123316cb34f6be101135c1d0101614660bce6b9ae66a0091bc1cdde6d
```

기본 처리 범위는 **Validation 180 + Synthetic Test 720 + OSM Test 500 = 1,400개 Raw/Refined 쌍**이다.
모든 중간 checkpoint를 다시 평가하지 않고 최종 40M만 사용한다. 학습·optimizer update는 수행하지 않는다.

## 2. 실행

ZIP을 `uav_chapter4_suite_v2`에 넣고 다음을 실행한다.

```bash
cd <ORIGINAL_WORKSPACE>/uav_chapter4_suite_v2
conda activate dqn_env
unzip -n refinement_relaxed_b6_40m.zip
python -u refinement_relaxed_b6_40m/run_b6_40m_refined.py \
  --root . --phase all --workers 3
```

GPU는 사용하지 않는다. 단일 B6 모델의 대상 지도를 CPU 프로세스 3개에 나누어 처리한다.
각 프로세스에서 원래 평가와 같은 PyTorch thread 설정(4)을 사용한다. NTERM/NCOV 등과 CPU 자원이
겹쳐 부담이 크면 `--workers 1`로 줄인다. 기존 학습 프로세스를 중단하거나 기존 출력에 쓰지 않는다.

입력만 점검하려면:

```bash
python -u refinement_relaxed_b6_40m/run_b6_40m_refined.py --root . --phase check
```

`check`는 파일·메타데이터·해시·비교집합 점검이다. 실제 모델 로드/추론/연결 개선을 수행했다는 뜻이 아니다.

원본 폴더의 위치가 달라졌을 때에는 실제 완료된 출력 폴더를 `--training-root PATH`로 지정한다.
그 경로 안에는 `plan.json`, 해당 최종 run의 설정/등록정보, 실제 40M 가중치가 있어야 한다.
확인되지 않은 모델로 자동 대체하거나 새로 학습하지 않는다. 입력 불일치는 진단 ZIP을 만들고 중단한다.

## 3. 첫 실행에서는 Raw 행동 경로를 복원한다

기존 B6 학습기의 평가 코드는 map별 수치 JSON/CSV를 저장하지만 전체 행동 시퀀스를 저장하지 않는다.
이 때문에 첫 실행은 아래 순서를 수행한다.

1. 원래 checkpoint와 같은 CPU masked-argmax 방식으로 Raw 행동 경로 생성.
2. 업로드된 정확한 40M 기록과 지도 hash·초기위치·heading·성공/실패·T·Lw·step·회전·재방문·우회·우선순위 도달시간 대조.
3. 일치한 Raw를 `raw_complete.json`과 함께 별도 확정 저장.
4. 공통 연결 개선을 수행하고 원래 환경에서 재생·검증.
5. Refined 경로, 구간별 기록, 보존조건 및 파일 해시를 확정 저장.

이 CPU 추론은 재학습이 아니다. 과거 Raw 수치를 수정하거나 결과가 다른데도 진행하지 않는다.
입력 차이나 재현 불일치는 오류로 남기며, Raw 값을 Refined 칸에 복사하지 않는다.

## 4. 연결 개선 조건

`ch5_common.py`, `ch5_vendor/segment_graph_original.py`, `b124_core_guard.py`,
`b124_native_support.py`는 기존 B1·B2·B4 연결 개선 패키지와 **byte-identical**이다.

**최초 방문순서, 각 최초 방문 셀의 도착 heading, 원래 B=6 예산의 소진/재충전 및 허용 행동 규칙**을 유지하고,
각 연결구간의 이동 횟수를 늘리지 않으면서 시간만 줄이는 방식이다.
시간 동률이면 기존 연결구간을 유지한다. 전체 방문순서를 다시 최적화하거나 비행곡선을 단순화하는 절차가 아니다.
실패 경로는 최적화하지 않고 같은 행동과 실패 상태를 보존한다.

원래 환경/CNN은 서버의 `_runtime/relaxed`를 우선 사용한다. 없을 때에는 패키지의 `native_original` 사본이
원래 plan에 등록된 각 파일의 SHA와 완전히 일치하는 경우에만 사용한다. 다른 보상/모델 구현으로 대체하는 것이 아니다.
원래 B6 run에는 `encoder_ablation` 키가 없다. 그 키를 요구하지 않고 실제 원본 CNN 파일 hash와 모델 로드 후
`audit_cnn3_model`로 확인한다.

## 5. 이전 결과의 재사용

`B124_40M_REFINED_REVIEW.zip`에서 이미 확인한 다음 map별 Raw/Refined 기록을 패키지에 압축하여 포함했다.

- **B0·B1·B2·B4:** β=0.5, 최종40M, Validation 180 및 Test 1,220.
- **BCD·WLG·STC:** 실제 연결 개선을 수행한 Test 1,220. Classical Validation Refined는 포함되지 않아 그 비교는 만들지 않는다.

이들의 정책을 다시 불러오거나 경로를 다시 최적화하지 않는다. B6의 지도 hash·초기조건·시간모델과 대응 확인 후 사용한다.
기존 B6 5M Refined 수치는 이 40M budget 표에 섞지 않는다.

주요 비교집합:

- `B0_B6`: Strict40M vs Relaxed40M 직접 비교.
- `B0_B1_B2_B4_B6`: 다섯 PPO의 같은40M budget 비교.
- `B0_B1_B2_B4_B6_BCD_WLG`: 위 다섯 PPO + BCD/WLG.
- `B0_B1_B2_B4_B6_BCD_WLG_STC`: STC까지 포함.
- `MAIN5_B0_B6_BCD_WLG_STC`: 제5장 주 비교를 위한 다섯 방법.

각 비교의 공통 성공집합을 **B6까지 포함해 새로 계산**하고 모든 방법의 평균을 다시 낸다.
숫자가 같은 n이어도 지도 목록까지 같은 것은 아니므로 `comparison_group` 및 `common_success_map_ids.csv`와 함께 읽는다.

성공률의 분모는 전체 대상 수이고, 시간 평균은 표에 명시한 자체 또는 공통 성공 수이다.
Validation, Synthetic, OSM은 분리하며 Combined Test는 Synthetic+OSM만 합친다.
A–D는 BCD-Refined/WLG-Refined를 기준으로 계산한다. 실패/동률은 달성에 포함하지 않고, 중복 가능한 A–D 수를 합산하지 않는다.
같은 지도에서 T와 Lw 중 어느 방법이 작은지 나눈 숫자는 **지도 수(개)**이며 시간 단위가 아니다.

## 6. 출력과 재개

```text
refinement_b6_common/
├── protocol.json
├── input_audit.json
├── evaluation_manifest.csv
├── index.html
├── methods/B6/
│   ├── raw_per_map.csv
│   ├── refined_per_map.csv
│   ├── audit_per_map.csv
│   ├── workers/
│   └── maps/<map_id_hash>/
│       ├── raw_complete.json
│       ├── raw_metrics.json
│       ├── raw_trace.npz
│       ├── refined_metrics.json
│       ├── refined_trace.npz
│       ├── refinement_audit.json
│       ├── segments.csv
│       ├── result.json
│       └── complete.json
├── logs/B6_part1.log  (part2·3 동일)
├── report/
│   ├── audit.json
│   ├── all_B6_raw_refined_per_map.csv
│   ├── raw_vs_refined_B6.csv
│   ├── budget_validation180_refined.csv
│   ├── budget_synthetic720_refined.csv
│   ├── budget_osm500_refined.csv
│   ├── budget_combined1220_refined.csv
│   ├── common_success_raw_refined.csv
│   ├── common_success_map_ids.csv
│   ├── paired_refined_summary.csv
│   ├── paired_refined_map_outcomes.csv
│   ├── tier_pattern_refined_common.csv
│   ├── criteria_ABCD_REFINED_baselines.csv
│   ├── reused_reference_raw_refined_per_map.csv
│   └── B6_Refinement_KO.md
└── B6_40M_REFINED_REVIEW.zip
```

로그:

```bash
tail -n 40 -f refinement_b6_common/logs/B6_part1.log
```

중간 중단 후 **같은 명령을 다시 실행**하면 된다. Raw까지 확정된 지도는 모델 추론 없이 후처리부터 이어간다.
Refined까지 완료된 지도는 둘 다 생략한다. `--workers`를 3에서 1로 변경해도 지도별 cache를 재사용한다.
손상된 확정 cache는 없는 것으로 간주하여 덮어쓰지 않고 오류로 처리한다.

보고서/ZIP만 다시 만들기:

```bash
python -u refinement_relaxed_b6_40m/run_b6_40m_refined.py --root . --phase report
```

범위를 제한할 때는 `--scope validation` 또는 `--scope test`를 사용한다.
기본 출력 폴더에 각각 `_validation` 또는 `_test` 접미사를 추가하므로 다른 범위의 실행과 혼합하지 않는다.
그 경우 보고서 명령에도 같은 scope를 지정하거나 `--out`을 명시한다.

`--cached-only`는 **이 새 출력**에 확정 저장된 Raw가 있을 때만 후처리한다. 원래 수치 CSV만으로 후처리할 수 있다는 뜻이 아니다.

이전 KMZ manifest, waypoint, 기체 파일은 바꾸지 않는다. `result.json` 및 trace는 나중의 경로 추출을 위한 자료이며,
이 패키지에서 KMZ 생성이나 실제 비행 실행·검증을 수행하는 것은 아니다.

## 7. 검증 범위

로컬 27개 단위·통합 검사를 수행했다. 정확한 업로드 B6 40M Raw 1,400행과 이전 참조 18,520행의 메타데이터 대응,
B6 예산·후처리의 실제 개선 발생·최초 방문/heading 보존·실패 보존·손상 cache 거부·중단 후 재개·CPU 병렬 처리·분모를 검사했다.
source_context 연결 검사는 실제 업로드 메타데이터와 **가중치 내용의 mock**을 쓴 파일 경로 테스트이며 실제 모델 로드 검사가 아니다.

기존 업로드 **B6 5M의 실제 경로 샘플 2개**에서는 같은 엔진으로 계산한 Refined 지표가 이전 값과 일치하는지 추가 확인했다.
이 두 샘플을 B6 40M 결과로 쓰지 않는다.

로컬 환경에는 사용자의 40M 모델 가중치·원본 지도 전체·SB3 환경이 없으므로 **실제 1,400개 모델 추론과 연결 개선은 서버에서 실행**해야 한다.
현재 패키지는 계산 실행 코드이며 새로운 성능 수치를 포함한 완료 결과가 아니다. 상세 기록은 `validation/`에 있다.

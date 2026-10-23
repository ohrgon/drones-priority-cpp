# Table 5.14: κ=1만 새로 5M 학습 → Validation → 연결 개선

기존 COST1 모델을 찾는 작업은 수행하지 않는다. **κ=1 조건만 새 초기 가중치에서 5M 학습**하고, 최종 모델을 기존 Validation 180개에서 평가한 뒤 연결 개선한다. κ=3·7·9·10·11·12·13·14는 이미 저장된 1,440개 map별 Raw/Refined 기록을 그대로 재사용한다.

## 무엇을 새로 수행하나

| 대상 | 학습 | 경로 추론 | 연결 개선 | 평균 재계산 |
|---|---|---|---|---|
| κ=1 | 새로 0→5M | Validation 180만 | 같은 공통 엔진 | 수행 |
| 기존 여덟 κ | 하지 않음 | 하지 않음 | 하지 않음 | 새 공통 지도 집합으로만 재집계 |

κ=1의 과거 성공 수 165/180이나 과거 평균의 재현을 완료조건으로 요구하지 않는다. **이번 κ=1은 새 재학습 결과**이고, 과거 κ=1 기록과 독립적으로 보존한다. 동일 seed라도 과거 실행과 비트 단위 동일성을 주장하지 않는다.

## 고정 조건

- κ=1 / Relaxed B=6 / β=0.5 / FULL / seed=42.
- 원래 C3NT의 CNN3/no-clock: 6채널 지도 + heading + 남은 budget, 결합 특징 261.
- H=ceil(5N), 완료 시 +2, 신규 방문 보상 +1/N, 외부 상한 실패 보상 0 및 truncation bootstrap 유지.
- 새 학습량 5,000,000; PPO update 단위 실제 종료 5,005,312 steps. 다른 κ들과 같은 5M 비교이며 40M이 아니다.
- num_envs=8, n_steps=1024, batch_size=2048, n_epochs=4, gamma=1, gae_lambda=0.95, ent_coef=0.0005, clip_range=0.2, target_kl=0.02.
- 원래 학습률 max(5e-5, 2.5e-4×progress). Adam 상태를 다른 COST에서 가져오지 않는다.
- 원래 checkpoint 200,000-step 간격 및 50,000-step 단위 복구 요청(실제 저장은 완료된 update 경계)을 유지한다. 중간 checkpoint를 평가·선택하지 않고 최종 고정 5M만 평가한다.
- 학습: 원래 Synthetic Train 900; 평가: 원래 Synthetic Validation 180. Test 720 / OSM500은 읽거나 실행하지 않는다.

## 실행

ZIP을 suite 폴더에 두고:

```bash
cd <ORIGINAL_WORKSPACE>/uav_chapter4_suite_v2
conda activate dqn_env
unzip -n CH5_KAPPA1_FRESH5M_v1.zip
python -u CH5_KAPPA1_FRESH5M_v1/run_kappa1_fresh5m.py \
  --root . --gpu 0 --phase all --workers 3
```

GPU 0은 지정 기본값일 뿐 현재 비어 있음을 확인한 것은 아니다. 사용 가능한 GPU 번호를 최초 실행의 `--gpu`로 지정한다. Validation/연결 개선은 CPU만 사용한다. CUDA_VISIBLE_DEVICES로 GPU를 재번호화한 셸에서는 해당 프로세스에 보이는 번호를 사용한다.

학습 출력은 현재 터미널과 `logs/kappa1_train.log`에 함께 나온다. GPU 2·3의 NTERM/NCOV를 중지하거나 기존 출력에 쓰지 않는다. 기존 COST1·COST11 모델을 불러오지 않는다.

### 단계별 실행 및 재개

- `--phase check`: 데이터 동일성 및 원래 모델/환경의 사전검사. 학습 업데이트 없음.
- `--phase train`: 새 κ=1의 5M 학습까지만. 이후 `--phase refine` 수행.
- `--phase refine`: 완료된 **이 새 κ=1 5M**으로 Validation 180 평가·연결 개선·아홉 조건 재집계.
- `--phase report`: 저장된 완료 결과로 표/ZIP만 생성. 새 학습·추론 없음.
- `--phase all`: 학습 → Validation/Refined → 표까지.

중단되면 같은 명령으로 재실행한다. 새 학습의 원래 전체 상태 복구 기능을 사용하며, 이미 완료된 5M은 다시 학습하지 않는다. Validation Raw 경로가 저장되어 있으면 연결 개선부터, Refined까지 완료되어 있으면 저장 결과를 재사용한다. 출력 폴더를 삭제하지 않는다.

사전검사 후에는 장치와 학습 설정이 고정되므로, 학습을 재개할 때는 같은 `--gpu`를 유지한다. `--phase refine`과 `report`에는 GPU 번호가 필요하지 않다. CPU 학습이 필요한 경우 최초 실행부터 `--device cpu`를 사용한다.

## 데이터 경로

패키지에 기존 manifest의 Train900+Validation180 행만 포함하였다(모델과 지도 배열 자체는 미포함). 원래 서버에 있는 `.npy` 파일을 읽는다. 원래 절대경로 및 현재 suite 상위의 `train/`, `test/` 위치를 확인한다. 재귀 탐색이나 오래된 COST1 폴더 검색은 하지 않는다.

지도 폴더만 이동한 경우:

```bash
python -u CH5_KAPPA1_FRESH5M_v1/run_kappa1_fresh5m.py \
  --root . --dataset-root /actual/mix_dataset2 --gpu 0 --phase all --workers 3
```

`--dataset-root`는 원래 map_id의 `train/...`, `test/...`를 포함하는 경로이다. 파일명만으로 승인하지 않고 원래 지도 배열 해시와 Validation 초기조건 180개를 확인한다. 새 데이터셋을 만들거나 Test 지도로 대체하지 않는다. 기존 8개 CSV는 패키지에 포함되어 추가 수집이 필요 없다.

## 출력

```text
chapter5_TABLE514_KAPPA1_FRESH5M_v1/
  protocol.json
  manifest.csv
  source_snapshot/
  runs/ppo_relaxedB6_NRG1_C3NT_b0.5_g1_l0.95_budgetobs_seed42/
    last_model.zip
    checkpoints.json
    status.json
    training_updates.csv
    resume/   # 진행 중 전체 상태 복구
  validation_refinement/maps/<map_id_hash>/
    raw_trace.npz
    raw_metrics.json
    raw_complete.json
    refined_trace.npz
    refined_metrics.json
    refinement_audit.json
    segments.csv
    complete.json
  logs/kappa1_train.log
  audit/
  report/
    cost1_fresh_per_map.csv
    table_5_14a_historical_RAW_n163.csv
    existing_eight_RAW_REFINED_n167.csv
    existing_eight_common_success_maps.csv
    table_5_14b_nine_costs_RAW_REFINED.csv  # 180개 완료 후
    nine_common_success_maps.csv
    per_map_nine_costs.csv
    Table_5_14b_KO.md
    audit.json
  TABLE514_KAPPA1_FRESH5M_REVIEW.zip
```

새 Table 5.14(b)의 시간 평균은 κ=1·3·7·9·10·11·12·13·14가 모두 성공한 같은 지도에서 계산한다. 기존 163/167을 고정하지 않는다. κ=1을 새로 학습하므로 성공 수 자체도 새 결과로 기록한다. **다른 κ의 경로는 그대로지만 공통집합이 바뀌면 그 표시 평균은 달라질 수 있다.** 공통 성공집합이 비면 성공률은 보고하고 시간 평균은 미정의로 남긴다. 실패 경로를 완주로 바꾸지 않는다.

Table 5.14(a)는 과거 κ1을 포함한 역사적 Raw 표로 보존하며 새 κ1 행만 삽입하지 않는다. 새 아홉 조건 표에는 κ1만 fresh_rerun, 나머지는 existing_saved_result로 명시한다.

## 구현 근거·검증 범위

보상·마스크·물리·CNN은 업로드한 원래 COST1 코드이다. PPO trainer는 업로드한 원래 NRG1 trainer에 원래 `run_nrg1_c3nt.py`의 `transform_trainer`를 적용한 것이다. `native_provenance.json`에 원자료 경로와 해시를 남겼다. 기존 큰 출력 파일 탐색이나 COST11 가중치의 비용계수 변경은 수행하지 않는다.

episode CSV의 선택적 진단 열 문제에만 기존 `run_nrg1_c3nt_logfix.py`의 로그 호환층을 적용한다. 환경 전이, 보상, optimizer, learning rate와 recovery 코드는 변경하지 않는다.

연결 개선은 이전 패키지의 `ch5_common.py` 및 `SegmentGraph`를 그대로 사용한다. 최초 방문순서·도착방향·원래 B6 행동 규칙을 유지하고 구간별 이동 횟수는 늘리지 않는다.

로컬 검사는 표 재집계, 데이터 membership, 원래 COST1 보상·마스크 환경, 임시 테스트 경로의 공통 개선, 캐시 무결성 등을 확인한다. **현재 작성 환경에는 gymnasium/SB3/sb3-contrib가 없어 모델 생성·GPU 학습·중단복구의 실제 실행은 수행하지 않았다.** 서버의 기존 dqn_env에서 원래 사전검사가 실행된 뒤 학습이 시작된다. 임시 테스트 경로를 논문 수치로 사용하지 않는다. 180개 실제 결과는 서버 실행 후 확보해야 한다.

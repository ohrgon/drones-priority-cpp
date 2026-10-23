# Table 5.14(b): κ=5·15·20 추가 — 5M / Validation180 / Raw–Refined

## 실행

이 ZIP을 기존 `uav_chapter4_suite_v2` 폴더에 놓고 압축을 푼 뒤, **기존 dqn_env**에서 실행한다.

```bash
unzip -n kappa_sweep_5m.zip
python -u kappa_sweep_5m/run_kappa5_15_20.py \
  --root . --gpus 0 --phase all --workers 3
```

`--gpus 0`은 GPU 0 하나에서 **κ=5 → 15 → 20을 순서대로** 학습한다. 이후 세 조건을 차례로 CPU Validation/연결 개선한다. GPU 0이 비어 있다는 의미는 아니며 다른 GPU가 필요하면 최초 실행 시 번호를 바꾼다. 다른 학습 프로세스를 종료하거나 기존 출력 폴더를 지우지 않는다.

예를 들어 실제로 비어 있는 GPU가 세 개이면 `--gpus 0 1 2`로 각각의 κ를 병렬 학습할 수 있다. 기존 No-mask/Boundary-only 학습과 GPU를 중복 배정하지 않는 것이 좋다. 여러 GPU를 주어도 Validation/연결 개선은 κ 하나씩 진행하며, `--workers`는 그 κ에 배정하는 CPU 작업자 수이다.

## 정확한 범위

| 구분 | 수행 내용 |
|---|---|
| 새 학습 | **κ=5, 15, 20만**, 각자 독립된 새 가중치와 Adam 상태로 0→5M |
| 재사용 | κ=1·3·7·9·10·11·12·13·14의 저장된 지도별 Raw/Refined 결과 |
| 최종 비교 | **1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 20의 12조건** |
| 마스크/보상 | Relaxed B=6, FULL, β=0.5, H=ceil(5N), seed=42 |
| CNN | 원본 CNN3/no-clock, 지도 6채널 + heading + 남은 budget, 결합 261차원 |
| 학습 데이터 | 원래 Synthetic Train 900개 |
| 평가 데이터 | 원래 Synthetic Validation 180개만 |
| 실행하지 않음 | Test 720, OSM 500, 기존 9조건의 학습·추론·연결 개선 |
| 평가 시점 | 고정 최종 5M만 평가, 중간 checkpoint 평가/선택 없음 |
| 연결 개선 | 업로드한 ch5_common.py / SegmentGraph 그대로, fixed-heading T-only |

**이번 20은 비용계수 κ=20이다. 20M/40M 학습이나 κ=1부터20까지 모든 정수의 sweep이 아니다.** 새로운 세 조건의 과거 Raw 보고값/성공 수를 강제 재현하지 않는다. 새 결과는 `fresh_added_run`, 기존 아홉 조건은 `existing_saved_result`로 구분한다. 과거 Table 5.14(a)의 숫자를 새 학습 완료 조건으로 사용하거나 덮어쓰지 않는다.

## 원래 5M 학습 설정 유지

`upstream/reference/training_settings.json`의 설정을 그대로 사용한다. num_envs=8, n_steps=1024, batch_size=2048, n_epochs=4, gamma=1, gae_lambda=0.95, ent_coef=0.0005, clip_range=0.2, target_kl=0.02이다. 학습률은 `max(5e-5, 2.5e-4 × progress)`이며, 명목 5,000,000의 실제 PPO update 종료는 5,005,312 steps이다.

기존 κ=1 코드와 동일하게 **중간 checkpoint 200,000-step 간격, 복구 상태 요청 50,000-step 간격**을 유지한다. 실제 저장은 완료된 optimizer update 경계에서 이루어진다. 따라서 이번 κ 실험은 앞선 No-mask/Boundary-only의 ‘중간 저장 없는 40M’ 작업과 별개다. 중간 저장 파일을 Validation 점수로 선택하지 않으며 고정 최종 5M을 사용한다.

시간비용항만 다음과 같이 바뀐다.

```text
r = Δnew_active/N
    - κ × [(1-β) + β × W_pre/W_total] × Δt / [ceil(5N) × t_turn]
    + 2 × I(complete)
```

신규 방문항 `1/N`, 완료항 `+2`, 실패 종료 보상 0, 외부 step 상한과 truncation bootstrap은 유지한다. 물리 시간계수·행동 마스크·관측·초기조건·CNN·PPO 업데이트·연결 개선 알고리즘은 바꾸지 않는다. `source_snapshot`의 core에는 비용 두 항과 return-identity/진단의 대응 변경만 적용하고, 같은 동작을 재생하여 κ 이외의 차이가 없는지 검사한다.

## 기존 9조건 결과를 읽는 방법

기존 여덟 κ=3·7·9·10·11·12·13·14의 1,440행은 업로드한 패키지에 포함되어 있다. 그대로 재사용한다. **κ=1의 학습 결과는 코드 ZIP에 포함되지 않으므로 서버의 저장 결과를 읽는다. 모델 가중치나 지도별 경로를 새로 추론할 필요는 없다.**

자동 확인 위치는 다음으로 제한한다. 큰 결과 폴더의 재귀 검색이나 옛 모델 탐색은 하지 않는다.

```text
<root>/chapter5_TABLE514_KAPPA1_FRESH5M_v1/report/per_map_nine_costs.csv
<root>/chapter5_TABLE514_KAPPA1_FRESH5M_v1/report/cost1_fresh_per_map.csv
<root>/chapter5_TABLE514_KAPPA1_FRESH5M_v1/TABLE514_KAPPA1_FRESH5M_REVIEW.zip
<root>/TABLE514_KAPPA1_FRESH5M_REVIEW.zip
<root>/TABLE514_KAPPA1_FRESH5M_REVIEW(1).zip
```

다른 위치라면 원래 결과 폴더, CSV 또는 결과 ZIP을 직접 지정한다.

```bash
python -u kappa_sweep_5m/run_kappa5_15_20.py \
  --root . --gpus 0 --phase all --workers 3 \
  --existing-nine /actual/path/TABLE514_KAPPA1_FRESH5M_REVIEW.zip
```

재사용 전에 180개 Validation 지도 목록·지도 해시·초기조건 ID·5M step·κ=1 checkpoint ID를 확인하고, 나머지 여덟 조건이 포함되어 있으면 원본 여덟 CSV와 시간/성공/단계 수를 대조한다. 받아들인 κ=1 기록은 새 출력의 `existing_reference`에 고정한다. **원래 결과를 수정하지 않는다.**

κ=1 자료가 자동 위치에 없으면 새 5·15·20 학습은 진행하되, 최종 12조건 표는 `PARTIAL`로 남긴다. κ=1을 다시 학습하거나 평균표 한 행을 복사해 만들지 않는다. 세 조건이 완료된 후 자료를 지정하여 아래 명령만 실행하면 된다.

```bash
python -u kappa_sweep_5m/run_kappa5_15_20.py \
  --root . --phase report \
  --existing-nine /actual/path/TABLE514_KAPPA1_FRESH5M_REVIEW.zip
```

`report`는 저장 결과만 읽으며 새 학습·모델 추론·연결 개선을 하지 않는다. `all/refine/report` 종료 코드 2는 최종 12조건 자료가 아직 모이지 않은 `PARTIAL` 상태이며 성공 0과 다르다. 명시적으로 지정한 파일이 잘못되거나 결과가 충돌하면 중단한다.

## 단계별 실행과 재개

```text
--phase all     : 새 세 조건의 5M 학습 → Validation180/연결 개선 → 12조건 재집계
--phase train   : 새 세 조건의 학습만
--phase refine  : 완료된 새 최종 모델로 누락된 Validation Raw/Refined만 처리
--phase report  : 저장 결과의 표/검토 ZIP만 재생성
--phase check   : 원본 코드의 사전검사만; optimizer update 없음
```

학습이 중단되면 **같은 명령, 같은 GPU 번호, 같은 출력 폴더**로 재실행한다. 이 새 실행이 이미 저장한 원본 전체 상태 복구를 사용한다. 새 κ의 5M 학습이 완료되어 있으면 재학습하지 않는다. 해당 Validation Raw가 저장되어 있으면 연결 개선부터 수행하고, Refined까지 완료되었으면 재사용한다. 성공/실패 모든 지도를 기록하며 새 실패 경로를 완주로 고치지 않는다.

기존 여덟 조건의 과거 실패-prefix 진단에 부분 연결 개선 수치가 들어 있더라도 원자료 그대로 보존한다. 이러한 수치를 완주 T/Lw로 사용하지 않는다. 새 세 조건은 업로드된 common_refine 규칙대로 실패 action trace를 그대로 보존한다.

지도 `.npy`만 다른 경로로 이동했다면 `--dataset-root`에 원래 map_id의 `train/...`, `test/...`를 포함하는 폴더를 지정한다. 여기서 `test/...` 경로 아래에도 원래 Validation으로 지정된 파일이 있을 수 있으며, **manifest의 split으로 구분**한다. 파일명만으로 새 split을 구성하지 않는다.

```bash
python -u kappa_sweep_5m/run_kappa5_15_20.py \
  --root . --dataset-root /actual/mix_dataset2 --gpus 0 --phase all --workers 3
```

## 출력

```text
chapter5_TABLE514_KAPPA5_15_20_FRESH5M_v1/
  plan.json
  existing_reference/                 # 기존 κ=1 저장값의 읽기 전용 참조 사본
  cost5/                              # 같은 구조로 cost15/, cost20/
    protocol.json
    manifest.csv                      # Train900 + Validation180만
    source_snapshot/
    runs/<cost-specific-run-id>/
      last_model.zip
      checkpoints.json
      status.json
      training_updates.csv
      resume/                         # 학습 진행 중 복구 상태
    validation_refinement/maps/<map_id_hash>/
      raw_trace.npz
      raw_metrics.json
      raw_complete.json
      refined_trace.npz
      refined_metrics.json
      refinement_audit.json
      summary.json
      complete.json
    report/per_map.csv                # κ 한 조건의 180행
  report/
    per_map_added_costs.csv            # 새 세 조건 완료시 540행
    added_three_common_RAW_REFINED.csv # 새 세 조건끼리만의 공통집합 (보조)
    per_map_twelve_costs.csv           # 기존9 + 새3, 총 2,160행
    table_5_14b_twelve_costs_RAW_REFINED.csv
    twelve_common_success_maps.csv
    Table_5_14b_KO.md
    audit.json
  logs/kappa5_train.log                # 각 조건의 train/refine 로그
  TABLE514_KAPPA5_15_20_REVIEW.zip      # 문서 반영용 전달 파일
```

한 지도별 행에 Raw와 Refined 값이 함께 들어 있다. 새 3조건은 **540쌍**, 최종 12조건은 **2,160쌍**이다. 완료된 조건의 기록만 최종 묶음에 들어가며, 부분 처리 중의 상세 진행은 각 조건 `report/audit.json`과 map별 commit 파일에 남는다.

**Table 5.14(b)의 모든 시간 평균을 12개 κ가 모두 성공한 새 교집합에서 다시 계산한다.** 기존 공통 163 또는 167은 고정하지 않는다. 기존 아홉 경로를 재실행하지 않아도 공통집합이 달라지면 표의 표시 평균은 달라질 수 있다. 공통 성공집합이 비면 성공 수를 보고하고 시간 평균은 ‘—’로 남긴다.

## 구현 및 점검 범위

`upstream/`은 이번에 업로드한 κ=1 코드 ZIP을 바이트 단위로 보존한 사본이다. 원본 PPO trainer, CNN, 물리/마스크, 복구 코드, `ch5_common.py`, `SegmentGraph`의 해시를 검사한다. 새 두 실행 모듈은 κ별 사본/메타데이터, 비용계수 적용, 프로세스 분리, 재사용 입력 및 12조건 집계만 담당한다.

로컬 자동검사에는 실제 원본 환경에서 비용항 스케일·완주/상한/실패 처리, 동일 action의 물리/마스크/관측 동일성, 공통 연결 개선, spawned CPU 작업자에서 κ별 모듈 분리, Raw/Refined 캐시 재사용, 기존 결과 CSV/ZIP 읽기, 새 공통집합 재계산, 자료 혼합/위변조 거부가 포함된다. 테스트용 지도/임시 수치는 논문 결과가 아니다.

현재 작성 환경에는 gymnasium / stable-baselines3 / sb3-contrib가 없어 실제 PPO 생성·5M GPU 학습·optimizer 복구 실행은 수행하지 않았다. **서버의 기존 dqn_env(SB3 및 sb3-contrib 2.7.0)**에서 업로드된 원본 사전검사와 학습이 실행된다. 패키지가 라이브러리를 자동 설치하거나 업그레이드하지 않는다.

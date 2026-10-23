# Table 5.16 — Strict 40M β 민감도: Raw와 Refined 비교

## 실행

이 폴더를 기존 `uav_chapter4_suite_v2` 아래에 놓고 실행합니다. 패키지 안의 파일은 함께 유지합니다.

```bash
conda activate dqn_env
python -u strict_beta_sensitivity_40m/run_table516_strict40m_refined.py \
  --root . --phase all --workers 3
```

입력만 먼저 검사하려면 `--phase check`, 완료/부분 결과의 보고서만 만들려면 `--phase report`를 사용합니다. 다시 실행하면 완료된 지도는 재사용합니다. `--workers 4`도 허용하지만 실제 프로세스 수는 세 β에 맞춰 최대 3개입니다.

## 평가 범위

- Strict(B=0), COST11, FULL, seed42, β=0 / 0.5 / 1.0.
- 각 β의 **같은 40M checkpoint**: 실제 40,001,536 steps. 다른 학습량이나 최신 last_model로 대체하지 않습니다.
- 같은 **Synthetic Validation 180개**. 총 540개의 Raw/Refined 쌍.
- Train 900, Synthetic Test 720, OSM 500에는 rollout을 수행하지 않습니다. 새 학습·optimizer update가 없습니다.
- 이 코드가 Table 5.15의 기존 Relaxed 민감도, Table 5.5 Four-way, COST1 탐색, B=1·2·4 학습을 변경하지 않습니다.

## 기존 결과만 읽는데 왜 PPO 추론이 필요한가?

업로드된 `run_strict_50m.py`의 validation evaluator와 이를 사용한 `validate_strict_20M_40M_every5M.py`는 원래 지도별 성능 JSON/CSV를 저장했습니다. **primitive action 시퀀스나 전체 trajectory를 저장하지 않는 구조**입니다. 평균 T/Lw 또는 최초 도착시간만으로 원래 경로를 복원해서는 안 됩니다.

따라서 이 패키지는 다음과 같이 처리합니다.

1. 원래 학습계획의 등록된 40M checkpoint·해시, 고정된 COST11 소스, Validation 목록을 확인합니다.
2. 기존 Raw map별 수치를 검산 기준으로 읽습니다. 완주/실패와 평가 단계는 수정하지 않습니다.
3. 새 출력 폴더에 이미 저장한 행동 경로가 있으면 그대로 재사용합니다. 경로가 없으면 **동일한 40M checkpoint로 CPU 결정론적 추론**을 수행하여 경로를 저장합니다. 이는 새 학습이 아닙니다.
4. 복원한 Raw T/Lw, 이동·회전·재방문 수, map hash, 시작 위치·heading을 원래 Raw 기록과 대조합니다. 불일치를 무시하지 않습니다.
5. 저장된 Raw 경로에 공통 연결 개선을 실제 적용하고 원래 환경에서 다시 재생합니다.
6. 세 β 모두의 공통 성공집합에서 전후 평균을 다시 계산합니다.

처음 실행할 때 원래 평가 폴더에 수치 기록만 있다면 **540개 전체의 Raw 경로 생성이 필요**합니다. 다음 실행부터는 새로 저장한 경로를 재사용합니다. 연결 개선 도중 멈춰도 그전에 확정된 Raw 경로는 남습니다.

`--cached-only`는 checkpoint를 로드하지 않도록 하는 보조 옵션입니다. 이 출력 폴더에 행동 경로가 없으면 명확히 중단하며, CSV를 Refined로 복사하지 않습니다. 첫 실행에는 이 옵션을 붙이지 않습니다.

## 동일 연결 개선 엔진

이전 `CH5_REMAINING_TASKS_v1` / 제5장 공통 후처리에 사용한 `ch5_common.py`와 `ch5_vendor/segment_graph_original.py`를 byte-identical하게 포함했습니다.

- 최초 방문순서 고정.
- 각 최초 방문 셀의 도착 heading 고정.
- Strict의 원래 진행 규칙 유지(B=0). 설정 파일의 보관용 detour_budget_max=6을 B=6 행동으로 해석하지 않습니다.
- T-only 목적.
- 각 연결구간의 이동 횟수가 원래보다 증가하지 않는 제약.
- 수치 동률에서는 원래 연결 유지.
- 실패 경로는 실패 상태·원래 행동 시퀀스를 유지.

개선이 없다는 결론도 이 절차를 실제로 적용하고 검증한 다음 기록합니다. Test의 β=0.5에서 변경 0이었던 사실을 Validation의 세 β에 대신 적용하지 않습니다.

## 원본 경로

기본 학습계획:

```text
strict_beta_runs_seed42/plan.json
```

기본 Raw 수치 기록:

```text
cost11_strict_validation_20M_40M_every5M/validation/<beta>/step_40001536/
strict_beta_runs_seed42/validation/<beta>/step_40001536/
```

학습계획에는 β=0.5가 이전 20M 연장학습을 거친 이력이 들어 있으나, 선택하는 weight는 **현재 등록된 40,001,536 steps 하나**입니다. 세 모델의 실험기록 형식 차이(β=0.5 continuation spec에는 encoder_ablation 키가 없음)는 원래 CNN 소스 해시와 로드된 모델 검사로 확인하며 임의로 바꾸지 않습니다.

실제 위치가 다르면:

```bash
python -u strict_beta_sensitivity_40m/run_table516_strict40m_refined.py \
  --root . --phase all --workers 3 \
  --training-root /실제/Strict_학습계획_폴더 \
  --validation-root /실제/Raw_validation_결과_폴더
```

원본 전체 manifest가 이동/삭제된 경우에는 기존 Raw 평가가 내보낸 `report/validation_manifest180.csv`를 읽고, 앞서 검증한 180개 ID/hash 목록과 일치하는 경우에만 사용합니다. 어떤 경우에도 새로운 Validation 분할을 만들지 않습니다. map_id 경로에 `test/`가 있어도 실제 분할은 manifest의 `split=validation` 기준입니다.

기존 원본 파일은 읽기 전용으로 사용합니다. 진행 중인 학습의 checkpoint registry가 추가될 수 있으므로 전체 registry를 잠그지 않고 **이번에 사용할 고정된 40M 항목과 실제 weight bytes**를 검증합니다. 다른 학습 프로세스를 종료하지 않습니다.

## 출력

```text
chapter5_TABLE516_STRICT40M_BETA_REFINED_VAL180_v1/
├── protocol.json
├── input_audit.json
├── validation_manifest.csv
├── index.html
├── methods/
│   ├── strict_b0/
│   ├── strict_b05/
│   └── strict_b1/
├── logs/
├── report/
│   ├── table_5_16_strict40m_raw_refined.csv
│   ├── all_betas_validation_raw_refined_per_map.csv
│   ├── refinement_audit_per_map.csv
│   ├── common_success_maps.csv
│   ├── beta_pairwise_refined.csv
│   ├── checkpoint_status.csv
│   ├── raw_reproduction_audit.json
│   ├── Table_5_16_KO.md
│   └── audit.json
└── TABLE516_STRICT40M_BETA_REFINED_REVIEW.zip
```

표에는 성공/180, 공통 성공 n, Raw T/Lw, Refined T/Lw, 두 시간 감소율, 실제 변경 경로 수를 넣습니다. `Lw/T`는 **평균 Lw / 평균 T**로 별도 저장합니다. 성공률은 전체 180개, 시간 비교는 세 β가 모두 성공한 **같은 지도들**입니다.

세 β × 180개가 완결되고 Raw 검산이 일치해야 `COMPLETE`입니다. 일부만 처리했을 때 전체 공통 평균을 완료 결과처럼 쓰지 않고 `PARTIAL`과 산출 대기를 표시합니다. Raw 수치만으로 Refined를 채우지 않습니다.

ZIP에는 전체 map별 수치·검사 결과와 β별 경로 샘플을 담습니다. 모델 가중치나 전체 지도 원본은 넣지 않습니다. 실행 후 이 ZIP을 전달하면 실제 Refined 결과로 Table 5.16과 해석을 교체할 수 있습니다.

## 중단·재개

같은 명령을 다시 실행합니다. `raw_complete.json`까지 저장됐으면 PPO 추론 없이 후처리부터 진행합니다. `complete.json`까지 확정됐으면 추론·연결 개선 모두 반복하지 않습니다. 완료된 파일의 해시가 달라졌으면 조용히 덮어쓰지 않고 오류로 표시합니다.

프로토콜·원본 모델이 실제로 바뀌었다면 다른 `--out`을 사용해야 합니다. 이 패키지를 수정하면서 기존 결과를 같은 실험처럼 섞지 않습니다.

입력 단계 실패 시 `TABLE516_INPUT_DIAGNOSTIC.zip`을 생성합니다. 실패의 의미는 입력 누락/불일치이며 실험 성공률 0이 아닙니다.

## 검증 범위

`VALIDATION.md`와 `validation/`을 참고하세요. 로컬 테스트용 지도의 결과는 논문 실험 수치가 아닙니다. 실제 등록 checkpoint를 로드한 540개 경로 생성과 연결 개선은 사용자의 서버에서 수행됩니다.

# 기존 B6 · β=0.5 · COST11 20M → 총 40M 이어학습

## 바뀐 점

이 패키지는 `COST11_B6_BETA05_40M_SAME_B124_v1`의 0→5→20→40M 새 학습기가 아니다.
기존 `chapter4_COST11_5M_to_20M_seed42/relaxed_b05`의 **완료된 20,004,864-step 정책과 Adam optimizer**를 불러와 총40M까지 학습한다. 기존0–20M 학습·평가를 반복하지 않는다.
실제 마지막 optimizer update는 40,001,536 steps이고 추가 optimizer rollout block은 2,441개(19,996,672 interactions)이다. 기존20M에 40M을 더해60M까지 학습하는 코드가 아니다.

## 실행

아래 패키지를 `uav_chapter4_suite_v2` 폴더에 압축 해제한다. 패키지 안의 파일은 함께 유지한다.

```bash
cd <ORIGINAL_WORKSPACE>/uav_chapter4_suite_v2
conda activate dqn_env
unzip -n relaxed_b6_resume_20m_to_40m.zip
python -u relaxed_b6_resume_20m_to_40m/run_cost11_B6_resume20M_to40M.py \
  --root . --gpu 0 --phase all
```

이전 **새로 시작하는 B6 학습기**가 같은 GPU에서 실행 중이면 그 프로세스를 실행한 터미널에서 Ctrl+C로 멈춘 다음 이 코드를 실행한다. 기존 출력은 삭제하지 않는다. 이 패키지는 다른 프로세스를 자동 종료하지 않는다. B1/2/4 실행은 수정하거나 종료하지 않는다.

- `--phase all`: 기존20M 검사 → 20M부터40M 학습/매1M Validation → 고정40M Raw Test720/OSM500 → 보고서.
- `--phase train`: 검사와 학습/Validation까지만. Test 없음.
- `--phase check`: 입력 검증, 원본 B6 동작 검사, GPU 가용성, 원래모델의 policy/Adam 복원·sampler 기록 재생. 학습 update 없음.
- `--phase validate`: 중단된 새 Validation과 남은 영구 checkpoint의 Validation을 처리. 삭제된 임시1M 모델을 새로 생성하지 않음.
- `--phase test`: 이 새 출력의 고정40M이 완료된 경우에만 Test720/OSM500.
- `--phase report`: 기존 새 출력의 완료 평가만 재집계하고 전달 ZIP 생성. 학습/추론 없음.

GPU0은 기본 지정일 뿐 현재 비어 있음을 확인한 것이 아니다. 새 실행에서 다른 GPU를 쓰려면 `--gpu`를 지정한다. 재개에는 같은 GPU 번호와 같은 출력 경로를 사용한다.

## 입력

1. **학습 시작점**

```text
chapter4_COST11_5M_to_20M_seed42/relaxed_b05/
  continuation_protocol.json
  manifest.csv
  runs/ppo_relaxedB6_NRG1_C3NT_COST11_b0.5_g1_l0.95_budgetobs_seed42__continue_to_20000000/
    last_model.zip
    checkpoints.json
    run_spec.json
    training_settings.json
    status.json
    map_order_rank0.csv ... map_order_rank7.csv
```

기존20M source는 이전 업로드의 실제 protocol·run_spec·등록 checkpoint checksum과 대조한다. `last_model.zip`이 없고 동일20M의 등록된 `checkpoints/step_20004864.zip`만 남아 있으면 **그 동일 학습시점**만 사용한다. 손상된 final을 발견한 경우에는 이를 다른 모델로 숨기지 않고 중단한다.

2. **기존 B124의 설정·지도·비학습 기준선 (가중치 아님)**

```text
chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42/
  plan.json
  manifest.csv
  _runtime/strict/...
  _runtime/relaxed/...
  reference_validation_baselines.csv
  reference_test_baselines.csv
```

기존20M과 B124 B6 설정의 물리환경·보상·네트워크·PPO·지도순서가 일치하는지 확인한다. B124의 정책 가중치를 B6에 불러오지 않는다. 기존 checkpoint는 전체 정책과 **Adam moment/counter**가 포함됐는지 확인하고 별도 새 출력에 복사한다.

경로가 다른 경우 실제 experiment root 또는 run 폴더를 명시할 수 있다.

```bash
python -u relaxed_b6_resume_20m_to_40m/run_cost11_B6_resume20M_to40M.py \
  --root . --gpu 0 --phase all \
  --source-root /실제/기존20M/relaxed_b05 \
  --b124-root /실제/B124/출력폴더
```

기존20M이 없으면 입력 오류로 중단한다. **COST5/β0/Strict/5M/새 가중치로 자동 대체하지 않는다.** 모델은 사용자가 직접 생성한 신뢰할 수 있는 checkpoint만 사용한다.

## 고정 조건과 이어받는 범위

- B=6, β=0.5, COST11 FULL, CNN3/no-clock, 261차원 특징, H=ceil(5N), seed42.
- 기존20M과 같은 8 environments × 1024 rollout, batch2048, epochs4, γ1, GAE λ.95, clip.2, entropy.0005.
- 추가 구간의 LR은 기존 하한 `5e-5` 고정.
- 정책·Adam 상태는 tensor digest로 복원 전후를 확인한다.
- 지도 선택 순서는 `map_order_rank*.csv`를 원래 balanced sampler로 재생하여 이어받는다.
- **새 continuation 경계**에서 진행 중 source episode는 버리고 새 episode 및 전역 RNG로 시작한다. 기존0–40M 무중단 실행과 완전히 동일하다고 주장하지 않는다.
- **새 출력 내 중단 후 재개**는 원래 native full-state recovery(모델, optimizer, 환경, sampler, RNG, 로그 위치)를 사용한다.
- 새0M 학습기가 아니므로 B124와 동일한 새로운 초기가중치에서 반복했다고 기록하지 않는다. B6의 기존 학습 이력을 유지하면서 총학습량·설정·새 저장/평가 간격을 맞춘 실험이다.

## 저장과 평가

- 신규 영구 checkpoint: 25M,30M,35M,40M (실제25,001,984 / 30,007,296 / 35,004,416 / 40,001,536).
- 신규 Validation: 21M부터40M까지 매1M, 매번 동일 Synthetic180, CPU 별도 프로세스.
- 5M 배수가 아닌 Validation용 모델은 임시이며 완료 후 지운다. Native recovery와 평가결과는 남긴다.
- 기존1–20M 성능은 원래 결과폴더에 그대로 둔다. 이번보고서는 그 결과가 없다고 신규미완료로 세지 않고 **새20개 평가시점**만 완료수를 계산한다.
- `all`의 최종 Test: Synthetic720와 OSM500 분리. Validation을 섞지 않음.
- Stage1 Raw만 생성한다. 연결 개선을 자동 수행하지 않는다.
- 참고 BCD/WLG/Strict40M 수치는 기존 B124의 Raw 기준선이다. Chapter5의 다섯 방법 Refined 비교표를 자동 갱신한 것이 아니다.

## 출력 및 재개

```text
chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42/
  plan.json
  source20M/                         # 복사한 원래 정책+Adam, 설정, sampler 로그
  audit/existing20M_import.json
  audit/source_model_preflight.json
  B6/continue_existing20M_to40M/
    continuation_audit.json
    runs/ppo_COST11_FULL_B6_beta05_seed42_existing20M_to40M/
      checkpoints/
      last_model.zip
      resume/
      training_updates.csv
  validation/B6/step_.../
  test/B6/step_40001536/
  logs/B6_train.log
  report/
    index.html
    checkpoint_summary.csv
    category_tier_summary.csv
    training_diagnostics.csv
    permanent_checkpoint_inventory.csv
    B1_B2_B4_B6_same_checkpoint_common_success.csv
    status.json
    UPLOAD_B6_EXISTING20M_TO40M_results.zip
```

중단 후 처음과 같은 명령을 다시 실행한다. source20M을 다시 학습하지 않고 새 출력의 native recovery에서 재개한다. 기존 fresh-B6 출력 폴더를 `--out`으로 지정하면 다른 실험이므로 거부한다.

```bash
tail -f chapter4_COST11_B6_beta05_EXISTING20M_to40M_CP5M_VAL1M_seed42/logs/B6_train.log
```

전달 ZIP에는 모델 가중치·전체 TensorBoard 로그는 넣지 않는다. 코드·계획·원본식별정보·검사기록·집계표·지도별 수치가 포함된다. 완료 후 `report/UPLOAD_B6_EXISTING20M_TO40M_results.zip`을 전달하면 된다.

## 검증 범위

원래 B124의 다섯 학습/환경/평가 지원 모듈은 byte 동일하다. 추가한 것은 기존20M의 provenance 검증, 새 continuation 경로·spec, 시작시점20M 재평가를 피하는 start-hook 조건, 새21–40M 보고서이다.

`selftest_resume20m.py`는 16개 검사로 실제 업로드 설정과 runtime 일치, 가짜 checkpoint로 만든 입력검증/오류/복사/재개, 단계0·1 금지, 총학습량, CPU/GPU 라우팅, 실제 native NumPy B6 경로, 보고서 분모를 확인한다. `selftest_native_B124.py`의 기존16개도 실행했다. 합계32개 통과. 가짜 모델 bytes는 metadata 테스트용일 뿐 학습 가능 모델이나 실험 결과가 아니다.

실제 사용자의20M checkpoint를 이 환경에서 로드하거나 GPU로 학습하지 않았다. 서버의 실제 파일 검증과 policy/Adam 로드는 `check/all/train`에서 수행한다. 소프트웨어 버전이 다르면 자동 설치·업데이트 대신 중단한다.

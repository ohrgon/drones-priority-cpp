# COST11 B=1·2·4 / β=0.5 / 40M — 서버 원본 반영 수정판

## 실행

기존 `dqn_env` 환경을 사용한다. 서버 패키지 설치나 업그레이드는 하지 않는다.
ZIP은 기존 `uav_chapter4_suite_v2` 폴더에 놓는다.

```bash
cd <ORIGINAL_WORKSPACE>/uav_chapter4_suite_v2
unzip -o COST11_B124_BETA05_40M_CP5M_VAL1M_FIX_NATIVE.zip
python budget_b1_b2_b4_40m/run_cost11_B124_beta05_40M.py \
  --root . --gpus 1 2 3 --phase all
```

이번 실행 경로는 `FIX_NATIVE/`이다. 이전 `NATIVE/`, `FIX_BGUARD/`, `FIX_BGUARD2/`의 실행기를 사용하지 않는다.

새 기본 출력 폴더:

```text
chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42/
```

이전 사전검사 실패 출력과 새로운 코드의 identity를 섞지 않는다. 이전 output을 `--out`으로 지정하지 않는다.
기존 원본, source_snapshot, BCD/WLG/Strict 결과, 실패 실행 폴더는 삭제할 필요 없다.
이번 수정판으로 진행을 시작한 이후 중단/재개는 동일 명령과 동일 output을 사용한다.

## 이번에 수정한 원인

사용자가 수집한 서버 원본의 Relaxed COST11 adapter SHA256:

```text
807817035c9d5c3da4c36caf2b56a52128f68f7eb50b2787873ac0215a4960b0
```

이 adapter는 하위 bounded core가 여러 B를 지원하는 것과 달리, 상위 감사 코드에 B=6을 고정하고 있었다.
FIX_BGUARD2는 첫 번째 설정 검사의 우변만 바꾸었으므로 이후 초기 예산 검사에서 실패했다.
`reset`의 후속 검사만 바꾸어도 `step`에서 같은 상수가 남기 때문에 충분하지 않다.

| 원본의 B 고정 지점 | 이번 처리 |
|---|---|
| 설정값 `detour_budget_max == 6` | worker에 배정된 B와 일치해야 함 |
| reset의 `observed_budget(obs, 6) == 6` | 배정된 B로 환산하고 B와 비교 |
| step 전 `observed_budget(pre_obs, 6)` | 배정된 B로 환산 |
| 신규 셀 방문 후 `expected_budget = 6 if dn ...` | 배정된 B로 복원되는지 검사 |
| step 후 예산 검사 및 로그의 두 `observed_budget(..., 6)` | 모두 배정된 B로 환산 |
| `np.zeros(7)`의 예산 상태 빈도 배열 | 0부터 B까지, 길이 B+1 |
| B=0일 때의 관측 예산 환산 | 원래 scalar가 정확히 0인지 먼저 확인 |

예산 관련 표현식 7곳과 B=0 관측 검사를 함께 처리한다. 모든 숫자 6을 전역 치환하지 않는다.
6개 지도 관측 채널, CNN, 움직임/방향, 보상식, 성공·실패 정의는 변경하지 않는다.
원본 하위 `core_ch4_bounded_b6.py`가 실제 마스크·예산 소진·신규 방문 복원을 계속 담당한다.
검사를 끄거나 B를 6으로 위장하지 않는다. 요청한 B와 실제 상태가 다르면 즉시 실패한다.

원본 snapshot 파일 자체는 편집하지 않는다. 인증된 소스에서 정확한 AST 위치를 검사한 뒤,
`C3NoClockMixin.reset`, `step`, `observed_budget`의 코드만 해당 worker 메모리에서 교체한다.
기존 클래스의 `super()` closure와 함수 객체를 보존한다. 변경 감사는
`audit/core_guard_relaxed_B*.json`에 저장된다. 원본 버전/legacy log prefix는 유지하고,
새 plan version과 patch audit가 B 변경 이력을 구분한다.

## 유지하는 실험 설정

- B=1, 2, 4, β=0.5, seed=42; B>0은 Bounded-Detour Relaxed Progress이다.
- 비교 기준은 기존 Strict B=0 β=0.5 40M. Strict 가중치로 warm-start하지 않는다.
- 세 B 모두 새 초기 가중치에서 시작하고, 같은 seed/network의 초기 매개변수 digest를 검사한다.
- COST11 FULL, 외부 실행 한도 H=ceil(5N), γ=1, GAE λ=0.95.
- CNN3 no-clock, map6ch + heading + budget, 결합 특징 261차원.
- PPO: 8 environments, n_steps=1024, batch_size=2048, n_epochs=4,
  clip_range=0.2, target_kl=0.02, ent_coef=0.0005, vf_coef=0.5, max_grad_norm=0.5.
- 기존 학습 이력과 맞춰 0→5M: LR=max(5e-5,2.5e-4×5M 구간 progress),
  5→20M 및 20→40M: LR=5e-5 고정.
- 연장 경계에서는 같은 B의 정책·Adam과 sampler 누적 위치를 이어받는다.
  일반 중단/재개에서는 원본 전체 상태 복구를 사용한다.
- GPU 1=B1, GPU 2=B2, GPU 3=B4.

## 저장과 평가

| 항목 | 설정 |
|---|---|
| 영구 비교용 checkpoint | 5·10·15·20·25·30·35·40M |
| Validation | 매 1M, Synthetic 180개 |
| 비-5M 검증용 임시 모델 | 검증 성공 후 삭제; 수치/지도별 cache 보관 |
| 전체 복구 상태 | 50k를 넘긴 완료 update, 검증 직전; 구간별 최근 2세대 유지 |
| Train | Synthetic 900개 |
| Test | 고정 40M: Synthetic 720개 / OSM 500개 분리 |
| 경로 개선 | 없음 — Stage 1 Raw |

한 update는 8192 interaction이므로 명목 checkpoint 이후 첫 완료 update에서 저장한다.

| 명목 M | 실제 step |
|---:|---:|
| 5 | 5,005,312 |
| 10 | 10,002,432 |
| 15 | 15,007,744 |
| 20 | 20,004,864 |
| 25 | 25,001,984 |
| 30 | 30,007,296 |
| 35 | 35,004,416 |
| 40 | 40,001,536 |

구간 마지막 모델과 복구용 model.zip은 비교용 checkpoint 외에 별도로 남는다.
기준선 비교는 지도 SHA와 초기 위치·heading pair ID를 대조한다.
Test로 checkpoint를 자동 선택하지 않는다. 개발 중 기존 Test를 확인한 이력이 있으므로 blind Test라고 표현하지 않는다.

## 확인한 범위와 확인하지 않은 범위

### 실제 서버 원본으로 실행한 통합검사: 16개 PASS

`NATIVE_INTEGRATION_TEST_RESULT.json`과 `TEST_LOGS/native_integration.txt`에 상세 기록이 있다.
최소 재현용 가짜 클래스가 아니라 업로드된 실제 native core와 하위 모듈을 사용했다.

- Strict/Relaxed 각 9개, 총 18개 원본 파일 hash 확인.
- 수정 전 원본의 `B must remain 6`와 이전 guard-only 수정의 후속 오류 재현.
- production `trace_preflight`의 Strict B=0, Relaxed B=0/1/2/4 실행 통과.
- Strict와 Relaxed B=0: 진단 지도 20개에서 관측·행동·경로·보상·종료 일치.
- 원래 B=6과 수정 후 B=6: 진단 지도 20개에서 관측·info·경로·보상 일치.
- B=0/1/2/4에 대해 80개 진단 rollout, 6,302회 step/완료 확인 호출 검증.
- 예산 소진, Strict 강제 전환, 신규 active-cell 첫 방문 시 B 복원, b/B 관측 확인.
- 성공, 정확히 실행 한도에서 성공, 시간 한도 중단, 단일 셀 완료 처리.
- 원본 episode CSV logger의 성공/실패/단일 셀 저장과 고정 header 보존.
- 실제 native trainer의 신규/연장 patch 매칭과 Python compile.
- 실제 core 상태 복원 후 같은 행동 시 경로·보상·관측 일치.

진단 지도는 코드 검사용 생성 자료이며 학습/Validation/Test 성능 결과가 아니다.
이 실행 검사는 Python 3.13.5 / NumPy 2.3.5 CPU 환경에서 수행했다.
서버의 Python 3.10 / NumPy 2.2.6 및 기존 CUDA/PPO 환경과 동일한 실행 환경은 아니다.
GPU 학습, PPO optimizer update, 실제 900/180/1220개 전체 자료 실행은 여기서 하지 않았다.
서버 실행기의 기존 패키지 버전 검사와 실제 GPU 사전검사는 유지된다.

### 기존 실행기 테스트: 8개 PASS

checkpoint 5M/Validation 1M 분리, 5M 실제 step 목록, B만 변경,
기준선 무결성, 임시 checkpoint 삭제 조건, 영구 checkpoint 유지,
기준선 비교 및 보고서 재집계의 기존 테스트도 통과했다.

총 24개 테스트가 통과했으며 CUDA/40M 학습 완료를 의미하지 않는다.

## 별도 점검 / 실행 범위

```bash
# 포함된 실제 native 원본으로 통합검사만 수행: GPU/학습/실험자료 불필요
python budget_b1_b2_b4_40m/selftest_native_B124.py

# 서버 snapshot/manifest/software/GPU와 예산 동작 검사만 수행
python budget_b1_b2_b4_40m/run_cost11_B124_beta05_40M.py \
  --root . --gpus 1 2 3 --phase check

# 학습 + 1M Validation까지만 수행; Test 제외
python budget_b1_b2_b4_40m/run_cost11_B124_beta05_40M.py \
  --root . --gpus 1 2 3 --phase train

# 완료한 40M 모델의 Test만 수행
python budget_b1_b2_b4_40m/run_cost11_B124_beta05_40M.py \
  --root . --phase test

# 중간/최종 보고서와 전달 ZIP만 재집계
python budget_b1_b2_b4_40m/run_cost11_B124_beta05_40M.py \
  --root . --phase report
```

`all`은 check → train/validation → fixed40M test → report 순서다.
로그:

```bash
tail -f chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42/logs/B1_train.log
```

B2/B4는 로그 파일 이름의 B만 바꾼다.
최종 전달 자료:

```text
chapter4_COST11_B124_beta05_40M_CP5M_VAL1M_nativefix_seed42/report/UPLOAD_B124_beta05_40M_results.zip
```

`tests/native_snapshot/`은 이번 업로드 원본의 재현 테스트 fixture다.
실제 학습은 여전히 서버의 인증된 snapshot을 별도 출력에 복사하여 사용한다.
임의의 다른 환경 구현이나 이름이 비슷한 trainer로 대체하지 않는다.

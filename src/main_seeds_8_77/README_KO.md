# 메인 Strict·Relaxed 추가 seed 실험 — 8, 77

추가 학습 seed는 **8과 77**로 고정했습니다. 같은 두 번호를 Strict와 Relaxed에 공통으로 사용합니다. 기존 seed 42 결과는 보존하고, 새 학습은 다음 네 번입니다.

| 방법 | 실제 우회 예산 | 추가 학습 seed | 각 seed의 누적 학습량 |
|---|---:|---|---:|
| Strict | B=0 | 8, 77 | 40M |
| Relaxed | B=6 | 8, 77 | 40M |

새 두 seed의 학습·평가가 끝나면 방법마다 **42·8·77의 세 seed**를 비교할 수 있습니다. 이 묶음은 기존 seed 42를 다시 학습하지 않습니다. 두 방법의 학습 가중치는 각자 새로 초기화하며, 같은 seed의 모델 파일을 서로 복사하지 않습니다.

## 1. 실행

압축을 풀고, 기존 논문 실험에 사용한 Python 환경을 활성화한 뒤 **원래 데이터 프로젝트 폴더**에서 실행합니다. 아래 `main_seeds_8_77`은 압축을 푼 프로그램 폴더입니다.

학습 전 점검:

```bash
python /압축을푼경로/main_seeds_8_77/run_seed8_77.py --root . --gpus 0 --phase check
```

네 모델 학습과 최종 Raw·Refined 평가:

```bash
python /압축을푼경로/main_seeds_8_77/run_seed8_77.py --root . --gpus 0 --phase all
```

GPU 0 한 장이면 **Strict 8 → Relaxed 8 → Strict 77 → Relaxed 77** 순서로 실행합니다. GPU 두 장을 쓰려면 두 명령 모두 `--gpus 0 1`처럼 같은 번호를 지정합니다. 한 GPU에는 한 작업만 배정합니다. 번호는 물리 GPU 인덱스입니다.

기본 출력 폴더는 `원래프로젝트/main_seeds_8_77`입니다. 별도 위치를 쓰려면 첫 실행부터 `--out /결과저장경로`를 추가합니다. 재개할 때에는 같은 명령·출력 폴더·GPU 배정을 사용합니다.

### 데이터 파일을 자동으로 찾지 못하는 경우

`--root` 아래 기존 COST11 Relaxed 또는 Strict 실험 폴더, `protocols_5m_seed42`, 프로젝트 바로 아래의 `manifest.csv`를 차례로 확인합니다. 파일 내용의 해시가 원래 분할과 일치하는 경우만 선택합니다. 다른 위치에 있다면 다음처럼 전체 manifest를 지정합니다.

```bash
python /압축을푼경로/main_seeds_8_77/run_seed8_77.py --root . --manifest /기존실험경로/manifest.csv --gpus 0 --phase all
```

이 manifest는 **Train 900 + Validation 180 + Synthetic Test 720 + OSM Test 500 = 2,300개**를 모두 포함해야 합니다. 평가용 1,400개 manifest만으로 학습하지 않습니다. 지도·분할·manifest를 새로 생성하거나 seed에 따라 바꾸지 않습니다. 원래 manifest에 기록된 절대 경로에서 기존 `.npy` 배열을 읽습니다.

## 2. 학습 조건

FULL 보상, κ=11, β=0.5, 기존 CNN3 구조와 관측, PPO 설정, 이동·마스크·예산 규칙을 유지합니다. 프로그램에는 실제 Strict와 Relaxed 실행 기록의 해시와 일치하는 원본 학습기·코어·CNN·로깅·재개 모듈을 포함했습니다. 기존 seed 42의 모델 가중치는 포함하지 않았습니다.

| 항목 | 설정 |
|---|---|
| 알고리즘 | 기존 MaskablePPO |
| 병렬 환경 수 | 8 |
| rollout 길이 | 환경당 1,024 step |
| batch / epoch | 2,048 / 4 |
| γ / GAE λ | 1.0 / 0.95 |
| clip / entropy coefficient | 0.2 / 0.0005 |
| value coefficient / gradient norm | 0.5 / 0.5 |
| target KL | 0.02 |
| 네트워크 | 원래 CNN3, 결합 입력 261차원 |
| 지도 순서 기본 seed | 20260506 |
| 학습 seed | 두 방법 모두 8·77 |
| 평가 seed / 초기조건 seed | 기존 42 / 20260907 |

학습 seed는 PyTorch·NumPy·Python 난수, 정책 초기화와 행동 샘플링, 학습 지도 순서·초기조건에 적용됩니다. 지도 순서용 기본값 20260506은 유지하지만 실제 rank별 샘플러 seed는 **20260506 + 1009×학습 seed + rank**입니다. 따라서 seed별 학습 지도 순서가 달라지고, 같은 seed의 두 방법은 같은 rank의 n번째 episode에 같은 지도 순서를 사용합니다. episode 길이가 달라서 같은 전체 timestep에 같은 지도를 처리한다는 뜻은 아닙니다.

Strict는 기존 구현처럼 실제 우회 예산과 예산 관측을 0으로 유지합니다. 공유 환경 설정의 `detour_budget_max=6`은 기존 관측 공간을 보존하는 값이며 Strict에 우회 6회를 허용하는 값이 아닙니다.

## 3. 기존 모델과 같은 단계 경계

새 seed도 **처음부터 5M → 같은 seed의 20M → 같은 seed의 40M**으로 학습합니다. 숫자는 모두 누적 목표입니다. 한 모델에 65M를 학습하는 구성이 아닙니다.

| 단계 | 시작 | 누적 목표 | rollout 경계의 실제 종료 step |
|---|---|---:|---:|
| 1 | 새 가중치·새 optimizer | 5,000,000 | 5,005,312 |
| 2 | 자기 seed의 1단계 정책·Adam 상태 | 20,000,000 | 20,004,864 |
| 3 | 자기 seed의 2단계 정책·Adam 상태 | 40,000,000 | 40,001,536 |

첫 단계는 원래 학습률 식을 그대로 사용합니다.

**LR(t) = max(5×10⁻⁵, 2.5×10⁻⁴ × (1 − t/5,000,000)).**

따라서 약 4M부터 하한에 도달합니다. 5M 이후 두 continuation 단계는 **5×10⁻⁵로 고정**합니다. 새 seed만 40M 전체에 걸쳐 다시 감쇠시키지 않습니다.

5M·20M 경계에서는 기존 continuation 코드가 정책·Adam 상태와 지도 샘플러 진행을 이어받고, 진행 중 episode와 전역 난수 상태를 새 단계의 같은 seed로 시작합니다. 각 rank의 이전 단계 지도 로그를 연결해 sampler의 다음 위치를 재구성합니다. 이 단계 경계에서 버린 미완료 episode를 완주·실패로 추가 집계하지 않습니다.

실행 도중의 일반적인 중단은 위 단계 전환과 다릅니다. 원래 전체 상태 복원 기능으로 해당 단계의 정책·optimizer·환경·진행 중 episode·sampler·난수 상태를 함께 복원합니다. 완료한 단계는 체크포인트 해시를 확인하고 건너뜁니다.

원래 주기에 따라 첫 5M에서는 200k 간격, 이후에는 1M 간격으로 모델 체크포인트를 저장합니다. 별도의 전체 복원 상태도 약 50k 간격의 PPO update 경계에 저장하며, 최근 두 세대를 보관합니다. **이번 묶음의 Validation·Test는 각 seed의 최종 40M 모델에 대해 수행합니다.** 중간 체크포인트로 모델을 선택하지 않습니다.

## 4. 평가와 결과

`--phase all`은 각 모델의 40M 학습이 끝나면 별도 CPU 프로세스에서 Validation 180개, Test 1,220개를 평가합니다. Test는 **Synthetic 720개와 OSM 500개**이며 Validation을 합산하지 않습니다.

평가에서는 학습 seed와 별개로 기존 평가 seed 42와 초기조건 seed 20260907을 유지합니다. 원래 초기조건 함수로 지도별 시작 셀·방향을 재현하고, 허용 행동 중 결정적 argmax를 선택합니다. Raw 지표는 원래 환경의 최종 정보 값을 사용합니다. 저장한 Raw trace에 기존 공통 연결 개선 함수를 적용하고, 이동 규칙·최초 방문순서·도착방향·시간·이동 수를 재생 검증합니다.

각 `strict_seed8`, `relaxed_seed8`, `strict_seed77`, `relaxed_seed77` 폴더에는 다음 결과가 생깁니다.

- `training_result.json`: 최종 모델 경로·해시, 실제 step, 단계별 학습 이력.
- `stages/stage05M`, `stage20M`, `stage40M`: 단계별 모델·학습 로그·전체 복원 상태.
- `evaluation/validation/`, `evaluation/test/`: 분할별 Raw·Refined 결과.
- 각 평가 폴더의 `raw_per_map.csv`, `refined_per_map.csv`, `refinement_audit_per_map.csv`, `summary.json`.
- 지도별 경로 trace, 구간 개선 기록, 캐시 해시.

평가가 중단되면 완료한 지도 결과를 검증 후 재사용합니다. Raw까지 저장되어 있으면 그 trace부터 연결 개선을 이어갑니다. 학습만 먼저 수행하려면 `--phase train`, 완료 모델 평가를 나중에 수행하려면 `--phase evaluate`를 사용합니다.

전체 진행 상태는 출력 폴더의 `experiment_status.json`에 모읍니다. 다음 명령으로 이미 저장된 상태를 다시 정리할 수 있습니다.

```bash
python /압축을푼경로/main_seeds_8_77/run_seed8_77.py --root . --phase report
```

현재 프로그램은 새 seed 8·77의 결과를 생성합니다. 논문에서 **n=3의 평균·표준편차**를 제시할 때에는 기존 seed 42 결과까지 합쳐야 합니다. seed별 실패 지도가 다르면 시간·가중 지연 비교에 사용한 지도 집합을 명시하고, 같은 지도 ID·초기조건의 공통 성공 집합도 함께 확인합니다. 아직 없는 seed 결과를 생성하거나 세 seed의 통계가 이미 계산되었다고 표시하지 않습니다.

## 5. 환경과 확인 범위

원래 실행 환경은 Python 3.10.18, NumPy 2.2.6, PyTorch 2.7.1, Gymnasium 1.1.1, stable-baselines3 2.7.0, sb3-contrib 2.7.0, CUDA runtime 12.6이었습니다. 원래 Python 환경을 사용하는 것을 전제로 하며, 프로그램이 패키지를 자동 설치·업그레이드하지 않습니다. 학습 전 원래 핵심 패키지 버전·CUDA와 원본 데이터 해시를 확인합니다.

이 전달본을 만들면서 원본 18개 runtime 파일과 설정의 해시 대조, 프로그램 문법, 실제 원본 continuation 패치, seed 8·77의 sampler 로그 연결과 재구성을 확인했습니다. 상세 결과는 `verification/verification_report.json`에 기록합니다.

**이 작업 환경에는 원래 Train900 배열, GPU 학습 환경과 PyTorch/SB3가 없으므로 40M 학습이나 실제 새 모델 평가는 아직 실행하지 않았습니다.** 위 명령을 기존 실험 서버에서 실행하면 새 학습을 시작합니다. 코드와 설명을 생성한 상태를 실험 완료 결과로 사용하지 않습니다.

## 포함 파일

`run_seed8_77.py`는 네 작업의 실행·동결·배정을, `seed_training.py`는 원본 학습기와 단계별 재개를, `seed_evaluation.py`는 최종 Raw·Refined 평가를 연결합니다. `runtime/`은 기존 Strict·Relaxed 원본 모듈, `templates/`는 확인한 원래 설정·환경 기록, `vendor/`와 `ch5_vendor/`는 재사용하는 continuation·평가·연결 개선 함수입니다. 실제 학습에는 이 폴더 구성을 유지해야 합니다.

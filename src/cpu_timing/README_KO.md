# UAV 경로계획 CPU 실행시간 측정 V3

**서버에 있는 기존 코드·최종 모델·Test 지도를 직접 사용합니다. 모델 재학습은 필요하지 않습니다.** V3는 원본 `uav_chapter4_suite_v2`와 이전에 수집한 저장소의 폴더 배치를 지원합니다. 측정은 최대 3개 모델 프로세스로 병렬 실행하고 하나의 출력 폴더에 모읍니다.

## 1. 원본 서버에서 바로 실행

원래 학습하던 conda 환경을 활성화하세요. V3를 `uav_chapter4_suite_v2` 안의 **새 `cpu_timing_package_v3` 폴더**로 풀고 아래 명령을 실행합니다. `--suite-root ..`는 바로 위 원본 실험 폴더를 뜻합니다. V1·V2 파일은 덮어쓰지 않습니다.

```bash
cd cpu_timing_package_v3
```

먼저 입력·모델·CPU 설정을 확인합니다. 이 단계는 시간을 측정하지 않습니다.

```bash
python benchmark_cpu.py --suite-root .. \
  --maps-per-source 2 --workers 3 --threads 4 \
  --check-only --out timing_check_v3
```

다음은 실제 경로 생성과 개선을 포함한 소규모 실행입니다.

```bash
python -u benchmark_cpu.py --suite-root .. \
  --maps-per-source 2 --repeats 1 --workers 3 --threads 4 \
  --out timing_smoke_v3
```

Synthetic 2개와 OSM 2개를 6개 모델로 한 번씩 실행하므로 **측정 24회**입니다. 지도는 고정된 hash 순서로 선택합니다. 이 결과는 실행 확인용이며 전체 Test 결과로 사용하지 않습니다.

정상 실행을 확인한 뒤 전체를 측정합니다.

```bash
python -u benchmark_cpu.py --suite-root .. \
  --repeats 3 --workers 3 --threads 4 --out timing_full_v3
```

기본 범위는 Strict·Relaxed × seed 8·42·77의 6개 모델, Synthetic 720개+OSM 500개, 지도별 3회입니다. 총 측정은 **`1,220 × 6 × 3 = 21,960회`**이며 warm-up은 제외합니다. `--workers` 기본값은 1이므로 병렬 실행에는 위와 같이 3을 명시합니다.

## 2. 가지고 있는 파일 배치에 맞춰 선택

`--suite-root`와 `--repo-root` 중 하나만 사용합니다. 아래 상대경로는 V3 폴더를 원본 suite 안에 둔 경우입니다. 파일 위치가 다르면 해당 옵션에 실제 폴더 경로를 지정하세요.

| 기존 입력 | 지정할 옵션 | 확인하는 소스 배치 |
|---|---|---|
| 서버 원본 실험 폴더 — 우선 사용 | `--suite-root ..` | 원본 suite의 등록된 코드·모델·Test 경로 |
| 이전 1GB 수집본 | `--repo-root ../../drones-priority-cpp-1gb` | `src/MAIN_SEEDS_8_77` 및 `audit` 메타데이터 |
| cpp2 수집본 | `--repo-root ../../drones-priority-cpp2` | `src/original/root0/MAIN_SEEDS_8_77` |
| 공개 저장소 배치 | `--repo-root ../../drones-priority-cpp` | `src/main_seeds_8_77` 및 `provenance` |

예를 들어 이미 받은 1GB 수집본으로 소규모 측정을 하려면 다음을 사용합니다.

```bash
python -u benchmark_cpu.py --repo-root ../../drones-priority-cpp-1gb \
  --maps-per-source 2 --repeats 1 --workers 3 --threads 4 \
  --out timing_smoke_v3
```

입력 연결만 바꾸며 새 지도 생성, 전체 파일 검색, 대용량 복사, 다운로드는 수행하지 않습니다. 모델·소스 패키지가 없거나 hash가 등록된 값과 다르면 확인한 정확한 경로와 함께 중단합니다. 파일명을 비슷하게 맞추거나 임의의 모델로 대체하지 않습니다.

### 입력을 찾지 못할 때

```bash
python diagnose_cpu_inputs.py --suite-root .. --out cpu_inputs_report.json
```

생성된 **`cpu_inputs_report.json`과 마지막 오류 메시지**를 함께 전달하세요. 진단기는 정해진 경로와 제한된 깊이에서 작은 메타데이터·소스 목록·모델 파일 존재 여부를 확인합니다. 모델 실행이나 원본 수정은 하지 않으며, 보고서 자체가 입력 검증 통과를 의미하지는 않습니다. 같은 이름의 보고서가 있으면 번호를 붙여 보존합니다.

## 3. 원래 Test와 모델을 유지하는 방법

`reference_test_maps.json`에는 논문 Test 1,220개의 ID·자료군·분할·배열 SHA-256이 들어 있습니다. **실제 지도 `.npy`는 포함하지 않습니다.**

- 원본 manifest가 있으면 Test만 추려 논문의 1,220개 ID와 hash 집합을 대조합니다.
- 원본 manifest가 없으면 내장된 정확한 `map_id`로 원본 root의 상위 `test` 또는 root 안 `test` 경로를 확인합니다.
- 사용되는 배열은 50×50, priority 0–5 및 canonical-array SHA-256을 검증합니다. `.npy` 파일 hash와 배열 hash는 구분합니다.
- source manifest는 검증된 **원본 서버판과 공개판 두 종류만** 허용하며, 각 코드·template·runtime과 최종 6개 checkpoint의 등록된 SHA-256 검사를 유지합니다.

원본과 공개판 `test8_original.py`는 경로 문자열 5곳이 다릅니다. 실제 계산에 사용하는 `evaluate_map`, `initial_condition`, `policy_action`과 A* 구현이 동일한 것으로 확인된 두 패키지를 각각 지원합니다. 알 수 없는 패키지를 자동 등록하거나 hash 검사를 생략하는 fallback은 없습니다.

## 4. CPU와 병렬 실행 조건

`--workers 3 --threads 4`는 **최대 3개 모델 프로세스, 프로세스당 계산 thread 4개**입니다. 사용 가능한 물리 코어에서 겹치지 않는 4개짜리 CPU 그룹 3개를 자동 배정하므로 CPU 번호를 직접 적을 필요는 없습니다.

기본 6개 모델에서는 각 seed의 Strict 다음 Relaxed가 같은 그룹을 사용합니다. CPU 그룹·모델 배정은 `run_config.json`과 metadata에 기록됩니다. `--cpu-ids`를 직접 지정하면 병렬 실행 시 각 그룹을 나눌 후보 CPU 집합으로 사용합니다.

NumPy·BLAS 로딩 전 thread 환경변수를 설정하고 PyTorch intra-op은 4, inter-op은 1로 맞춥니다. `3 × 4 = 12`는 배정하는 코어 수이며 항상 12개 코어가 모두 동작한다는 뜻은 아닙니다. GPU는 사용하지 않습니다.

CPU affinity는 독점 예약이 아닙니다. 다른 CPU 집약 작업과의 경합을 피하고, 이 실행기 하나로 병렬 작업을 관리하세요. **각 경로의 측정시간을 3으로 나누지 않습니다.** 논문에는 최대 3개 동시 프로세스와 프로세스당 thread 수를 함께 보고합니다.

한 worker가 비정상 종료해도 다른 모델은 계속 실행합니다. 하나라도 비정상 종료하면 최종 상태는 `failed`입니다. Raw 경로의 미완주는 정상적인 평가 결과이며 worker 오류와 구별합니다. 최종 집계는 모든 worker가 종료한 뒤 수행합니다.

## 5. 중단 후 재개와 로그

동일한 명령·출력 폴더에 `--resume`만 추가합니다.

```bash
python -u benchmark_cpu.py --suite-root .. \
  --repeats 3 --workers 3 --threads 4 --out timing_full_v3 --resume
```

완료한 반복은 유지하고 남은 반복만 실행합니다. 입력 배치, 모델, 지도, workers, threads, CPU 그룹, 코드와 소프트웨어 조건이 같아야 합니다. **V1·V2 결과는 해당 버전의 원래 코드로 재개**하며, V3에서는 새 출력 폴더를 사용합니다. `--check-only` 출력을 측정 재개에 사용하지 않습니다.

SSH 종료 후에도 유지하려면 선택적으로 다음과 같이 시작합니다.

```bash
nohup python -u benchmark_cpu.py --suite-root .. \
  --repeats 3 --workers 3 --threads 4 --out timing_full_v3 \
  > timing_full_v3_console.log 2>&1 &
```

부모 로그에는 모델 시작·종료가, `timing_full_v3/logs/strict_seed8.log` 같은 개별 로그에는 지도별 진행 상황이 기록됩니다. `nohup`은 연결 종료 시 실행 유지 기능이며 환경이나 CPU 자원을 분리하지 않습니다.

## 6. 측정 시간의 범위

시간 단위는 초이며 wall-clock 시간입니다. V2의 측정·집계 정의를 유지합니다.

| 열 | 의미 |
|---|---|
| `raw_wall_sec` | mask/BFS·정책 결정·환경 전이·진단·trace 구성을 포함한 원본 Raw rollout |
| `refine_validate_wall_sec` | Raw replay·연결 검색·Refined replay·보존 조건 검증을 포함한 refinement |
| `total_plan_wall_sec` | 성공한 경로의 Raw + refinement·검증 시간 |
| `attempt_wall_sec` | 성공 시 전체 계획시간, Raw 실패 시 실패까지의 Raw 시간 |
| `policy_action_only_wall_sec` | 정책 결정 부분만의 누적시간; 전체 계획시간과 구별 |

모델 로딩, 지도 읽기·hash 확인, 초기조건 선정, warm-up, 결과 저장과 KMZ 생성은 제외합니다. worker마다 warm-up 1회가 기본이며 Raw가 실패하면 그 warm-up의 refinement도 생략됩니다. 실제 warm-up 횟수는 metadata에 기록합니다.

Raw 실패 시 A*를 실행하지 않고 refinement·전체 완료계획 시간은 `null` 또는 빈칸으로 둡니다. 실패를 0초로 평균에 넣지 않습니다. 결과는 **진단·검증을 포함한 오프라인 계획시간**이며 실시간 마감이나 이론적 최악 실행시간을 보장하지 않습니다.

## 7. 결과 확인과 재집계

| 파일 | 용도 |
|---|---|
| `SUMMARY.txt`, `status.json` | 전체 완료 여부와 모델별 요약·실행 상태 |
| `paper_table.csv`, `summary.csv` | 성공·공통 성공 및 전체 시도·실패 집계 |
| `per_map.csv`, `per_trial.csv` | 지도별 반복 요약과 개별 측정값 |
| `run_config.json`, `hardware.json`, `models/*_metadata.json` | 입력 경로·hash, CPU 그룹·모델 배정, 버전·warm-up 정보 |
| `trials/*.jsonl`, `logs/*.log` | 재개용 기록과 진행 로그 |

먼저 각 모델·지도의 반복 3회 **중앙값**을 구하고, 지도별 중앙값들의 평균·SD·P95·최댓값을 집계합니다. P95는 선형 보간이며 SD는 지도 간 변동입니다. `max_single_trial`은 개별 반복의 최대값으로, 지도별 중앙값의 최대와 별도로 기록합니다.

`successful`은 해당 모델의 성공 지도, `common_success`는 선택한 모든 모델의 공통 성공 지도입니다. 기본 실행에서는 6개 모델 공통 집합입니다. 모델 간 비교는 공통 분모를 함께 확인하고 Synthetic·OSM을 구분하세요. 필요한 반복이 모두 끝난 지도만 요약에 포함하며 `PARTIAL`을 전체 완료 결과로 보고하지 않습니다.

모든 worker가 종료한 뒤 저장된 결과만 다시 집계할 수 있습니다.

```bash
python benchmark_cpu.py --summarize-only timing_full_v3
```

## 8. 환경 버전과 검증 기록

Linux, Python 3.10 이상이 필요합니다. 원본 환경의 주요 버전은 NumPy 2.2.6, PyTorch 2.7.1, Gymnasium 1.1.1, stable-baselines3 2.7.0, sb3-contrib 2.7.0입니다. 기존 정상 학습 환경을 우선 사용하며 이 측정 때문에 일괄 재설치하지 마세요.

기본값은 위 다섯 패키지의 설치 버전 일치를 검사합니다. 다른 환경으로 측정하기로 정했다면 기존 명령에 `--allow-version-difference`를 명시하고 새 출력 폴더를 사용합니다. 버전 접미사 차이도 기록하며, 소스·모델 hash와 구조 검사는 유지합니다. 다른 소프트웨어 환경의 결과를 원본 환경과 동일 조건으로 표현하지 않습니다.

패키지 검증의 실제 범위와 한계는 `VALIDATION.txt`에 기록합니다. 기준 공개 commit은 [`8e19d8822983245b811b83e88c99fd2be548a044`](https://github.com/ohrgon/drones-priority-cpp/tree/8e19d8822983245b811b83e88c99fd2be548a044)입니다. 기존 [worker 실행기](https://github.com/ohrgon/drones-priority-cpp/blob/8e19d8822983245b811b83e88c99fd2be548a044/src/main_seeds_8_77/ch5_vendor/test8_original.py#L1072), [Raw rollout](https://github.com/ohrgon/drones-priority-cpp/blob/8e19d8822983245b811b83e88c99fd2be548a044/src/main_seeds_8_77/ch5_vendor/test8_original.py#L620), [연결 개선](https://github.com/ohrgon/drones-priority-cpp/blob/8e19d8822983245b811b83e88c99fd2be548a044/src/main_seeds_8_77/ch5_common.py#L195)을 확인할 수 있습니다.

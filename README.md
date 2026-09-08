# Team4 — video-SALMONN-2+ 기반 Audio-Visual Temporal Video Grounding

Audio-Visual TVG (multi-segment) 태스크를 위한 **SFT → GDPO(RL) → 평가** 파이프라인.
video-SALMONN-2+ 백본에 **Time-Token Interleaving (TTI)** 와 **GDPO** 를 적용한 논문 제출 코드.

Backbone reference: [video-SALMONN-2 (ByteDance)](https://github.com/bytedance/video-SALMONN-2)

---

## 📁 레포 구조

이 레포에는 **두 개의 `video_SALMONN2_plus` 소스 트리**가 있다. 논문 실험이 실제로
두 트리에서 각각 수행됐기 때문에, 재현성을 위해 병합하지 않고 단계별로 분리해 둔다.

| 트리 | 역할 | 특징 |
|---|---|---|
| `sft/video_SALMONN2_plus/`  | **Stage 1 — SFT** 학습 | 모듈별 LR/WD 오버라이드(`lora_lr`/`embed_lr`/`lm_head_lr`/`visual_merger_lr`/`audio_*_lr`), embed·lm_head row-mask, ordinal loss (`lambda_ord`, 최종 실험에선 미사용) |
| `video_SALMONN2_plus/` (루트) | **Stage 2 — GDPO** 및 **Stage 3 — 평가** 백본 | TTI 4개 모드 (`off`/`special_token`/`natural_text`/`from_to`) 완전 구현 + 디버그 덤프 |

```
Team4/
├── sft/                        # Stage 1: SFT 학습
│   ├── video_SALMONN2_plus/    #   SFT 전용 소스 트리
│   └── scripts/                #   train_unpucha_sft.sh (최종), train_v8*.sh, merge_v8_*.py
├── video_SALMONN2_plus/        # GDPO/평가용 백본 소스 트리 (TTI 구현체)
├── gdpo/                       # Stage 2: GDPO(RL) 학습
│   ├── gdpo_trainer_batch.py   #   트레이너 (단일 진입점)
│   ├── reward_functions_rM_sep3*.py  # 리워드 (+ ablation: nocount/noglobal/nolocal/noprecision)
│   ├── config_sep3*.yaml       #   하이퍼파라미터
│   ├── run_rMsep3_*.sh         #   런처
│   └── demo_app.py             #   Gradio 데모
├── eval/                       # Stage 3: 추론 + mIoU 평가
│   └── analysis/              #   보조 분석·재채점 스크립트 + 파서 검증
├── tools/                      # 보조: TTI 회귀검증, debug dump, time-token 추가, 데이터 준비
├── docs/                       # 상세 가이드 (GDPO 학습법 / 평가법 / 결과 기록)
└── paths.example.env           # 경로 템플릿
```

---

## 🔧 셋업

### 1) 환경 변수

```bash
cp paths.example.env paths.env   # BASE_ROOT / DATA_ROOT 두 루트만 잡으면 나머지 자동 파생
```

> ⚠️ `paths.env` 는 개인 키(`WANDB_API_KEY`) 포함 → `.gitignore` 대상. 템플릿만 커밋한다.

| 변수 | 의미 |
|---|---|
| `BASE_ROOT` | 프로젝트/데이터 JSON 공통 루트 |
| `DATA_ROOT` | 체크포인트·원본 영상 공통 루트 |
| `CKPT_DIR`  | 체크포인트 루트 (`${DATA_ROOT}/checkpoints`) |
| `TRAIN_DIR` / `TEST_DIR` | 학습 / 평가 JSON |
| `EVAL_DIR`  | 평가 결과 출력 |
| `WANDB_ENTITY` / `WANDB_API_KEY` | WandB 로깅 |

### 2) 베이스 체크포인트

1. [video-SALMONN2_plus_7B_full](https://huggingface.co/tsinghua-ee/video-SALMONN2_plus_7B_full) 다운로드
2. time-token 11개 추가 → `${CKPT_DIR}/base/video_salmonn2_plus_7B_time_tokens`

```bash
python tools/sft/add_time_tokens_salmonn2plus.py
python tools/sft/verify_time_tokens.py          # 등록 확인
```

### 3) 데이터

| 경로 | 용도 |
|---|---|
| `${TRAIN_DIR}/unpucha_sft.json` | Stage 1 SFT (balanced 3k: charades/puvalor/UnAV) |
| `${TRAIN_DIR}/unpucha_v2.json`  | Stage 2 GDPO |
| `${TEST_DIR}/{TESTSET}/chunk_*.json` | Stage 3 평가 (chunk 단위) |

> chunk 분할: `python eval/_chunk_helpers.py split --test_json <원본> --chunks_dir data/test/<NAME>/`

---

## 📦 아카이브 & 복원

서버 정리(2026-09-08)로 대용량 산출물은 전부 Google Drive 로 옮겼다. 레포에는 **코드만**
남아 있으므로, 재현하려면 아래 세 디렉터리를 먼저 내려받아야 한다.

| 원래 로컬 경로 | gdrive 원격 | 크기 | 내용 |
|---|---|---|---|
| `workspace/checkpoints` | `gdrive:checkpoints` | 59 GB  | base(time-token 추가본), SFT/GDPO LoRA 및 머지 체크포인트 |
| `workspace/data`        | `gdrive:data`        | 407 MB | 학습/평가 JSON (`train/`, `val/`, `test/<TESTSET>/chunk_*.json`) |
| `workspace/outputs`     | `gdrive:outputs`     | 378 MB | 추론 결과(`test_results_rank*.json`) + eval summary + `table.txt` |

> ⚠️ `outputs/**/.merged_model/` (평가 시 LoRA→base 자동 머지본, 총 68 GB) 은 **백업하지 않는다.**
> base + LoRA 가 `gdrive:checkpoints` 에 있으므로 `eval.sh` 가 다시 만든다. 로컬 `outputs` 는
> 77 GB 지만 그중 76.6 GB 가 이 파생물이고, 실제 결과물은 1,618 개 / 378 MB 다.

### 전체 복원

```bash
WS=$HOME/workspace     # = DATA_ROOT / BASE_ROOT 가 가리키는 곳

rclone copy gdrive:data        $WS/data        --transfers 8 --checkers 16 --progress
rclone copy gdrive:checkpoints $WS/checkpoints --transfers 8 --checkers 16 --drive-chunk-size 32M --progress
rclone copy gdrive:outputs     $WS/outputs     --transfers 8 --checkers 16 --progress
```

`outputs` 는 결과 JSON 뿐이라 금방 받아진다. 재평가할 때 `.merged_model` 이 자동으로 다시 생기며
런당 12~18 GB 를 먹으니 디스크 여유를 확인할 것.

> `--transfers` × `--drive-chunk-size` 가 곧 rclone 의 메모리 사용량이다. RAM 이 작은 서버면
> `--transfers 4 --drive-chunk-size 16M` 으로 낮출 것. (이관 당시 서버: RAM 3 GB / 2 vCPU)

### 부분 복원 (권장)

표를 다시 뽑거나 수치만 확인할 목적이면 `outputs` 전체를 받을 필요가 없다.
런 하나는 보통 수백 MB ~ 수 GB 다.

```bash
rclone lsd  gdrive:outputs/gdpo                                    # 런 목록 확인
rclone copy gdrive:outputs/gdpo/<RUN_ID> $WS/outputs/gdpo/<RUN_ID> --progress
python eval/maketable.py $WS/outputs/gdpo                          # 받은 것만으로 표 생성
```

체크포인트도 마찬가지로 필요한 스텝만:

```bash
rclone copy gdrive:checkpoints/gdpo/<RUN_ID>/checkpoint-1000 \
            $WS/checkpoints/gdpo/<RUN_ID>/checkpoint-1000 --progress
```

### 복원 후

`paths.env` 의 `BASE_ROOT` / `DATA_ROOT` 를 새 서버 경로로 맞추면 나머지 경로는 자동 파생되고
파이프라인이 그대로 돈다 (`paths.env` 는 gitignore 대상 — `paths.example.env` 에서 복사).

---

## 🚀 재현 파이프라인

### Stage 1 — SFT

time-token **형식**을 가르치는 단계 (grounding 자체는 Stage 2 에 위임).
UnAV 단독 학습은 GT ≤ 60s 라 백/십자리 time-token 이 미학습 → ActivityNet 예측이 ~50s 에 갇힌다.
최종 실험은 GT end 최대 214s 를 커버하는 **balanced 3k 셋**을 쓴다.

```bash
bash sft/scripts/train_unpucha_sft.sh          # 최종 SFT (LoRA r=16, 1 epoch ≈ 375 step)
```

산출물: `${CKPT_DIR}/sft/salmonn2plus_v8_unpucha_sft/checkpoint-375/` (LoRA adapter)

### Stage 1.5 — LoRA 를 base 에 머지

GDPO 는 **머지된 base** 에서 출발한다.

```bash
python sft/scripts/merge_v8_unpucha_to_base.py
```

산출물: `${CKPT_DIR}/base/salmonn2p_7b_unpucha_v8`

### Stage 2 — GDPO (RL)

temporal-IoU 계열 reward 로 grounding 강화학습. 상세는
[docs/GDPO-학습방법-총정리.md](docs/GDPO-학습방법-총정리.md).

```bash
bash gdpo/run_rMsep3_unpucha_v8_ttifix.sh      # 메인 실험 (TTI on)

# Ablation
bash gdpo/run_rMsep3_unpucha_v8_ttifix_nocount.sh      # count reward 제거
bash gdpo/run_rMsep3_unpucha_v8_ttifix_noglobal.sh     # global reward 제거
bash gdpo/run_rMsep3_unpucha_v8_ttifix_nolocal.sh      # local reward 제거
bash gdpo/run_rMsep3_unpucha_v8_ttifix_noprecision.sh  # precision reward 제거
bash gdpo/run_rMsep3_noscaling_ttioff.sh               # TTI off
bash gdpo/run_rMsep3_nosft.sh                          # SFT 없이 base 에서 바로 RL
bash gdpo/run_rMsep3_natural.sh                        # natural_text 모드
```

베스트 체크포인트 선택: `python gdpo/select_best_ckpt.py`

### Stage 3 — 평가

`eval/eval.sh` 가 추론(LoRA→base 자동 머지 포함)과 mIoU 평가를 모두 수행한다.
인자 전체 표는 [docs/평가-방법-총정리.md](docs/평가-방법-총정리.md).

```bash
cd eval

# 체크포인트 추론 + 평가
bash eval.sh STAGE=gdpo CKPT_MODEL_ID=<RUN> CKPT_STEP=1000 \
     TEST_JSON=${TEST_DIR}/unav100_v2_500.json GPUS=0

# 베이스 모델만
bash eval.sh CKPT_STEP=base TEST_JSON=${TEST_DIR}/unav100_v2_500.json

# 이미 추론된 결과 재평가 (GPU 불필요)
bash eval.sh MODE=eval RESULTS=<out_dir>/test_results_rank0.json TEST_JSON=<GT>.json

# 결과 표 생성 (해당 폴더 하위의 모든 summary → table.txt)
python maketable.py ${EVAL_DIR}/gdpo
python maketable.py ${EVAL_DIR}/gdpo --full   # 구버전 21열 (F1@0.9, SCR, R@θ, gt/pred 평균)
```

결과: `${EVAL_DIR}/<branch>/fps<N>_<format>/<TESTSET_TAG>/eval_miou_summary.json`

#### 보고 지표

`maketable.py` 기본 표(12열). 보고할 때는 아래 **세 계열을 항상 같이** 싣는다 —
mIoU 만으로는 세그먼트를 몇 개 잡았는지가 안 보이고, F1 만으로는 위치 품질이 안 보인다.

| 열 | 의미 |
|---|---|
| `sample_mIoU` | 샘플 단위 All_IoU 평균 (`sample_miou_summary.json`) |
| `F1@0.1/0.3/0.5/0.7` | pairwise(best-match 세그먼트) 기준 F1 (`pairwise_miou_summary.json`) |
| `CountF1` | `USA`·`OSA` 의 조화평균. 하나라도 0 이면 0 |
| `USA` (= CR*) | N_gt≥2 에서 chance 보정 개수일치도. under-segmentation 해소력 |
| `OSA` (= 1−FMR) | N_gt=1 을 쪼개지 않은 비율. over-segmentation 억제력 |

> ⚠️ `sample_mIoU` 는 **sample 단위**, `F1@θ` 는 **pairwise 단위**로 계열이 다르다.
> 열 이름에서 `sample` 을 떼고 그냥 `mIoU` 로 적지 말 것.
>
> ⚠️ 파싱 실패(pred 0개)가 많은 모델은 `N_pred≤1` 로 잡혀 `OSA` 가 부풀 수 있다.
> 비교 전 summary 의 `parse_fail` 을 반드시 같이 확인할 것.

#### 보조 분석 스크립트 (`eval/analysis/`)

`eval.sh` 가 만든 결과 JSON 을 다시 파고들 때 쓰는 것들. 전부 `Team4/eval` 을 `sys.path` 에
넣고 `eval_miou.py` / `count_f1.py` 의 채점 함수를 그대로 재사용하므로 본 평가와 수치가 맞물린다.

| 스크립트 | 용도 |
|---|---|
| `report_metrics.py` | **보고 표준 지표를 한 번에** — F1@0.1/0.3/0.5/0.7 + mIoU + CountF1 |
| `recompute_countf1_unav.py` | UnAV-100 USA/OSA/CountF1 재계산. summary 를 믿지 않고 `test_results_rank0.json` 에서 다시 뽑는다 |
| `countf1_mae_123plus.py` | N_gt = 1/2/3+ 버킷별 count MAE 및 CountF1 |
| `breakdown_by_ngt.py` | UnAV-100 결과를 N_gt(정답 세그먼트 수)별로 재집계 |
| `osa_charades.py` | Charades OSA(over-segmentation avoidance) 계산 |
| `score_charades_all.py` | Charades-STA 전체(3,720) 재채점. ChronusOmni `cal_iou.py` 와 동일 규칙 |
| `parse_rate.py` | 파싱 성공률. 정의는 `eval_miou.py` 와 동일 |
| `run_unav_eval_1200.sh` / `run_countf1_1200.sh` / `run_noaudio_eval.sh` / `finish_charades_eval.sh` | 장시간 평가 배치 런처 (선행 작업 pid 대기 → 평가 → 집계) |
| `parser_verification/` | LLM 으로 파서 출력을 교차검증 (`pv_data`/`pv_llm`/`pv_metrics`/`pv_report`) |

경로는 스크립트 위치에서 유도한다 — `Team4/eval/analysis/` 에 있다는 전제로 `WS`(workspace)
와 `EV`(Team4/eval) 를 잡는다. 다른 배치로 쓰려면 `WORKSPACE` 환경변수로 덮어쓴다.

```bash
export WORKSPACE=/path/to/workspace     # 기본값이 안 맞을 때만
python3 eval/analysis/report_metrics.py ${EVAL_DIR}/gdpo/<RUN>/checkpoint-1000/fps5_tti/<TESTSET>
```

> 배치 런처(`run_*.sh`)는 원래 이 서버 전용이라 `PY`(conda python), `WAIT_PID`(선행 작업)
> 같은 값이 박혀 있었다. 지금은 환경변수로 덮어쓸 수 있게 바꿨지만, 다른 서버에서는
> `PY` / `WAIT_PID` / `RUN` / `CK` 를 상황에 맞게 확인하고 쓸 것.

---

## 🎯 Time-Token Interleaving (TTI)

비디오/오디오 청크 사이에 시간 마커를 삽입해 temporal grounding 을 강화. `tti_time_format` 로 제어.

| 모드 | 청크당 마커 | 예시 (1.5s) | 설명 |
|---|---|---|---|
| `off` (기본)    | 0 토큰  | (없음) | Qwen2.5-VL 베이스라인 |
| `special_token` | 5 토큰  | `<t0><t0><t1><tdot><t5>` | VTG-LLM 식 special token |
| `natural_text`  | 9 토큰  | `second{0001.5}` | 자연어 (zero-pad) |
| `from_to`       | 14 토큰 | `From <t*>×5 to <t*>×5` | 출력 GT 와 동일 포맷 |

> 출력(GT) 형식은 모드와 무관하게 항상 special_token — 베이스 모델이 time-token 임베딩을 갖고 있다.

검증 (7/7 PASS 면 정상):

```bash
bash tools/tti/run_all.sh ${CKPT_DIR}/base/video_salmonn2_plus_7B_time_tokens
```

---

## 🛠 Config

| 파일 | 용도 | TTI 키 |
|---|---|---|
| `tools/sft/config.yaml`  | (구) SFT 런처용 하이퍼파라미터 | `BASE_INTERVAL`, `TTI_TIME_FORMAT` |
| `gdpo/config_sep3.yaml`  | GDPO 학습 (reward/clip/num_generations …) | `tti_mode` |
| `eval/config.yaml`       | 평가 (해상도/프레임/deepspeed) | `BASE_INTERVAL`, `TTI_TIME_FORMAT` |

다른 실험은 `cp config_sep3.yaml my_config.yaml` 후 `--config` 로 지정.

---

## 🧪 Debug

```bash
bash tools/debug/smoke_dump_all_modes.sh        # 모드별 샘플 dump
bash tools/debug/sweep_dump.sh                  # BASE_INTERVAL × VIDEO_MAX_FRAMES sweep
python tools/debug/compare.py --in_dir _debug_out/... --format csv
```

### 알려진 함정 — `repetition_penalty`

`generation_config.json` 의 `repetition_penalty=1.05` 는 **명시하지 않으면 greedy 에서도
적용된다**. 타임토큰 출력은 같은 숫자 토큰을 반복 사용하므로 페널티가 예측 시각 값 자체를
왜곡하고, 종료(`.`+EOS)까지 억제해 세그먼트를 과분할한다 (학습 pred/샘플 1.36 → 평가 2.10).

학습측(rollout·val)과 `train_qwen.py` 의 debug_interleave generate 는 모두
`repetition_penalty=1.0` 을 **명시**한다. 새 `generate()` 호출을 추가할 때도 반드시 같이 넣을 것 —
안 넣으면 학습/평가 지표가 조용히 어긋난다.

## 📄 License

Apache-2.0 (see `LICENSE`). 서드파티 라이선스는 `third-party-license/` 참고.

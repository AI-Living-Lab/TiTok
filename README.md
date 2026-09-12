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
├── tools/                      # 보조: TTI 회귀검증, debug dump, time-token 추가, 데이터 준비
├── docs/                       # 상세 가이드 (GDPO 학습법 / 평가법 / 결과 기록)
├── tools/data_manifests/       # Charades-STA 재현용 파일 목록 (아래 "데이터" 참고)
├── setup.sh                    # RunPod 서버 환경 부트스트랩 (rclone/conda/WandB env)
└── paths.example.env           # 경로 템플릿
```

---

## 🔧 셋업

### 0) 서버 부트스트랩 (RunPod)

새 RunPod 인스턴스에서 시작할 때:

```bash
source setup.sh   # rclone(/workspace/home) + miniconda PATH + WandB env 설정
```

> ⚠️ `setup.sh` 의 `WANDB_API_KEY` 는 플레이스홀더. 실제 키로 교체해 쓰되 **커밋 금지**.

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

`tools/data_manifests/`: Charades-STA 원본 영상 세트를 다시 받을 때 필요한 파일 목록 3종.

| 파일 | 용도 |
|---|---|
| `charades_sta_needed_files.txt` | Charades-STA 평가에 실제로 쓰이는 mp4 목록 (`Charades_v1/<id>.mp4`) — 전체 데이터셋 대신 이 목록만 받으면 됨 |
| `charades_sta_include.txt` | 위 목록의 glob 패턴 버전 (`**/<id>.mp4`) — 다운로드 스크립트 필터용 |
| `charades_sta_test.txt` | Charades-STA 평가 GT 원본(`구간 시작 끝##caption`) |

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

# 결과 표 생성
python maketable.py
```

결과: `${EVAL_DIR}/<branch>/fps<N>_<format>/<TESTSET_TAG>/eval_miou_summary.json`

과거 결과 스냅샷(참고용, 재현 시 `maketable.py` 로 재생성 권장): [docs/results_snapshot_2026-08-30.txt](docs/results_snapshot_2026-08-30.txt)

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

## ☁️ 백업 (Google Drive)

서버(구 RunPod team404 인스턴스) 정리 시점(2026-09-12)에 아래 3개 디렉토리를
Google Drive(`ewhaailab2026@gmail.com`)로 백업함. `.merged_model/`(머지된 풀 체크포인트,
용량이 커서 재현 시 Stage 1.5 로 재생성 가능)은 `outputs` 백업에서 제외.

| 로컬 (RunPod) | Google Drive |
|---|---|
| `${CKPT_DIR}` (`/workspace/checkpoints`) | `gdrive:checkpoints` |
| `${TRAIN_DIR}`/`${TEST_DIR}` 상위 (`/workspace/data`) | `gdrive:data` |
| `${EVAL_DIR}` (`/workspace/outputs`, `.merged_model/` 제외) | `gdrive:outputs` |

새 서버에서 복원:

```bash
source setup.sh   # rclone 바이너리 + gdrive remote(rclone.conf) 준비
rclone copy gdrive:checkpoints ${CKPT_DIR} --transfers 8 --checkers 8
rclone copy gdrive:data        /workspace/data --transfers 8 --checkers 8
rclone copy gdrive:outputs     ${EVAL_DIR} --transfers 8 --checkers 8
```

## 📄 License

Apache-2.0 (see `LICENSE`). 서드파티 라이선스는 `third-party-license/` 참고.

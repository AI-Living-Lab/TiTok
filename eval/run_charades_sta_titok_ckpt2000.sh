#!/bin/bash
# ============================================================
# sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling / checkpoint-2000
# charades_sta_titok(3719, chronus/museg 기준셋과 동일 정렬) 청크 재추론+채점.
#
# 왜 다시 돌리나:
#   기존 outputs/gdpo/.../checkpoint-2000/fps5_tti/charades_rlp_noaudio 결과는
#   오디오 없는 입력(no-audio)에서 TTI 시간마커가 아예 안 들어가던 버그
#   (rope2d.py/dataset.py, commit 38c2306 "Before TTI..." → 3948326 "After TTI...")
#   가 있던 코드로 채점됐다. 이 저장소 HEAD 는 이미 그 수정을 포함하므로
#   (git merge-base --is-ancestor 3948326 HEAD 로 확인), 지금 코드로 다시 돌리면
#   자동으로 수정 반영본(=예전에 수동으로 만들었던 "fps5_ttifix" 급) 결과가 나온다.
#
#   TTI_DEBUG=1 로 [TTI-DBG] 로그를 켜서(첫 3샘플만) 마커가 실제로 삽입되고
#   rope_index 가 tti_active=True 로 마커-인지 분기를 타는지 직접 확인한다.
#
#   OUT_DIR 을 fps5_tti_verified 로 새로 둬서, 검증 안 된 기존 fps5_tti 결과를
#   덮어쓰지 않고 별도 행으로 남긴다 (maketable.py 의 cfg 열로 구분됨).
#
# 사전 준비(1회):
#   - data/test/charades_sta_titok.json / charades_sta_titok/chunk_*.json  (빌드 완료)
#   - conda env "titok" (Chronus128 클론 + transformers==4.51.3, peft==0.15.2 업그레이드)
#   - paths.env (BASE_DIR/CKPT_DIR/TRAIN_DIR/TEST_DIR/EVAL_DIR)
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONDA_ENV="${CONDA_ENV:-titok}"
export TTI_DEBUG=1
export TTI_DEBUG_SAMPLES=3

GPUS="${GPUS:-0}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling/checkpoint-2000/fps5_tti_verified/charades_sta_titok}"

bash "$SCRIPT_DIR/eval.sh" \
    MODE=infer \
    CHUNK=on \
    STAGE=gdpo \
    CKPT_MODEL_ID=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling \
    CKPT_STEP=2000 \
    BASE_MODEL_ID=base/salmonn2p_7b_unav_v8 \
    TESTSET=charades_sta_titok \
    TESTSET_TAG=charades_sta_titok \
    LABEL=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling \
    TTI_TIME_FORMAT=special_token \
    NATURAL=off \
    OUT_DIR="$OUT_DIR" \
    GPUS="$GPUS"

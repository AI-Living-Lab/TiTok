#!/bin/bash
# ============================================================
# run_noaudio_eval.sh — no-audio 런의 checkpoint 를 UnAV-100(TiTok) 로 평가.
#   ck400 → GPU0, ck200 → GPU1 병렬. OUT_DIR/.merged_model 과 .chunk_workdir 이
#   체크포인트별로 갈리고 master_port 도 GPU 번호로 갈려 충돌 없음.
#
#   ⚠️ 기존 run_unav_eval_1200.sh 대비 고친 것 2가지:
#     1) BASE_MODEL_ID=base/salmonn2p_7b_unav_v8
#        (checkpoints/base/ 는 rclone 이 salmonn2p_7b_unpucha_v8 를 flat 하게
#         풀어놓은 상태라 BASE_MODEL_ID=base 로 두면 **다른 base 에 머지**된다.)
#     2) TESTSET=unav100_titok_noaudio
#        (학습에서 오디오를 뺐으므로 평가 입력도 같아야 한다. audio 키 제거판 —
#         data/strip_audio.py 로 생성, 3455개 동일.)
#
#   사용: bash run_noaudio_eval.sh          # 두 개 병렬
#   중단: kill $(cat $WS/.noaudio_eval.pid)
# ============================================================
set -uo pipefail
# 스크립트 위치(Team4/eval/analysis) 기준으로 workspace 유도. 환경변수로 덮어쓸 수 있다.
WS=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export WS                      # 내장 python heredoc 에서 os.environ["WS"] 로 읽는다
RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling_noaudio
TESTSET=unav100_titok_noaudio
BASE=base/salmonn2p_7b_unav_v8

echo $$ > $WS/.noaudio_eval.pid
export PATH="${CONDA_BIN:-$HOME/miniconda3/bin}:$PATH"   # 비대화형 셸에 conda 없음
export CONDA_ENV=salmonn2plus                       # eval.sh 기본값은 salmonn2p

run_one() {
  local CK=$1 GPU=$2
  local LOG=$WS/outputs/gdpo/$RUN/unav_eval_ck$CK.log
  mkdir -p "$(dirname "$LOG")"
  {
    echo "=== START ck$CK gpu$GPU $(date -Is) ==="
    bash "$WS/Team4/eval/eval.sh" CHUNK=on STAGE=gdpo \
      CKPT_MODEL_ID="$RUN" CKPT_STEP="$CK" \
      BASE_MODEL_ID="$BASE" \
      TESTSET="$TESTSET" GPUS=$GPU
    echo "=== DONE ck$CK rc=$? $(date -Is) ==="
  } >>"$LOG" 2>&1
}

run_one 400 0 &
P4=$!
run_one 200 1 &
P2=$!
wait $P4; echo "ck400 종료"
wait $P2; echo "ck200 종료"

# 머지 캐시 회수 (체크포인트당 ~17GB)
for CK in 400 200; do
  rm -rf "$WS/outputs/gdpo/$RUN/checkpoint-$CK/fps5_tti/$TESTSET/.merged_model" \
         "$WS/outputs/gdpo/$RUN/checkpoint-$CK/fps5_tti/$TESTSET/.chunk_workdir" 2>/dev/null
done
echo "=== ALL DONE $(date -Is) ==="
rm -f $WS/.noaudio_eval.pid

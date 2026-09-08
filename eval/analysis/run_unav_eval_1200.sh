#!/bin/bash
# ============================================================
# run_unav_eval_1200.sh
#   1) 진행 중인 GRPO 학습(step 1600 저장 확인 후) 종료
#   2) checkpoint-1200 을 TiTok UnAV-100 프로토콜로 평가
#      (기존 eval.sh / eval_miou.py / maketable.py 그대로 사용)
#
#   ⚠️ 디스크 여유 30G뿐 → LoRA 머지 산출물(~18GB)은 /dev/shm 에 두고
#      결과 json/summary 만 실제 경로로 회수한다.
#
#   중단: kill $(cat $WS/.run_unav_eval.pid)
# ============================================================
set -uo pipefail

# 스크립트 위치(Team4/eval/analysis) 기준으로 workspace 유도. 환경변수로 덮어쓸 수 있다.
WS=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export WS                      # 내장 python heredoc 에서 os.environ["WS"] 로 읽는다
RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_GRPO
CK=1200
TESTSET=unav100_titok
TRAIN_PARENT=7879      # torchrun
TRAIN_CHILD=7919       # 실제 학습 프로세스

WORK=/dev/shm/titok_unav_eval/ck$CK
REAL=$WS/outputs/gdpo/$RUN/checkpoint-$CK/fps5_tti/$TESTSET
LOG=$WS/outputs/gdpo/$RUN/unav_eval_ck$CK.log

mkdir -p "$(dirname "$LOG")" "$REAL"
exec >>"$LOG" 2>&1
echo ""
echo "============================================================"
echo "=== START $(date -Is)  pid=$$"
echo "============================================================"
echo $$ > $WS/.run_unav_eval.pid

# ---------- 1) 학습 종료 ----------
if kill -0 "$TRAIN_CHILD" 2>/dev/null; then
    echo "[STOP] 학습 종료 요청 (SIGTERM) $(date -Is)"
    kill -TERM "$TRAIN_PARENT" 2>/dev/null
    kill -TERM "$TRAIN_CHILD"  2>/dev/null
    for i in $(seq 1 60); do
        kill -0 "$TRAIN_CHILD" 2>/dev/null || break
        sleep 5
    done
    if kill -0 "$TRAIN_CHILD" 2>/dev/null; then
        echo "[STOP] SIGTERM 무응답 → SIGKILL"
        kill -9 "$TRAIN_PARENT" "$TRAIN_CHILD" 2>/dev/null
        sleep 20
    fi
    echo "[STOP] 학습 종료 확인 $(date -Is) — GPU 메모리 반환 대기 90s"
    sleep 90
else
    echo "[STOP] 학습 프로세스 이미 종료됨"
fi

# ---------- 2) 자원 확인 ----------
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
echo "[RES] gpu_free=$(gpu_free_mb)MiB shm=$(df -BG --output=avail /dev/shm | tail -1) disk=$(df -BG --output=avail / | tail -1)"
if [ "$(gpu_free_mb)" -lt 60000 ]; then
    echo "[WARN] GPU 여유 <60GB — 30s 더 대기"; sleep 30
    echo "[RES] gpu_free=$(gpu_free_mb)MiB"
fi

# ---------- 3) 평가 (기존 eval.sh 그대로) ----------
mkdir -p "$WORK"
# 이전 부분결과가 있으면 이어서 (resume)
cp "$REAL/test_results_rank0.json" "$WORK/" 2>/dev/null
cp "$REAL/.chunk_idx"              "$WORK/" 2>/dev/null

echo "[RUN] eval.sh 시작 $(date -Is)"
# eval.sh 를 그대로 쓰기 위한 환경 보정 2가지 (eval.sh 자체는 수정하지 않음):
#  1) 비대화형 셸엔 conda 가 PATH 에 없어 eval.sh 의
#     _cbase="$(conda info --base)" 가 127 로 죽는다(set -e) → PATH 에 추가
#  2) eval.sh 의 conda env 기본값이 'salmonn2p' 인데 이 서버는 'salmonn2plus'
export PATH="${CONDA_BIN:-$HOME/miniconda3/bin}:$PATH"
export CONDA_ENV=salmonn2plus
bash "$WS/Team4/eval/eval.sh" CHUNK=on STAGE=gdpo \
    CKPT_MODEL_ID="$RUN" CKPT_STEP="$CK" \
    BASE_MODEL_ID=base \
    TESTSET="$TESTSET" OUT_DIR="$WORK" GPUS=0 \
    || echo "[WARN] eval.sh 비정상 종료 — 부분 결과는 회수"

# ---------- 4) 결과 회수 ----------
for f in test_results_rank0.json pairwise_miou_summary.json union_miou_summary.json \
         sample_miou_summary.json eval_miou_progress.jsonl inference.log .chunk_idx; do
    [ -f "$WORK/$f" ] && cp "$WORK/$f" "$REAL/$f"
done
n=$(python3 -c "import json;print(len(json.load(open('$REAL/test_results_rank0.json'))))" 2>/dev/null || echo 0)
echo "[RUN] 회수 완료 n=$n/3455 $(date -Is)"

# ---------- 5) 집계표 (기존 maketable.py) ----------
rm -rf /dev/shm/titok_unav_eval
python3 "$WS/Team4/eval/maketable.py" "$WS/outputs/gdpo" || echo "[WARN] maketable 실패"

echo "=== DONE $(date -Is) ==="
rm -f $WS/.run_unav_eval.pid

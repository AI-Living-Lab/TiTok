#!/bin/bash
# ============================================================
# finish_charades_eval.sh
#   실행 중인 GRPO 학습이 끝나기를 기다렸다가,
#   sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling 의
#   charades_rlp_noaudio 미완성 평가(ckpt-2000)를 재개해 완성한다.
#   (ckpt-1600/1800 은 사용자 요청으로 대상에서 제외 — 2000 step 결과만 필요)
#
#   ⚠️ 디스크 여유가 13GB뿐이라(이전 실패 원인 = No space left on device)
#      base 모델과 LoRA 머지 산출물을 전부 /dev/shm(tmpfs, RAM)에 둔다.
#      실제 디스크에는 결과 json/summary(수 MB)만 회수한다.
#
#   중단: kill $(cat $WS/.finish_charades_eval.pid)
# ============================================================
set -uo pipefail

# 스크립트 위치(Team4/eval/analysis) 기준으로 workspace 유도. 환경변수로 덮어쓸 수 있다.
WS=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export WS                      # 내장 python heredoc 에서 os.environ["WS"] 로 읽는다
RUN=sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling
TESTSET=charades_rlp_noaudio
CKPTS="2000"

SHM=/dev/shm/titok_eval
BASE_SHM=$SHM/salmonn2p_7b_unav_v8
BASE_LINK=$WS/checkpoints/base_unav_v8_shm
LOG=$WS/outputs/gdpo/$RUN/finish_eval.log

# 대기 대상: 현재 돌고 있는 GRPO 학습
WAIT_PID=${WAIT_PID:-7919}
WAIT_MATCH="gdpo_trainer_batch_GRPO.py"

exec >>"$LOG" 2>&1
echo ""
echo "============================================================"
echo "=== START $(date -Is)  pid=$$"
echo "============================================================"
echo $$ > $WS/.finish_charades_eval.pid

# ---------- 1) 학습 종료 대기 ----------
if kill -0 "$WAIT_PID" 2>/dev/null && tr '\0' ' ' < /proc/$WAIT_PID/cmdline | grep -q "$WAIT_MATCH"; then
    echo "[WAIT] GRPO 학습(pid $WAIT_PID) 종료 대기 시작 $(date -Is)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
    echo "[WAIT] 종료 확인 $(date -Is) — GPU 메모리 반환 대기 3분"
    sleep 180
else
    echo "[WAIT] pid $WAIT_PID 는 대상 학습이 아님/이미 종료 — 바로 진행"
fi

# ---------- 2) 자원 확인 ----------
shm_free_gb() { df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9'; }
disk_free_gb() { df -BG --output=avail / | tail -1 | tr -dc '0-9'; }
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

echo "[RES] shm=$(shm_free_gb)G disk=$(disk_free_gb)G gpu_free=$(gpu_free_mb)MiB"

if [ "$(shm_free_gb)" -lt 45 ]; then echo "[ABORT] /dev/shm 여유 부족(<45G)"; exit 1; fi
if [ "$(disk_free_gb)" -lt 5 ]; then echo "[ABORT] 디스크 여유 부족(<5G)"; exit 1; fi
if [ "$(gpu_free_mb)" -lt 60000 ]; then echo "[ABORT] GPU 여유 부족(<60GB) — 다른 작업이 점유 중"; exit 1; fi

# ---------- 3) base 모델(salmonn2p_7b_unav_v8)을 tmpfs 로 ----------
mkdir -p "$SHM"
if [ ! -f "$BASE_SHM/config.json" ]; then
    echo "[DL] gdrive:checkpoints/base/salmonn2p_7b_unav_v8 → $BASE_SHM (17.2GiB) $(date -Is)"
    "${RCLONE:-$HOME/rclone}" --config "${RCLONE_CONF:-$HOME/rclone.conf}" copy \
        gdrive:checkpoints/base/salmonn2p_7b_unav_v8 "$BASE_SHM" \
        --transfers 8 --checkers 8 --stats 60s || { echo "[ABORT] base 다운로드 실패"; exit 1; }
fi
[ -f "$BASE_SHM/config.json" ] || { echo "[ABORT] base 모델 불완전"; exit 1; }
ln -sfn "$BASE_SHM" "$BASE_LINK"
echo "[DL] base 준비 완료 $(date -Is)"

# ---------- 4) 체크포인트별 재개 ----------
for CK in $CKPTS; do
    REAL=$WS/outputs/gdpo/$RUN/checkpoint-$CK/fps5_tti/$TESTSET
    WORK=$SHM/work_$CK
    echo ""
    echo "---------- checkpoint-$CK $(date -Is) ----------"
    [ -d "$REAL" ] || { echo "[SKIP] $REAL 없음"; continue; }

    n_before=$(python3 -c "import json;print(len(json.load(open('$REAL/test_results_rank0.json'))))" 2>/dev/null || echo 0)
    if [ "$n_before" -ge 3720 ]; then echo "[SKIP] 이미 완료 (n=$n_before)"; continue; fi
    echo "[RUN] 재개 시작 n=$n_before/3720"

    rm -rf "$WORK"; mkdir -p "$WORK"
    cp "$REAL/test_results_rank0.json" "$WORK/" 2>/dev/null
    cp "$REAL/.chunk_idx" "$WORK/" 2>/dev/null

    # tmpfs 를 OUT_DIR 로 써서 LoRA 머지 산출물(~18GB)이 디스크에 안 닿게 한다
    bash "$WS/Team4/eval/eval.sh" CHUNK=on STAGE=gdpo \
        CKPT_MODEL_ID="$RUN" CKPT_STEP="$CK" \
        BASE_MODEL_ID=base_unav_v8_shm \
        TESTSET="$TESTSET" OUT_DIR="$WORK" GPUS=0 \
        || echo "[WARN] checkpoint-$CK eval.sh 비정상 종료 (부분 결과는 회수)"

    # 결과 회수 (모델 가중치는 두고 결과물만)
    for f in test_results_rank0.json pairwise_miou_summary.json union_miou_summary.json \
             sample_miou_summary.json table.txt eval_miou_progress.jsonl .chunk_idx; do
        [ -f "$WORK/$f" ] && cp "$WORK/$f" "$REAL/$f"
    done
    [ -f "$WORK/inference.log" ] && cp "$WORK/inference.log" "$REAL/inference_resume.log"

    n_after=$(python3 -c "import json;print(len(json.load(open('$REAL/test_results_rank0.json'))))" 2>/dev/null || echo 0)
    echo "[RUN] checkpoint-$CK 종료 n=$n_before → $n_after $(date -Is)"
    rm -rf "$WORK"
done

# ---------- 5) 정리 + 최종 집계 ----------
rm -rf "$SHM"
rm -f "$BASE_LINK"
echo ""
echo "=== 최종 집계 $(date -Is) ==="
python3 "$WS/score_charades_all.py" | tee "$WS/outputs/CHARADES_FINAL.txt"
echo "=== DONE $(date -Is) ==="
rm -f $WS/.finish_charades_eval.pid

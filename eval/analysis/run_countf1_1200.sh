#!/bin/bash
# ============================================================
# run_countf1_1200.sh
#   run_unav_eval_1200.sh (UnAV-100 titok 평가) 가 끝나기를 기다렸다가,
#   같은 결과로 count_f1.py 를 돌려 N_gt별 breakdown + CSV 를 만든다.
#
#   ※ CountF1 자체는 eval_miou.py 가 이미 3종 summary 의 count_metrics 에
#     넣어준다. 이 스크립트는 그 값을 재현 대조하고(기존 파이프라인 검증),
#     summary 에는 없는 N_gt별 세부 지표를 추가로 뽑는 용도.
#
#   중단: kill $(cat $WS/.run_countf1.pid)
# ============================================================
set -uo pipefail

# 스크립트 위치(Team4/eval/analysis) 기준으로 workspace 유도. 환경변수로 덮어쓸 수 있다.
WS=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export WS                      # 내장 python heredoc 에서 os.environ["WS"] 로 읽는다
RUN=sft_7b_unpucha_v8_rl_rMsep3_unpucha_batch4_noscaling_GRPO
CK=1200
REAL=$WS/outputs/gdpo/$RUN/checkpoint-$CK/fps5_tti/unav100_titok
PY=${PY:-$(command -v python3)}          # 원본: miniconda3/envs/salmonn2plus/bin/python3.10
WAIT_PID=${WAIT_PID:-234656}          # run_unav_eval_1200.sh
LOG=$WS/outputs/gdpo/$RUN/countf1_ck$CK.log

exec >>"$LOG" 2>&1
echo ""
echo "============================================================"
echo "=== START $(date -Is)  pid=$$"
echo "============================================================"
echo $$ > $WS/.run_countf1.pid

# ---------- 1) 평가 종료 대기 ----------
if kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "[WAIT] UnAV 평가(pid $WAIT_PID) 종료 대기 $(date -Is)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
    echo "[WAIT] 종료 확인 $(date -Is)"
fi

RES=$REAL/test_results_rank0.json
[ -f "$RES" ] || { echo "[ABORT] 결과 없음: $RES"; rm -f $WS/.run_countf1.pid; exit 1; }
n=$($PY -c "import json;print(len(json.load(open('$RES'))))" 2>/dev/null || echo 0)
echo "[RES] n=$n"
if [ "$n" -lt 3455 ]; then
    echo "[WARN] 3455 미만 — 부분 결과로 계산한다(비교 시 주의)"
fi

# ---------- 2) 결과 JSON -> count_f1.py 용 JSONL ----------
#   결과 항목의 ref(시간토큰 GT) 를 GT 로, pred 를 예측으로 쓴다.
#   (eval_miou.py 의 gt_source="ref(auto)" 와 동일한 GT 출처라 수치가 맞물린다)
WORK=$REAL/countf1
mkdir -p "$WORK"
$PY - "$RES" "$WORK" <<'PYEOF'
import json, sys, os
sys.path.insert(0, os.path.join(os.environ["WS"], "Team4", "eval"))
from eval_miou import parse_tokens

res_path, work = sys.argv[1], sys.argv[2]
MT = 999.9
rows = json.load(open(res_path))

with open(os.path.join(work, "gt.jsonl"), "w") as fg, \
     open(os.path.join(work, "pred.jsonl"), "w") as fp:
    for x in rows:
        vid = x["video"].split("/")[-1].rsplit(".", 1)[0]
        q = x.get("gt_label") or ""
        fg.write(json.dumps({"video_id": vid, "query": q,
                             "segments": parse_tokens(x.get("ref", ""), MT)},
                            ensure_ascii=False) + "\n")
        fp.write(json.dumps({"video_id": vid, "query": q,
                             "raw_output": x.get("pred", "")},
                            ensure_ascii=False) + "\n")
print(f"[JSONL] {len(rows)} 행 -> {work}/gt.jsonl, pred.jsonl")
PYEOF

# ---------- 3) count_f1.py ----------
echo ""
echo "=== count_f1.py (format=token) $(date -Is) ==="
$PY "$WS/Team4/eval/count_f1.py" \
    --gt "$WORK/gt.jsonl" \
    --pred "ck$CK=$WORK/pred.jsonl" \
    --format token \
    --csv "$WORK/countf1.csv"

# ---------- 4) eval_miou.py 내장 CountF1 과 대조 ----------
echo ""
echo "=== 기존 summary 의 count_metrics (대조용) ==="
$PY - "$REAL" <<'PYEOF'
import json, os, sys
real = sys.argv[1]
for f in ("sample_miou_summary.json", "pairwise_miou_summary.json",
          "union_miou_summary.json"):
    p = os.path.join(real, f)
    if not os.path.exists(p):
        print(f"  {f}: 없음")
        continue
    d = json.load(open(p))
    # count_metrics 는 세 파일 공통(compute_shared_block) — 값이 같아야 정상.
    # 현재 eval_miou.py 는 중첩 dict 로 낸다:
    #   {n_multi, n_single, CR_multi, chance_floor, CR_star, FMR, SingleAcc, CountF1}
    # 예전 요약본은 top-level 평면 키(CR_chance_b, CR_multi_n ...)라 둘 다 받는다.
    cm = d.get("count_metrics")
    if isinstance(cm, dict):
        keys = ("n_multi", "n_single", "CR_multi", "chance_floor",
                "CR_star", "FMR", "SingleAcc", "CountF1")
        src = cm
    else:
        keys = ("CR_multi_n", "CR_single_n", "CR_multi", "CR_chance_b",
                "CR_star", "FMR", "SingleAcc", "CountF1")
        src = d
    got = "  ".join(f"{k}={src[k]}" for k in keys if k in src)
    print(f"  {f}: {got if got else '(count_metrics 없음)'}")

print("\n  대조 기준: count_f1.py 의 USA == CR_star, OSA == SingleAcc,"
      " b == chance_floor, CountF1 == CountF1")
PYEOF

echo ""
echo "=== DONE $(date -Is) ==="
rm -f $WS/.run_countf1.pid

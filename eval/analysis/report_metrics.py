#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_metrics.py — 성능 보고 표준 지표를 한 번에 뽑는다.
  F1@0.1 / F1@0.3 / F1@0.5 / F1@0.7 / mIoU / CountF1

  mIoU·F1 은 sample_miou_summary.json 에서,
  CountF1 은 test_results_rank0.json 의 raw pred 를 다시 파싱해 계산한다
  (정의·파서는 Team4/eval/{eval_miou,count_f1}.py 와 동일 — recompute_countf1_unav.py 참고):
     USA = max(0,(CR-b)/(1-b)),  OSA = |{N_gt==1 & N_pred<=1}|/|{N_gt==1}|
     CountF1 = 2·USA·OSA/(USA+OSA)

사용:
  python3 report_metrics.py <결과디렉토리> [<결과디렉토리> ...]
    결과디렉토리 = sample_miou_summary.json + test_results_rank0.json 이 있는 곳
"""
import json
import os
import sys

EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Team4/eval (스크립트 위치 기준)
sys.path.insert(0, EV)
from eval_miou import extract_answer_scope, parse_tokens          # noqa: E402
from count_f1 import compute_count_f1                              # noqa: E402

MAX_TIME = 999.9
KEYS = ["0.1", "0.3", "0.5", "0.7"]


def metrics(d):
    """결과 디렉토리 하나 → 표준 지표 dict."""
    s = json.load(open(os.path.join(d, "sample_miou_summary.json")))
    rows = json.load(open(os.path.join(d, "test_results_rank0.json")))
    pairs = []
    n_fail = 0
    for x in rows:
        gt = parse_tokens(x.get("ref", ""), MAX_TIME)
        pred = parse_tokens(extract_answer_scope(x.get("pred", "") or ""), MAX_TIME)
        if not pred:
            n_fail += 1
        pairs.append((len(gt), len(pred)))
    c = compute_count_f1(pairs)
    return {
        "n": s["n_samples"],
        "mIoU": s["mIoU_%"],
        "F1": {k: s["F1"][k] for k in KEYS},
        "CountF1": (c["CountF1"] or 0) * 100,
        "USA": (c["USA"] or 0) * 100,
        "OSA": (c["OSA"] or 0) * 100,
        "parse_fail": n_fail,
    }


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return 1
    got = []
    for d in dirs:
        if not os.path.exists(os.path.join(d, "sample_miou_summary.json")):
            print(f"[SKIP] 요약 없음: {d}")
            continue
        got.append((d, metrics(d)))

    hdr = f"{'':<22}{'F1@0.1':>9}{'F1@0.3':>9}{'F1@0.5':>9}{'F1@0.7':>9}{'mIoU':>9}{'CountF1':>9}"
    print(hdr)
    print("-" * len(hdr))
    for d, m in got:
        name = os.path.basename(os.path.dirname(os.path.dirname(d)))   # checkpoint-N
        print(f"{name:<22}" + "".join(f"{m['F1'][k]:>9.2f}" for k in KEYS)
              + f"{m['mIoU']:>9.2f}{m['CountF1']:>9.2f}")
    print()
    for d, m in got:
        name = os.path.basename(os.path.dirname(os.path.dirname(d)))
        print(f"{name}: n={m['n']}  USA={m['USA']:.2f}  OSA={m['OSA']:.2f}  "
              f"파싱실패={m['parse_fail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

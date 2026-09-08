#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pv_metrics.py — 지표 계산 (파서만 갈아끼우고 지표 코드는 그대로 재사용).

eval_miou.py 의 compute_method_block / compute_count_metrics 를 그대로 부른다.
보고 규약(팀 표준):
  mIoU      = sample  merge
  F1@0.5/0.7 = pairwise merge
  USA = CR_star, OSA = 1 - FMR, CountF1 = 조화평균(USA, OSA)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_data import (BASE, EM, MODELS, MODELS_ALL, SHORT, TITOK, TITOK_EVAL,  # noqa: E402
                     load_model_samples, rule_parse)

METRIC_COLS = ["mIoU", "F1@0.5", "F1@0.7", "USA", "OSA", "CountF1"]


def compute_metrics(pairs):
    """pairs: [(gt, pred), ...] → 6개 지표 dict."""
    smp = EM.compute_method_block(pairs, "sample")
    pw = EM.compute_method_block(pairs, "pairwise")
    cnt = EM.compute_count_metrics(pairs)
    osa = None if cnt["FMR"] is None else 1.0 - cnt["FMR"]
    return {
        "mIoU": smp["mIoU_%"],
        "F1@0.5": pw["F1"]["0.5"],
        "F1@0.7": pw["F1"]["0.7"],
        "USA": cnt["CR_star"],
        "OSA": osa,
        "CountF1": cnt["CountF1"],
        "n": len(pairs),
        "parse_ok": sum(1 for _, p in pairs if p),
        "parse_fail": sum(1 for _, p in pairs if not p),
    }


def _cm(s, key):
    """count 지표는 요약 JSON 버전에 따라 최상위(baseline) 또는 count_metrics 중첩(TiTok)."""
    if key in s:
        return s[key]
    return (s.get("count_metrics") or {}).get(key)


def stored_metrics(model):
    """논문에 실린 기존 수치 (저장된 summary JSON에서 읽음)."""
    d = (os.path.dirname(TITOK_EVAL) if model == TITOK
         else os.path.join(BASE, model, "unav100_multiseg/eval"))
    s = json.load(open(os.path.join(d, "sample_miou_summary.json")))
    p = json.load(open(os.path.join(d, "pairwise_miou_summary.json")))
    return {
        "mIoU": s["mIoU_%"],
        "F1@0.5": p["F1"]["0.5"],
        "F1@0.7": p["F1"]["0.7"],
        "USA": _cm(s, "CR_star"),
        "OSA": 1.0 - _cm(s, "FMR"),
        "CountF1": _cm(s, "CountF1"),
        "n": s["n_samples"],
        "parse_ok": s["parse_ok"],
        "parse_fail": s["parse_fail"],
    }


def selftest():
    """룰기반 파서로 재계산한 값이 저장된 summary 와 일치하는지 — 지표 코드 회귀 게이트."""
    print("[selftest] 룰기반 파서 재계산 vs 저장된 summary")
    worst = 0.0
    for m in MODELS_ALL:
        ss = load_model_samples(m)
        got = compute_metrics([(s["gt"], rule_parse(s)) for s in ss])
        exp = stored_metrics(m)
        diffs = {k: abs(got[k] - exp[k]) for k in METRIC_COLS}
        worst = max(worst, max(diffs.values()))
        ok = "OK " if max(diffs.values()) < 2e-3 else "MISMATCH"
        bad = max(diffs, key=diffs.get)
        print(f"  {ok} {SHORT[m]:<16} maxΔ={max(diffs.values()):.6f} ({bad})  "
              f"parse_ok {got['parse_ok']}=={exp['parse_ok']}")
    print(f"[selftest] 전체 최대 절대오차 = {worst:.6f}")
    # Qwen 만 mIoU 에서 0.001 (저장된 summary 가 testset='tmp' 로 다른 시점에 생성됨).
    # 세그먼트 총수·parse_ok·F1@all 은 전부 일치하므로 파서 차이가 아니다.
    return worst < 2e-3


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)

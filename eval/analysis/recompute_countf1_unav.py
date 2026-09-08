#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
recompute_countf1_unav.py — UnAV-100 결과 JSON 에서 USA/OSA/CountF1 재계산.

기존 summary 의 count_metrics 를 믿지 않고 test_results_rank0.json 의
raw pred 를 직접 파싱해서 (N_gt, N_pred) 를 다시 세고 지표를 계산한다.
정의·파서는 Team4/eval/{eval_miou,count_f1}.py 와 동일:

  S_multi={N_gt>=2}, S_single={N_gt==1}   (N_gt==0 은 양쪽 제외)
  CR  = mean_{S_multi} min(N_gt,N_pred)/max(N_gt,N_pred)
  b   = mean_{S_multi} 1/N_gt
  USA = max(0,(CR-b)/(1-b))     == eval_miou 의 CR_star
  OSA = |{S_single : N_pred<=1}|/|S_single|  == 1-FMR == SingleAcc
  CountF1 = 2·USA·OSA/(USA+OSA)

merge=on 행은 예측의 겹치는/맞닿은 구간을 합친 뒤 센 참고값(파이프라인 기본은 off).
"""
import json, os, sys

EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Team4/eval (스크립트 위치 기준)
sys.path.insert(0, EV)
from eval_miou import extract_answer_scope, parse_natural, parse_tokens  # noqa
from count_f1 import _normalize_plain, merge_overlapping, compute_count_f1, compute_breakdown  # noqa

MAX_TIME = 999.9
WS = os.environ.get("WORKSPACE",
     os.path.dirname(os.path.dirname(os.path.dirname(
         os.path.dirname(os.path.abspath(__file__))))))  # workspace

RUNS = [
    # (표시명, 결과 json, pred 포맷, GT 출처)
    ("ChronusOmni(FT) unav100_chronus",
     f"{WS}/outputs/sft/ChronusOmni/unav100_chronus/eval/test_results_rank0.json",
     "plain", "embedded"),
    ("TiTok mIoU62.2 (rMsep3_unpucha_b4_noscaling @2000) unav100_titok",
     f"{WS}/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling/"
     f"checkpoint-2000/fps5_tti/unav100_titok/test_results_rank0.json",
     "token", "ref"),
]


def parse_pred(raw, fmt):
    scope = extract_answer_scope(raw or "")
    if fmt == "token":
        return parse_tokens(scope, MAX_TIME)
    return parse_natural(_normalize_plain(scope), MAX_TIME)


def load(path, fmt, gt_src):
    rows = json.load(open(path))
    out = []
    for x in rows:
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        out.append((gt, parse_pred(x.get("pred", ""), fmt)))
    return out


def stored_summary(res_path):
    d = os.path.dirname(res_path)
    for f in ("sample_miou_summary.json", "pairwise_miou_summary.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            s = json.load(open(p))
            return s.get("count_metrics"), s.get("mIoU_%")
    return None, None


def f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    print("=" * 100)
    print("UnAV-100 count 지표 재계산 — USA(=CR*) / OSA(=SingleAcc) / CountF1")
    print("=" * 100)
    results = []
    for name, path, fmt, gt_src in RUNS:
        if not os.path.exists(path):
            print(f"[SKIP] 결과 없음: {path}")
            continue
        pairs_raw = load(path, fmt, gt_src)
        n_fail = sum(1 for _, p in pairs_raw if not p)
        n = len(pairs_raw)

        pr = [(len(g), len(p)) for g, p in pairs_raw]
        pm = [(len(g), len(merge_overlapping(p))) for g, p in pairs_raw]
        m_raw, m_mrg = compute_count_f1(pr), compute_count_f1(pm)
        bd = compute_breakdown(pr)
        stored, miou = stored_summary(path)

        results.append((name, n, n_fail, fmt, m_raw, m_mrg, bd, stored, miou))

    hdr = f"{'run':<52}{'merge':>6}{'USA':>9}{'OSA':>9}{'CountF1':>9}{'CR':>9}{'b':>9}{'|S_m|':>7}{'|S_s|':>7}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, n, nf, fmt, mr, mm, bd, st, miou in results:
        short = name[:50]
        print(f"{short:<52}{'off':>6}{f(mr['USA']):>9}{f(mr['OSA']):>9}{f(mr['CountF1']):>9}"
              f"{f(mr['CR']):>9}{f(mr['b']):>9}{mr['n_multi']:>7}{mr['n_single']:>7}")
        print(f"{'':<52}{'on':>6}{f(mm['USA']):>9}{f(mm['OSA']):>9}{f(mm['CountF1']):>9}"
              f"{f(mm['CR']):>9}{f(mm['b']):>9}{mm['n_multi']:>7}{mm['n_single']:>7}")

    for name, n, nf, fmt, mr, mm, bd, st, miou in results:
        print("\n" + "=" * 100)
        print(f"{name}")
        print(f"  n={n}  parser={fmt}  파싱실패(N_pred=0)={nf} ({100.0*nf/n:.2f}%)  "
              f"sample_mIoU={miou}")
        print(f"  [재계산 merge=off]  USA={f(mr['USA'])}  OSA={f(mr['OSA'])}  "
              f"CountF1={f(mr['CountF1'])}  (CR={f(mr['CR'])}, b={f(mr['b'])}, FMR={f(1-mr['OSA']) if mr['OSA'] is not None else '-'})")
        if st:
            same = (abs(st.get('CR_star', -1) - mr['USA']) < 1e-5 and
                    abs(st.get('SingleAcc', -1) - mr['OSA']) < 1e-5 and
                    abs(st.get('CountF1', -1) - mr['CountF1']) < 1e-5)
            print(f"  [기존 summary]      USA={st.get('CR_star')}  OSA={st.get('SingleAcc')}  "
                  f"CountF1={st.get('CountF1')}   -> {'일치' if same else '★불일치★'}")
        print(f"  [참고 merge=on]     USA={f(mm['USA'])}  OSA={f(mm['OSA'])}  CountF1={f(mm['CountF1'])}")
        print(f"\n  N_gt 별 breakdown (merge=off)")
        print("  " + "  ".join(h.ljust(11) for h in
                               ["N_gt", "n", "mean_N_pred", "CR", "exact_acc", "count_MAE"]))
        print("  " + "-" * 72)
        for k in ["1", "2", "3", "4", ">=5"]:
            r = bd[k]
            print("  " + "  ".join(c.ljust(11) for c in
                                   [k, str(r["n"]), f(r["mean_n_pred"], 3), f(r["CR"]),
                                    f(r["exact_acc"]), f(r["count_mae"], 3)]))
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

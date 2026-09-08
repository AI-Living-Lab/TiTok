#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
breakdown_by_ngt.py — UnAV-100 추론 결과를 N_gt(정답 세그먼트 개수)별로 재집계.

버킷마다 다음을 낸다:
  n, sample mIoU, pairwise mIoU, F1@THR(pairwise; env THR, 기본 0.5), 평균 N_pred,
  Exact-count Accuracy(N_pred==N_gt 비율), USA, OSA, CountF1

계산은 전부 Team4/eval/eval_miou.py 의 함수를 그대로 호출한다(정의 일치 보장).
  - mIoU/F1  : compute_method_block(버킷 샘플, method)
  - USA/OSA  : compute_count_metrics(버킷 샘플)  → CR_star / SingleAcc

⚠ 정의상 버킷별로 못 내는 값이 있다:
  USA 는 S_multi={N_gt>=2} 에서만, OSA 는 S_single={N_gt==1} 에서만 정의된다.
  따라서 N_gt=1 행은 OSA 만, N_gt>=2 행은 USA 만 값이 있고, 둘의 조화평균인
  CountF1 은 한 버킷 안에서 정의되지 않는다('-'). 전체/멀티 합계 행에서만 낸다.
"""
import json, os, sys

EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Team4/eval (스크립트 위치 기준)
sys.path.insert(0, EV)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,   # noqa
                       compute_method_block, compute_count_metrics)
from count_f1 import _normalize_plain                                        # noqa

MAX_TIME = 999.9
WS = os.environ.get("WORKSPACE",
     os.path.dirname(os.path.dirname(os.path.dirname(
         os.path.dirname(os.path.abspath(__file__))))))  # workspace
THR = os.environ.get("THR", "0.5")   # F1/Recall 임계값 (env THR 로 교체)
# 버킷 구성표: 이름 -> (표시 라벨 순서, N_gt -> 라벨 매핑 함수)
SCHEMES = {
    "1/2/3+":      (["1", "2", "3+"],
                    lambda g: "1" if g == 1 else ("2" if g == 2 else "3+")),
    "1/2/3/4+":    (["1", "2", "3", "4+"],
                    lambda g: str(g) if g <= 3 else "4+"),
    "1/2/3/4/5+":  (["1", "2", "3", "4", "5+"],
                    lambda g: str(g) if g <= 4 else "5+"),
}

RUNS = [
    ("ChronusOmni (finetuned)",
     f"{WS}/outputs/sft/ChronusOmni/unav100_chronus/eval/test_results_rank0.json",
     "plain", "embedded"),
    ("TiTok mIoU62.2 (rMsep3_unpucha_b4_noscaling @ck2000)",
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
    """-> [(gt_segs, pred_segs), ...]"""
    out = []
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        out.append((gt, parse_pred(x.get("pred", ""), fmt)))
    return out


def agg(samples):
    """한 묶음의 샘플 -> 지표 dict."""
    if not samples:
        return None
    n = len(samples)
    smp = compute_method_block(samples, "sample")
    pw = compute_method_block(samples, "pairwise")
    cnt = compute_count_metrics(samples)
    npred = [len(p) for _, p in samples]
    exact = sum(1 for g, p in samples if len(g) == len(p)) / n
    mae = sum(abs(len(g) - len(p)) for g, p in samples) / n
    return {
        "n": n,
        "sample_mIoU": smp["mIoU_%"],
        "pairwise_mIoU": pw["mIoU_%"],
        "F1@thr": pw["F1"][THR],
        "sampleR@thr": smp["Recall"][THR],
        "mean_N_pred": sum(npred) / n,
        "exact_acc": exact,
        "count_MAE": mae,
        "USA": cnt["CR_star"],
        "OSA": cnt["SingleAcc"],
        "CountF1": cnt["CountF1"],
        "CR": cnt["CR_multi"],
        "b": cnt["chance_floor"],
        "parse_fail": sum(1 for _, p in samples if not p),
    }


def fnum(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


COLS = [("N_gt", 6), ("n", 6), ("mIoU", 8), ("pwIoU", 8), ("F1@" + THR, 8),
        ("N_pred", 8), ("Exact", 8), ("USA", 8), ("OSA", 8), ("CountF1", 9)]


def print_table(title, rows):
    print(f"\n{title}")
    hdr = "".join(h.rjust(w) for h, w in COLS)
    print(hdr)
    print("-" * len(hdr))
    for label, r in rows:
        if r is None:
            continue
        cells = [label, str(r["n"]), fnum(r["sample_mIoU"]), fnum(r["pairwise_mIoU"]),
                 fnum(r["F1@thr"]), fnum(r["mean_N_pred"], 3),
                 fnum(100 * r["exact_acc"]),
                 fnum(r["USA"], 4) if r["USA"] is not None else "-",
                 fnum(r["OSA"], 4) if r["OSA"] is not None else "-",
                 fnum(r["CountF1"], 4) if r["CountF1"] is not None else "-"]
        print("".join(c.rjust(w) for c, (_, w) in zip(cells, COLS)))


def main():
    loaded = []
    for name, path, fmt, gt_src in RUNS:
        if not os.path.exists(path):
            print(f"[SKIP] 없음: {path}")
            continue
        samples = load(path, fmt, gt_src)
        loaded.append((name, samples, fmt, gt_src))

    for scheme, (labels, fn) in SCHEMES.items():
        print("#" * 79)
        print(f"# 버킷 구성: N_gt = {scheme}")
        print("#" * 79)
        per_model = {}
        for name, samples, fmt, gt_src in loaded:
            by = {k: [] for k in labels}
            for g, p in samples:
                if len(g) > 0:
                    by[fn(len(g))].append((g, p))
            rows = [(k, agg(by[k])) for k in labels]
            rows.append(("N>=2", agg([s2 for k in labels[1:] for s2 in by[k]])))
            rows.append(("ALL", agg([s2 for k in labels for s2 in by[k]])))
            per_model[name] = rows
            print("=" * 71)
            print(f"{name}   (n={len(samples)}, parser={fmt}, GT={gt_src})")
            print_table(f"N_gt별 집계 [{scheme}]", rows)

        if len(per_model) == 2:
            (na, ra), (nb, rb) = list(per_model.items())
            da, db = dict(ra), dict(rb)
            print("\n" + "=" * 71)
            print(f"차이 (TiTok - ChronusOmni)  [{scheme}]")
            hdr = "".join(h.rjust(w) for h, w in COLS)
            print(hdr)
            print("-" * len(hdr))
            for k in labels + ["N>=2", "ALL"]:
                A, B = da.get(k), db.get(k)
                if not A or not B:
                    continue

                def d(f, nd=2, scale=1.0, A=A, B=B):
                    if A[f] is None or B[f] is None:
                        return "-"
                    return f"{scale * (B[f] - A[f]):+.{nd}f}"

                cells = [k, str(B["n"]), d("sample_mIoU"), d("pairwise_mIoU"),
                         d("F1@thr"), d("mean_N_pred", 3), d("exact_acc", 2, 100.0),
                         d("USA", 4), d("OSA", 4), d("CountF1", 4)]
                print("".join(c.rjust(w) for c, (_, w) in zip(cells, COLS)))
        print()

    print("=" * 71)
    print("주) mIoU=sample(All_IoU, 샘플단위), pwIoU=pairwise(GT세그먼트 best-match),")
    print("    F1@%s = pairwise 기준" % THR + "(=table.txt 의 F1 열). Exact = N_pred==N_gt 비율(%).")
    print("    USA 는 N_gt>=2 에서만, OSA 는 N_gt==1 에서만 정의 -> 버킷별 CountF1 은 없음.")
    print("    버킷을 어떻게 묶어도 N_gt=1 행과 N>=2/ALL 행은 동일(합치는 건 다중 쪽뿐).")


if __name__ == "__main__":
    main()

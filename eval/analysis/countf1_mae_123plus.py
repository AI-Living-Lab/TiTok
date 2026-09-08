#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
countf1_mae_123plus.py — N_gt = 1/2/3+ 버킷별 count MAE 및 CountF1.

count MAE = mean |N_gt - N_pred|  (버킷 안에서 그냥 정의됨. 규약 불필요)

CountF1 은 원 정의상 버킷별로 못 낸다:
  USA 는 S_multi={N_gt>=2}, OSA 는 S_single={N_gt==1} 에서만 정의 → 한 버킷엔 한쪽뿐.
그래서 두 규약으로 채운다(둘 다 표기하고, 원 정의 값과의 관계를 남긴다).

  [A] 분해형 (decomposition)
      CountF1_b = HM(USA_b, OSA_global)   (N_gt>=2 버킷)
      CountF1_1 = HM(USA_global, OSA_1)   (N_gt=1 버킷; OSA_1 == OSA_global)
      → 한쪽 half 만 버킷 값으로 갈아끼운다. N>=2 합계/ALL 에서 원래 CountF1 로 정확히 복원.
        "전체 CountF1 을 어느 멀티 버킷이 끌고 가는가" 를 본다.

  [B] 버킷-국소형 (bucket-local)
      OSA 를 P(N_pred <= N_gt) 로 일반화 (N_gt=1 에선 원 OSA 와 정확히 같음).
      USA_b = max(0,(CR_b - b_b)/(1 - b_b)),  b_b = mean 1/N_gt  (버킷 내부에서 계산)
      CountF1_b = HM(USA_b, OSA_b).
      N_gt=1 버킷은 b_b=1 이라 USA 에 여유분이 없다(과소분할이 불가능) → 편면(one-sided),
      CountF1_1 := OSA_1 로 두고 '편면' 표시.
"""
import json, os, sys

EV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Team4/eval (스크립트 위치 기준)
sys.path.insert(0, EV)
from eval_miou import (extract_answer_scope, parse_natural, parse_tokens,  # noqa
                       compute_count_metrics, _clip01)
from count_f1 import _normalize_plain                                       # noqa

MAX_TIME = 999.9
WS = os.environ.get("WORKSPACE",
     os.path.dirname(os.path.dirname(os.path.dirname(
         os.path.dirname(os.path.abspath(__file__))))))  # workspace
LABELS = ["1", "2", "3+"]
BK = lambda g: "1" if g == 1 else ("2" if g == 2 else "3+")

RUNS = [
    ("ChronusOmni (finetuned)",
     f"{WS}/outputs/sft/ChronusOmni/unav100_chronus/eval/test_results_rank0.json",
     "plain", "embedded"),
    ("TiTok mIoU62.2 (@ck2000)",
     f"{WS}/outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling/"
     f"checkpoint-2000/fps5_tti/unav100_titok/test_results_rank0.json",
     "token", "ref"),
]


def parse_pred(raw, fmt):
    sc = extract_answer_scope(raw or "")
    return parse_tokens(sc, MAX_TIME) if fmt == "token" else parse_natural(_normalize_plain(sc), MAX_TIME)


def load(path, fmt, gt_src):
    out = []
    for x in json.load(open(path)):
        gt = ([list(s) for s in x.get("gt_segments") or []] if gt_src == "embedded"
              else parse_tokens(x.get("ref", ""), MAX_TIME))
        out.append((len(gt), len(parse_pred(x.get("pred", ""), fmt))))
    return [(g, p) for g, p in out if g > 0]


def hm(a, b):
    if a is None or b is None:
        return None
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def bucket_stats(pairs):
    n = len(pairs)
    if n == 0:
        return None
    mae = sum(abs(g - p) for g, p in pairs) / n
    exact = sum(1 for g, p in pairs if g == p) / n
    mean_np = sum(p for _, p in pairs) / n
    under = sum(1 for g, p in pairs if p < g) / n
    over = sum(1 for g, p in pairs if p > g) / n
    cr = sum((min(g, p) / max(g, p)) if max(g, p) > 0 else 0.0 for g, p in pairs) / n
    b = sum(1.0 / g for g, _ in pairs) / n
    usa_local = _clip01((cr - b) / (1 - b)) if (1 - b) > 1e-12 else None
    osa_gen = sum(1 for g, p in pairs if p <= g) / n     # P(N_pred <= N_gt)
    return dict(n=n, mae=mae, exact=exact, mean_np=mean_np, under=under, over=over,
                cr=cr, b=b, usa_local=usa_local, osa_gen=osa_gen)


def f(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


for name, path, fmt, gt_src in RUNS:
    pairs = load(path, fmt, gt_src)
    glob = compute_count_metrics([([None] * g, [None] * p) for g, p in pairs])
    USA_G, OSA_G, CF1_G = glob["CR_star"], glob["SingleAcc"], glob["CountF1"]

    by = {k: [] for k in LABELS}
    for g, p in pairs:
        by[BK(g)].append((g, p))
    groups = [(k, by[k]) for k in LABELS]
    groups.append(("N>=2", by["2"] + by["3+"]))
    groups.append(("ALL", pairs))

    print("=" * 104)
    print(f"{name}   n={len(pairs)}   [전역] USA={USA_G}  OSA={OSA_G}  CountF1={CF1_G}")
    print("=" * 104)
    hdr = (f"{'N_gt':>5}{'n':>6}{'MAE':>8}{'Exact':>8}{'N_pred':>8}{'under%':>8}{'over%':>8}"
           f"{'CR':>8}{'b':>8}{'USA_b':>9}{'OSA_b':>9}{'CF1[A]':>9}{'CF1[B]':>9}")
    print(hdr)
    print("-" * len(hdr))
    for k, pr in groups:
        s = bucket_stats(pr)
        if k == "1":
            usa_b = None                       # b=1 → 여유분 없음(과소분할 불가)
            cf1A = hm(USA_G, s["osa_gen"])     # OSA_1 == OSA_global → 전역 CountF1 복원
            cf1B = s["osa_gen"]                # 편면
            mark = "*"
        elif k == "ALL":
            usa_b = USA_G
            cf1A = CF1_G
            cf1B = CF1_G
            mark = ""
        else:
            usa_b = s["usa_local"]
            cf1A = hm(usa_b, OSA_G)
            cf1B = hm(usa_b, s["osa_gen"])
            mark = ""
        print(f"{k+mark:>5}{s['n']:>6}{s['mae']:>8.3f}{100*s['exact']:>8.2f}{s['mean_np']:>8.3f}"
              f"{100*s['under']:>8.2f}{100*s['over']:>8.2f}{s['cr']:>8.4f}{s['b']:>8.4f}"
              f"{f(usa_b):>9}{f(s['osa_gen']):>9}{f(cf1A):>9}{f(cf1B):>9}")
    print("  * N_gt=1 버킷: b=1 이라 USA 정의 안 됨(과소분할 불가) → CF1[B] 는 편면(=OSA).")
    print("    CF1[A]=HM(USA_b, OSA_전역), CF1[B]=HM(USA_b, OSA_b),  OSA_b = P(N_pred<=N_gt).")
    print()

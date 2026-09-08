#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
osa_charades.py — OSA (over-segmentation avoidance) 계산.

정의
  1) 샘플마다 파싱된 구간 개수 N_pred (겹침 병합 없이 원본 개수)
  2) OSA = |{ i : N_pred_i <= 1 }| / N_total
파서
  UnAV-100 평가에 쓴 것과 동일 — Team4/eval/eval_miou.py 의 parse_pred(natural=False)
  = extract_answer_scope() 후 parse_tokens(), 구분자 to|-|–|—|~ 허용, 병합 없음.
  (UnAV-100·Charades 양쪽 summary 모두 natural=False 로 채점됨을 확인)
"""
import json, os, sys, importlib.util, collections

EVAL_MIOU = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_miou.py")
MAX_TIME = 999.9   # eval.sh 가 config.yaml 의 MAX_TIME 으로 넘기는 값

_spec = importlib.util.spec_from_file_location("eval_miou", EVAL_MIOU)
em = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(em)


def osa(path, pred_key="pred"):
    data = json.load(open(path))
    counts = collections.Counter()
    for x in data:
        segs = em.parse_pred(x.get(pred_key, ""), False, MAX_TIME)  # ← 병합 없는 원본 구간 리스트
        counts[len(segs)] += 1
    total = len(data)
    le1 = sum(c for n, c in counts.items() if n <= 1)
    return {
        "score": round(le1 / total, 4),
        "total": total,
        "n0": counts[0], "n1": counts[1], "n2": counts[2],
        "n3plus": sum(c for n, c in counts.items() if n >= 3),
        "parse_fail": counts[0],
        "parse_fail_ratio": round(counts[0] / total, 4),
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = osa(p)
        print(f"\n파일: {p}")
        print(f"  OSA score          : {r['score']:.4f}")
        print(f"  전체 샘플 수        : {r['total']}")
        print(f"  N_pred 분포        : 0개={r['n0']}  1개={r['n1']}  2개={r['n2']}  3개이상={r['n3plus']}")
        print(f"  파싱 실패(N_pred=0) : {r['parse_fail']} ({r['parse_fail_ratio']*100:.2f}%)")

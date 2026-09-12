#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maketable.py — 폴더 경로를 주면 그 하위의 모든 평가 summary 를 탐색해
<경로>/table.txt 에 단일 표(탭 구분)를 생성한다. 재실행 시 최신 내용으로 덮어쓴다.

결과 폴더(=pairwise_miou_summary.json 이 있는 폴더)당 1행. n_samples<500 은 제외.
행은 sample_mIoU 내림차순 정렬.

  ⚠ checkpoint-N 바로 아래에 eval 설정 폴더(예: fps5_tti / fps5_ttifix / fps5_off)가
    끼어 있는 경우가 있다. 같은 (ID, ckpt, testset) 이라도 이 폴더가 다르면 서로 다른
    추론 경로/설정으로 채점된 별개의 결과이므로 반드시 별도 행으로 구분해야 한다
    (예: fps5_tti vs fps5_ttifix 는 같은 체크포인트인데도 예측 자체가 다름 — 실측
    charades_rlp_noaudio ckpt-1400 에서 3720개 중 3690개 pred 문자열이 다르고
    mIoU 47.29%→50.26%로 갈림). 그래서 이 폴더명을 'cfg' 열로 따로 뽑아 ID/ckpt/testset
    이 같아도 cfg 가 다르면 표에서 구분되게 한다.

열(탭 구분, 22개):
  ID  ckpt  cfg  sample_mIoU
  F1@0.1  F1@0.3  F1@0.5  F1@0.7  F1@0.9        # threshold 별 F1 (각 열)
  CR  FMR  CountF1
  SCR
  R@0.1  R@0.3  R@0.5  R@0.7  R@0.9            # threshold 별 Recall (각 열)
  gt(mean)  pred(mean)  n_samples  testset

  - sample_mIoU : sample_miou_summary.json 의 mIoU_% (샘플단위 All_IoU 평균)
  - F1@θ / R@θ  : pairwise_miou_summary.json 기준 (best-match 세그먼트 단위)
  - CR      : CR* = chance 보정 후 멀티(N_gt>=2) 개수일치도. (CR_multi-b)/(1-b), [0,1] 클립.
              0=찍기 수준, 1=완벽. 높을수록 좋음. (count_metrics.CR_star)
  - FMR     : 싱글(N_gt==1)을 2개 이상으로 쪼갠 비율. 낮을수록 좋음. (count_metrics.FMR)
  - CountF1 : 조화평균(CR*, 1-FMR). 두 항 중 하나라도 0이면 0 → 반복포착+단일비분할
              둘 다 잘해야 높음. (count_metrics.CountF1)
      ※ CR/FMR/CountF1 은 count_metrics 가 있는(=최신 eval_miou.py 로 채점한) summary 만 값,
         구버전 summary 는 '-'. 채우려면 eval_miou.py 재실행.
  - SCR (Segment Count Ratio) : min(N_pred, N_gt) / max(N_pred, N_gt).
      N_* = 샘플당 평균 세그먼트 수(pred/gt_segments.mean_per_sample). 1.0 에 가까울수록
      예측 세그먼트 개수 규모가 GT 와 일치. (개수 비율일 뿐, 위치 정확도와 무관)
  - gt/pred(mean): gt_segments.mean_per_sample, pred_segments.mean_per_sample (각 열)
  - USA/OSA/CountF1(uv)/b(uv)/parsefail(uv) : eval_countf1_unav100.py (UnAV-100 전용,
      경계값 없이 세그먼트 "개수"만 쓰는 CountF1) 의 결과. 같은 폴더에
      countf1_unav100_summary.json 이 있을 때만 값이 채워지고, 없으면 '-'.
      USA≈CR, OSA≈1-FMR, CountF1(uv)≈CountF1 과 같은 개념이지만 계산 스크립트가 달라
      (--unparseable/--merge-overlap 처리 등) 값이 정확히 같지 않을 수 있어 별도 열로 둔다.

사용:  python3 maketable.py /home/team404/workspace/outputs/gdpo
"""
import argparse
import json
import os
import re

THS = ["0.1", "0.3", "0.5", "0.7", "0.9"]

HEADER = (["ID", "ckpt", "cfg", "sample_mIoU"]
          + [f"F1@{t}" for t in THS]
          + ["CR", "FMR", "CountF1", "SCR"]
          + [f"R@{t}" for t in THS]
          + ["gt(mean)", "pred(mean)", "n_samples", "testset"]
          + ["USA(uv)", "OSA(uv)", "CountF1(uv)", "b(uv)", "parsefail(uv)"])


def _num(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "-"


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _series_cols(d, keys=THS):
    """threshold 별 값을 열 리스트로 (슬래시 결합 X)."""
    if not isinstance(d, dict):
        return ["-" for _ in keys]
    return [_num(d.get(k)) for k in keys]


def _scr(gmean, pmean):
    """Segment Count Ratio = min/max. 둘 다 양수일 때만."""
    if not (_isnum(gmean) and _isnum(pmean)):
        return "-"
    g, p = float(gmean), float(pmean)
    hi = max(g, p)
    if hi <= 0:
        return "-"
    return f"{min(g, p) / hi:.2f}"


def parse_path(dirpath):
    """결과 폴더 절대경로 → (ID, ckpt, cfg, testset). root 위치와 무관하게 동작
    (leaf 폴더를 직접 가리켜도 OK). checkpoint-* 앞 컴포넌트를 ID 로 본다.

    cfg = checkpoint-N 과 testset(leaf) 사이에 낀 폴더(들) (예: fps5_tti, fps5_ttifix).
    같은 checkpoint 를 서로 다른 eval 설정/추론 경로로 두 번 채점한 경우 여기서 갈리므로,
    비어 있지 않으면 반드시 ID/ckpt/testset 과 별개로 표시해 행이 뭉개지지 않게 한다."""
    parts = [p for p in os.path.abspath(dirpath).split(os.sep) if p]
    testset = parts[-1] if parts else "-"
    ckpt, ck_idx = "-", None
    for i, p in enumerate(parts):
        m = re.fullmatch(r"checkpoint-(\w+)", p)
        if m:
            ckpt, ck_idx = m.group(1), i
            break
    if ck_idx is not None and ck_idx > 0:
        ID = parts[ck_idx - 1]                      # checkpoint-* 바로 앞 = run 이름
    else:
        ID = "-"
        for stage in ("gdpo", "base", "sft", "merged"):   # checkpoint 없는 경우
            if stage in parts and parts.index(stage) + 1 < len(parts):
                ID = parts[parts.index(stage) + 1]
                break

    cfg = "-"
    if ck_idx is not None and len(parts) - 1 > ck_idx + 1:
        cfg = "/".join(parts[ck_idx + 1: -1])       # checkpoint-* 와 testset(leaf) 사이
    return ID, ckpt, cfg, testset


def _load(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(description="폴더 하위 평가 summary → table.txt (단일 표, 탭 구분)")
    ap.add_argument("path", help="탐색 루트 (예: outputs/gdpo). 이 안에 table.txt 생성")
    ap.add_argument("--out", default="table.txt", help="출력 파일명 (기본 table.txt)")
    args = ap.parse_args()
    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        raise SystemExit(f"[에러] 폴더가 아님: {root}")

    rows = []
    for dirpath, _d, files in os.walk(root):
        if "pairwise_miou_summary.json" not in files:
            continue
        p = _load(os.path.join(dirpath, "pairwise_miou_summary.json"))
        s = _load(os.path.join(dirpath, "sample_miou_summary.json"))
        if (p.get("n_samples") or 0) < 500:
            continue
        ID, ckpt, cfg, testset = parse_path(dirpath)
        sample = _num(s.get("mIoU_%"))

        gmean = _num(p.get("gt_segments", {}).get("mean_per_sample"))
        pmean = _num(p.get("pred_segments", {}).get("mean_per_sample"))
        scr = _scr(gmean, pmean)

        cm = p.get("count_metrics") or {}
        cr = _num(cm.get("CR_star"))        # None(서브셋 없음/구버전) → '-'
        fmr = _num(cm.get("FMR"))
        countf1 = _num(cm.get("CountF1"))

        # eval_countf1_unav100.py 가 같은 폴더에 저장한 summary(있을 때만) → 추가 열.
        uv = _load(os.path.join(dirpath, "countf1_unav100_summary.json"))
        usa_uv = _num(uv.get("USA"))
        osa_uv = _num(uv.get("OSA"))
        countf1_uv = _num(uv.get("CountF1"))
        b_uv = _num(uv.get("b"))
        parsefail_uv = _num(uv.get("parse_fail_rate"))

        rows.append([ID, ckpt, cfg, sample]
                    + _series_cols(p.get("F1"))
                    + [cr, fmr, countf1, scr]
                    + _series_cols(p.get("Recall"))
                    + [gmean, pmean, str(p.get("n_samples", "-")), testset]
                    + [usa_uv, osa_uv, countf1_uv, b_uv, parsefail_uv])

    # sample_mIoU (열 인덱스 3) 내림차순
    rows.sort(key=lambda r: (float(r[3]) if _isnum(r[3]) else -1.0), reverse=True)

    lines = ["\t".join(HEADER)] + ["\t".join(r) for r in rows]
    out_path = os.path.join(root, args.out)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)
    print(f"[SAVED] {out_path}  ({len(rows)} 행, {len(HEADER)} 열)")


if __name__ == "__main__":
    main()

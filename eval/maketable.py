#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maketable.py — 폴더 경로를 주면 그 하위의 모든 평가 summary 를 탐색해
<경로>/table.txt 에 단일 표(탭 구분)를 생성한다. 재실행 시 최신 내용으로 덮어쓴다.

결과 폴더(=pairwise_miou_summary.json 이 있는 폴더)당 1행. n_samples<500 은 제외.
행은 sample_mIoU 내림차순 정렬.

열(기본, 탭 구분, 12개):
  ID  ckpt  sample_mIoU  F1@0.1  F1@0.3  F1@0.5  F1@0.7  CountF1  USA  OSA
  n_samples  testset

  - sample_mIoU : sample_miou_summary.json 의 mIoU_% (샘플단위 All_IoU 평균).
                  F1 열은 pairwise 기준이라 지표 계열이 섞인다. 열 이름에 'sample' 을
                  남겨 둔 이유가 이것이니 그냥 'mIoU' 로 줄여 읽지 말 것.
  - F1@θ    : pairwise_miou_summary.json 기준 (best-match 세그먼트 단위)
  - CountF1 : 조화평균(USA, OSA). 둘 중 하나라도 0 이면 0 → 반복포착 + 단일비분할을
              둘 다 잘해야 높음.
  - USA     : = CR* = chance 보정 후 멀티(N_gt>=2) 개수일치도. (CR_multi-b)/(1-b), [0,1] 클립.
              0=찍기 수준, 1=완벽. under-segmentation 해소력.
  - OSA     : = SingleAcc = 싱글(N_gt==1) 을 쪼개지 않은 비율(=1-FMR). 높을수록 좋음.
      ※ 파싱 실패(pred 0개) 가 많은 모델은 N_pred<=1 로 잡혀 OSA 가 부풀 수 있다.
        비교 전 summary 의 parse_fail 을 같이 확인할 것.
      ※ 구버전 summary(count_metrics 블록 없음) 는 최상위 CR_star/FMR 에서 읽고,
        SingleAcc 가 없으면 1-FMR 로 환산한다. 셋 다 없으면 '-'.

  --full : 기존 21열(F1@0.9, CR/FMR, SCR, R@θ, gt/pred(mean) 포함) 로 출력.

사용:  python3 maketable.py /home/team404/workspace/outputs/gdpo
"""
import argparse
import json
import os
import re

THS = ["0.1", "0.3", "0.5", "0.7", "0.9"]
MAIN_THS = ["0.1", "0.3", "0.5", "0.7"]          # 기본 표에 싣는 threshold

HEADER = (["ID", "ckpt", "sample_mIoU"]
          + [f"F1@{t}" for t in MAIN_THS]
          + ["CountF1", "USA", "OSA", "n_samples", "testset"])

HEADER_FULL = (["ID", "ckpt", "sample_mIoU"]
               + [f"F1@{t}" for t in THS]
               + ["CountF1", "USA", "OSA", "SCR"]
               + [f"R@{t}" for t in THS]
               + ["gt(mean)", "pred(mean)", "n_samples", "testset"])


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


def _count_metrics(p):
    """USA/OSA/CountF1 추출. 최신(count_metrics 블록) + 구버전(flat) 스키마 모두 지원.

    USA = CR_star, OSA = SingleAcc. 구버전 summary 에는 SingleAcc 가 없어 1-FMR 로
    환산한다(정의상 FMR = 1 - SingleAcc).
    """
    cm = p.get("count_metrics")
    if not isinstance(cm, dict):
        cm = p                                   # 구버전: 최상위에 CR_star/FMR/CountF1
    osa = cm.get("SingleAcc")
    if osa is None and _isnum(cm.get("FMR")):
        osa = 1.0 - float(cm["FMR"])
    return _num(cm.get("CountF1")), _num(cm.get("CR_star")), _num(osa)


def parse_path(dirpath):
    """결과 폴더 절대경로 → (ID, ckpt, testset). root 위치와 무관하게 동작
    (leaf 폴더를 직접 가리켜도 OK). checkpoint-* 앞 컴포넌트를 ID 로 본다."""
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
    return ID, ckpt, testset


def _load(path):
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    ap = argparse.ArgumentParser(description="폴더 하위 평가 summary → table.txt (단일 표, 탭 구분)")
    ap.add_argument("path", help="탐색 루트 (예: outputs/gdpo). 이 안에 table.txt 생성")
    ap.add_argument("--out", default="table.txt", help="출력 파일명 (기본 table.txt)")
    ap.add_argument("--full", action="store_true",
                    help="기존 전체 열(F1@0.9, SCR, R@θ, gt/pred(mean) 포함) 로 출력")
    args = ap.parse_args()
    header = HEADER_FULL if args.full else HEADER
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
        ID, ckpt, testset = parse_path(dirpath)
        sample = _num(s.get("mIoU_%"))

        gmean = _num(p.get("gt_segments", {}).get("mean_per_sample"))
        pmean = _num(p.get("pred_segments", {}).get("mean_per_sample"))
        scr = _scr(gmean, pmean)

        countf1, usa, osa = _count_metrics(p)
        n = str(p.get("n_samples", "-"))

        if args.full:
            rows.append([ID, ckpt, sample]
                        + _series_cols(p.get("F1"))
                        + [countf1, usa, osa, scr]
                        + _series_cols(p.get("Recall"))
                        + [gmean, pmean, n, testset])
        else:
            rows.append([ID, ckpt, sample]
                        + _series_cols(p.get("F1"), MAIN_THS)
                        + [countf1, usa, osa, n, testset])

    # sample_mIoU (열 인덱스 2) 내림차순
    rows.sort(key=lambda r: (float(r[2]) if _isnum(r[2]) else -1.0), reverse=True)

    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    out_path = os.path.join(root, args.out)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, out_path)
    print(f"[SAVED] {out_path}  ({len(rows)} 행, {len(header)} 열)")


if __name__ == "__main__":
    main()

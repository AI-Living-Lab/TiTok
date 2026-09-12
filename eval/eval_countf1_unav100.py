#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_countf1_unav100.py — UnAV-100 멀티세그먼트 grounding, "개수(count)"만 보는 CountF1 평가.

경계값(초 단위 시작/끝 시간)은 채점에 전혀 쓰지 않는다. GT/예측 세그먼트 "개수"만 비교한다.
temporal IoU 기반 평가가 필요하면 eval_miou.py / eval_miou_multiseg.py 를 사용할 것.
(eval_miou.py 의 compute_count_metrics() 가 사실상 동일한 지표를 CR_star/SingleAcc/CountF1
 이름으로 이미 계산하고 있음 — 이 스크립트는 UnAV-100 전용으로 파싱/브레이크다운/자체검증을
 더 얹은 독립 실행형 버전.)

[정의] (N_gt, N_pred = 샘플당 GT/예측 세그먼트 "개수")
  S_multi  = {i : N_gt(i) >= 2}
  S_single = {i : N_gt(i) == 1}
  (N_gt(i) == 0 인 샘플은 GT 자체가 비정상이므로 양쪽 서브셋에서 제외하고 카운트만 남김)

  CR  = mean_{i in S_multi} min(N_gt,N_pred) / max(N_gt,N_pred)
  b   = mean_{i in S_multi} 1/N_gt            # "항상 1개만 예측"하는 trivial model 의 기대 CR.
                                               # GT 분포에만 의존(예측과 무관).
  USA = max(0, (CR - b) / (1 - b))            # chance-보정 개수일치도(Under/multi 측)
  OSA = |{i in S_single : N_pred <= 1}| / |S_single|   # 싱글을 안 쪼갠 비율(Over 측)
  CountF1 = 2*USA*OSA / (USA+OSA),  USA==OSA==0 이면 CountF1=0

사용 예:
  # 모델별 비교 테이블 + CSV
  python3 eval_countf1_unav100.py --gt unav100_test.jsonl \\
      --pred base=preds_base.jsonl --pred gdpo=preds_gdpo.jsonl \\
      --format auto --out-csv countf1_compare.csv

  # maketable.py 가 읽을 수 있는 summary json 도 같이 저장
  python3 eval_countf1_unav100.py --gt unav100_test.jsonl \\
      --pred gdpo=preds_gdpo.jsonl \\
      --summary-out gdpo=/path/to/outputs/gdpo/run1/checkpoint-500/unav100/countf1_unav100_summary.json

  # b 재현 확인 (GT 분포만 사용, --pred 불필요). UnAV-100 test split 이면 ≈0.397 이 나와야 함.
  python3 eval_countf1_unav100.py --gt unav100_test.jsonl --check-b

  # 내장 자체 테스트(더미 1개-고정 모델, GT 그대로 예측한 모델, b 재현)
  python3 eval_countf1_unav100.py --gt unav100_test.jsonl --selftest
"""
import argparse
import csv
import json
import os
import re
import sys

# ============================== pred 파싱 ==============================
# --- token 포맷: "From <t0><t0><t2><t4><tdot><t5> to <t0><t0><t2><t9><tdot><t4>." ---
# 스펙상 정수부는 "3자리"라고 되어 있으나, 실제 예시("<t0><t0><t2><t4>")는 4개 digit-token이고
# 이 저장소의 다른 파서(eval_miou.py `_TOKTIME`)도 digit-token 개수를 고정하지 않는다.
# 안전하게 정수부는 1개 이상의 digit-token, 소수부는 1개 이상(그 중 첫 자리만 사용)으로 파싱한다.
_TOK_NUM = r"(?:<t\d>)+(?:<tdot>(?:<t\d>)+)?"
_TOK_SEG_RE = re.compile(rf"({_TOK_NUM})\s*(?:to|-|–|—|~)\s*({_TOK_NUM})", re.IGNORECASE)
_TOK_DIGIT_RE = re.compile(r"<t(\d)>")


def _decode_tok(num_str):
    if "<tdot>" in num_str:
        ip, _, dp = num_str.partition("<tdot>")
    else:
        ip, dp = num_str, ""
    ip_digits = _TOK_DIGIT_RE.findall(ip)
    dp_digits = _TOK_DIGIT_RE.findall(dp)
    if not ip_digits:
        return None
    val = float(int("".join(ip_digits)))
    if dp_digits:
        val += int(dp_digits[0]) / 10.0
    return val


def parse_token_format(text):
    """"<t..> to <t..>" 패턴을 모두 찾아 [(s,e), ...] 로 반환. 실패 시 빈 리스트."""
    segs = []
    for a, b in _TOK_SEG_RE.findall(text or ""):
        s, e = _decode_tok(a), _decode_tok(b)
        if s is None or e is None:
            continue
        segs.append((s, e))
    return segs


# --- plain 포맷: "second{X.X}-second{X.X}", "12.3s to 45.6s" 등 baseline 자연어 출력 ---
def parse_plain_format(text):
    """구체적 → 일반적 순서로 매칭하고, 이미 매칭된 문자 구간은 consumed 로 막아 중복 방지."""
    text = text or ""
    consumed = [False] * len(text)
    segs = []

    def grab(pattern, conv):
        for m in re.finditer(pattern, text, re.IGNORECASE):
            a, b = m.span()
            if any(consumed[a:b]):
                continue
            r = conv(m)
            if r is not None:
                segs.append(r)
                for i in range(a, b):
                    consumed[i] = True

    # second{X.X}-second{Y.Y} / second{X.X} to second{Y.Y}
    grab(r"second\{?\s*([\d.]+)\s*\}?\s*(?:to|-|–|—|~)\s*second\{?\s*([\d.]+)\s*\}?",
         lambda m: (float(m.group(1)), float(m.group(2))))
    # X.Xs to Y.Ys / X.Xs-Y.Ys
    grab(r"(\d+(?:\.\d+)?)\s*s\b\s*(?:to|-|–|—|~)\s*(\d+(?:\.\d+)?)\s*s\b",
         lambda m: (float(m.group(1)), float(m.group(2))))
    # X to Y seconds/secs/sec
    grab(r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—|~)\s*(\d+(?:\.\d+)?)\s*(?:seconds|secs|sec)\b",
         lambda m: (float(m.group(1)), float(m.group(2))))
    # 마지막(가장 느슨함): bare "X-Y" / "X to Y"
    grab(r"(\d+(?:\.\d+)?)\s*(?:to|-|–|—|~)\s*(\d+(?:\.\d+)?)",
         lambda m: (float(m.group(1)), float(m.group(2))))

    segs.sort()
    return segs


def parse_pred_segments(raw_output, fmt):
    if fmt == "token":
        return parse_token_format(raw_output)
    if fmt == "plain":
        return parse_plain_format(raw_output)
    # auto: token 우선 시도, 없으면 plain
    segs = parse_token_format(raw_output)
    return segs if segs else parse_plain_format(raw_output)


def merge_overlapping(segs):
    """겹치는 구간을 병합(개수를 바꾸므로 --merge-overlap 일 때만 호출)."""
    if not segs:
        return []
    norm = sorted((min(s, e), max(s, e)) for s, e in segs)
    out = [list(norm[0])]
    for s, e in norm[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


# ============================== IO ==============================
def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(rec):
    return (rec.get("video_id"), rec.get("query"))


# ============================== 평가 ==============================
def evaluate_model(gt_rows, pred_rows, fmt, unparseable, merge_overlap):
    """GT/예측 jsonl → per_sample=[(N_gt,N_pred), ...] 와 카운트 딕셔너리.

    엣지 케이스:
      - N_gt==0 (GT 자체가 비정상): 항상 제외(S_multi/S_single 어디에도 안 들어감).
      - 예측 파일에 해당 (video_id, query) 가 아예 없음(missing): 파싱 실패와 동일하게 취급.
      - 파싱 실패 또는 세그먼트 0개: N_pred=0 이 됨.
          -> CR 항(min/max)은 0이 되어 "완전히 틀림"으로 채점되지만,
             OSA 는 "N_pred<=1"만 보므로 이 경우 오히려 "성공"으로 카운트된다.
             즉 아무것도 못 뽑아낸 모델이 싱글 GT에서는 유리해지는 비대칭이 있음 — 의도된
             정의(스펙에 명시)이지만 왜곡을 피하려면 --unparseable skip 으로 해당 샘플을
             평가에서 통째로 제외할 수 있다.
    """
    pred_by_key = {_key(r): r.get("raw_output", "") for r in pred_rows}

    counts = {"n_gt_zero": 0, "n_missing_pred": 0, "n_parse_fail": 0, "n_skipped": 0,
              "n_total_gt_rows": len(gt_rows)}
    per_sample = []

    for g in gt_rows:
        gt_segs = g.get("segments") or []
        n_gt = len(gt_segs)
        if n_gt == 0:
            counts["n_gt_zero"] += 1
            continue

        raw = pred_by_key.get(_key(g))
        if raw is None:
            counts["n_missing_pred"] += 1
            pred_segs, unparseable_flag = [], True
        else:
            pred_segs = parse_pred_segments(raw, fmt)
            unparseable_flag = (len(pred_segs) == 0)
            if unparseable_flag:
                counts["n_parse_fail"] += 1

        if unparseable_flag and unparseable == "skip":
            counts["n_skipped"] += 1
            continue

        if merge_overlap:
            pred_segs = merge_overlapping(pred_segs)

        per_sample.append((n_gt, len(pred_segs)))

    return per_sample, counts


def compute_metrics(per_sample):
    """USA/OSA/CountF1. S_multi 또는 S_single 이 비면 해당 값은 None(정의 불가)."""
    multi = [(ng, npd) for ng, npd in per_sample if ng >= 2]
    single = [(ng, npd) for ng, npd in per_sample if ng == 1]

    cr = b = usa = None
    if multi:
        ratios = [min(ng, npd) / max(ng, npd) for ng, npd in multi]  # max>=2>0 항상 안전
        floors = [1.0 / ng for ng, _ in multi]
        cr = sum(ratios) / len(ratios)
        b = sum(floors) / len(floors)
        denom = 1.0 - b  # ng>=2 이므로 floors<=0.5 -> b<1 -> denom>0 항상 보장
        usa = max(0.0, (cr - b) / denom) if denom > 0 else 0.0

    osa = None
    if single:
        osa = sum(1 for ng, npd in single if npd <= 1) / len(single)

    if usa is None or osa is None:
        # 둘 중 하나라도 정의 불가(서브셋이 빔)면 CountF1 도 정의 불가(0으로 강제하지 않음).
        countf1 = None
    else:
        countf1 = 0.0 if (usa == 0.0 and osa == 0.0) else (2 * usa * osa / (usa + osa))

    return {"CR": cr, "b": b, "USA": usa, "OSA": osa, "CountF1": countf1,
            "n_multi": len(multi), "n_single": len(single), "n_total": len(per_sample)}


_BUCKET_KEYS = ["1", "2", "3", "4", ">=5"]


def compute_breakdown(per_sample):
    """N_gt별(1,2,3,4,>=5) 샘플 수 / 평균 N_pred / CR / exact-count acc / count MAE.

    여기서 CR 은 (min/max) 을 해당 버킷에 대해서만 평균낸 값 — S_multi 전체에 대한
    공식 CR(USA 계산용)과는 다르게 N_gt=1 버킷도 포함해 참고용으로 낸다.
    """
    buckets = {k: [] for k in _BUCKET_KEYS}
    for ng, npd in per_sample:
        buckets[str(ng) if ng <= 4 else ">=5"].append((ng, npd))

    out = {}
    for k in _BUCKET_KEYS:
        items = buckets[k]
        if not items:
            out[k] = {"n": 0, "mean_Npred": None, "CR": None, "exact_acc": None, "MAE": None}
            continue
        n = len(items)
        out[k] = {
            "n": n,
            "mean_Npred": sum(npd for _, npd in items) / n,
            "CR": sum(min(ng, npd) / max(ng, npd) for ng, npd in items) / n,
            "exact_acc": sum(1 for ng, npd in items if ng == npd) / n,
            "MAE": sum(abs(ng - npd) for ng, npd in items) / n,
        }
    return out


def compute_b_from_gt(gt_rows):
    """b = mean_{N_gt>=2} 1/N_gt. 예측과 무관, GT 분포만으로 결정."""
    floors = [1.0 / len(g.get("segments") or []) for g in gt_rows if len(g.get("segments") or []) >= 2]
    return (sum(floors) / len(floors)) if floors else None


# ============================== 출력 ==============================
def _fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _print_table(results):
    cols = ["model", "USA", "OSA", "CountF1", "b", "n_multi", "n_single", "n_total", "parse_fail_rate"]
    widths = {c: max(len(c), 10) for c in cols}
    print("\t".join(c.ljust(widths[c]) for c in cols))
    for row, _bd in results:
        print("\t".join(_fmt(row.get(c)).ljust(widths[c]) for c in cols))

    for row, bd in results:
        print(f"\n[{row['model']}] N_gt별 breakdown")
        bcols = ["N_gt", "n", "mean_N_pred", "CR", "exact_acc", "MAE"]
        print("\t".join(bcols))
        for k in _BUCKET_KEYS:
            b = bd[k]
            print("\t".join([k, _fmt(b["n"], 0), _fmt(b["mean_Npred"]), _fmt(b["CR"]),
                              _fmt(b["exact_acc"]), _fmt(b["MAE"])]))


def _write_csv(path, results):
    fieldnames = ["model", "USA", "OSA", "CountF1", "b", "n_multi", "n_single", "n_total",
                  "parse_fail_rate", "n_gt_zero", "n_missing_pred", "n_parse_fail", "n_skipped"]
    bucket_field_map = {"1": "1", "2": "2", "3": "3", "4": "4", ">=5": "ge5"}
    for suffix in bucket_field_map.values():
        fieldnames += [f"ngt{suffix}_n", f"ngt{suffix}_mean_Npred", f"ngt{suffix}_CR",
                       f"ngt{suffix}_exact_acc", f"ngt{suffix}_MAE"]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row, bd in results:
            out = {k: row.get(k) for k in fieldnames if k in row}
            out["model"] = row["model"]
            for k, suffix in bucket_field_map.items():
                b = bd[k]
                out[f"ngt{suffix}_n"] = b["n"]
                out[f"ngt{suffix}_mean_Npred"] = b["mean_Npred"]
                out[f"ngt{suffix}_CR"] = b["CR"]
                out[f"ngt{suffix}_exact_acc"] = b["exact_acc"]
                out[f"ngt{suffix}_MAE"] = b["MAE"]
            w.writerow(out)
    print(f"[SAVED] {path}")


def _write_summary_json(path, row, breakdown, args):
    """maketable.py 가 읽어 table.txt 에 추가 열로 얹을 수 있는 요약 json.
    pairwise_miou_summary.json 과 같은 디렉토리에 저장하면 maketable.py 가 자동으로 픽업한다."""
    payload = {
        "model": row["model"],
        "USA": row["USA"], "OSA": row["OSA"], "CountF1": row["CountF1"], "b": row["b"],
        "n_multi": row["n_multi"], "n_single": row["n_single"], "n_total": row["n_total"],
        "parse_fail_rate": row["parse_fail_rate"],
        "breakdown": breakdown,
        "config": {"format": args.format, "unparseable": args.unparseable,
                   "merge_overlap": args.merge_overlap},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"[SAVED] {path}")


# ============================== 자체 테스트 ==============================
def _selftest(gt_rows):
    ok = True

    # 1) 항상 1개만 예측하는 더미 모델 -> USA == 0 (multi 서브셋에서 min/max <= 1/2 이므로
    #    CR 은 절대 b 를 넘지 못함 -> chance 수준 이하 -> clip 되어 0)
    dummy = [(len(g.get("segments") or []), 1) for g in gt_rows if len(g.get("segments") or []) > 0]
    m1 = compute_metrics(dummy)
    print(f"[selftest 1] 더미(항상 1개 예측): USA={m1['USA']}")
    ok &= (m1["USA"] is not None and abs(m1["USA"]) < 1e-9)
    print("  " + ("PASS" if ok else "FAIL"))

    # 2) GT 그대로 예측 -> USA==1, OSA==1, CountF1==1
    perfect = [(n, n) for n in (len(g.get("segments") or []) for g in gt_rows) if n > 0]
    m2 = compute_metrics(perfect)
    print(f"[selftest 2] GT 그대로 예측: USA={m2['USA']} OSA={m2['OSA']} CountF1={m2['CountF1']}")
    ok2 = (m2["USA"] is not None and abs(m2["USA"] - 1.0) < 1e-9
           and m2["OSA"] is not None and abs(m2["OSA"] - 1.0) < 1e-9
           and m2["CountF1"] is not None and abs(m2["CountF1"] - 1.0) < 1e-9)
    print("  " + ("PASS" if ok2 else "FAIL"))
    ok &= ok2

    # 3) b 재현: UnAV-100 test split GT 를 --gt 로 넘겼다면 ≈0.397 이어야 함(다른 GT면 값이 다름).
    b = compute_b_from_gt(gt_rows)
    print(f"[selftest 3] b(from --gt) = {_fmt(b, 6)}  (UnAV-100 test split 기준 목표 ≈ 0.397)")
    if b is not None:
        diff = abs(b - 0.397)
        print(f"  |b-0.397| = {diff:.6f}  -> " + ("PASS(≈0.397)" if diff < 0.01 else
              "이 --gt 파일이 UnAV-100 test split 이 아니면 다를 수 있음(정상)"))

    return ok


# ============================== CLI ==============================
def _parse_kv(s, flag):
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"{flag} 는 name=path 형식이어야 함: {s}")
    name, path = s.split("=", 1)
    return name, path


def main():
    ap = argparse.ArgumentParser(
        description="UnAV-100 멀티세그먼트 grounding CountF1(개수 기반) 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--gt", required=True, help="GT JSONL: {video_id, query, segments:[[s,e],...]}")
    ap.add_argument("--pred", action="append", default=[], metavar="NAME=PATH",
                     help="예측 JSONL(name=path), 여러 번 지정 가능")
    ap.add_argument("--format", choices=["token", "plain", "auto"], default="auto",
                     help="raw_output 파서 (기본 auto: token 우선, 실패시 plain)")
    ap.add_argument("--unparseable", choices=["zero", "skip"], default="zero",
                     help="파싱 실패/세그먼트 0개(예측 누락 포함) 샘플 처리: "
                          "zero=N_pred=0 으로 포함(기본), skip=평가에서 제외")
    ap.add_argument("--merge-overlap", action="store_true",
                     help="예측 세그먼트끼리 겹치면 병합 후 개수 산정(기본 False=병합 안 함)")
    ap.add_argument("--out-csv", default=None, help="모델 비교 테이블 CSV 저장 경로")
    ap.add_argument("--summary-out", action="append", default=[], metavar="NAME=PATH",
                     help="모델별 요약 json 저장 경로(maketable.py 연동용), 여러 번 지정 가능")
    ap.add_argument("--check-b", action="store_true", help="--gt 만으로 b 계산해 출력하고 종료")
    ap.add_argument("--selftest", action="store_true", help="내장 자체 테스트 실행 후 종료")
    args = ap.parse_args()

    gt_rows = load_jsonl(args.gt)

    if args.check_b:
        b = compute_b_from_gt(gt_rows)
        print(f"b (chance floor, GT 분포만 사용) = {_fmt(b, 6)}")
        print("UnAV-100 test split GT 기준 기대값 ≈ 0.397")
        return

    if args.selftest:
        ok = _selftest(gt_rows)
        sys.exit(0 if ok else 1)

    if not args.pred:
        raise SystemExit("--pred 를 최소 1개 지정해야 함 (예: --pred base=preds.jsonl)")

    summary_out_map = dict(_parse_kv(x, "--summary-out") for x in args.summary_out)

    results = []
    for spec in args.pred:
        name, path = _parse_kv(spec, "--pred")
        pred_rows = load_jsonl(path)
        per_sample, counts = evaluate_model(gt_rows, pred_rows, args.format,
                                             args.unparseable, args.merge_overlap)
        metrics = compute_metrics(per_sample)
        breakdown = compute_breakdown(per_sample)

        n_considered = counts["n_total_gt_rows"] - counts["n_gt_zero"]
        n_unparseable = counts["n_missing_pred"] + counts["n_parse_fail"]
        parse_fail_rate = (n_unparseable / n_considered) if n_considered else None

        row = {"model": name, **metrics, "parse_fail_rate": parse_fail_rate, **counts}
        results.append((row, breakdown))

        if name in summary_out_map:
            _write_summary_json(summary_out_map[name], row, breakdown, args)

    _print_table(results)
    if args.out_csv:
        _write_csv(args.out_csv, results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pv_report.py — [2] 두 파서 비교 + [3] 지표 재계산 + [4] 산출물 생성."""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_data import MODELS_ALL as MODELS  # noqa: E402
from pv_data import (SHORT, load_model_samples, pilot_subset,  # noqa: E402
                     postprocess_llm, rule_parse)
from pv_llm import OUT_ROOT, cache_path  # noqa: E402
from pv_metrics import METRIC_COLS, compute_metrics, stored_metrics  # noqa: E402

TOL = 0.05
BUCKETS = ["<=-3", "-2", "-1", "0", "+1", "+2", ">=+3"]


def bucket(d):
    if d <= -3:
        return "<=-3"
    if d >= 3:
        return ">=+3"
    return {-2: "-2", -1: "-1", 0: "0", 1: "+1", 2: "+2"}[d]


def seg_match(a, b, tol=TOL):
    """정렬 후 모든 구간이 tol 초 내에서 일치하는가."""
    if len(a) != len(b):
        return False
    for (s1, e1), (s2, e2) in zip(sorted(a), sorted(b)):
        if abs(s1 - s2) > tol or abs(e1 - e2) > tol:
            return False
    return True


def load_llm(model):
    p = cache_path(model)
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prev = out.get(r["key"])
            if prev is None or (prev["status"] != "ok" and r["status"] == "ok"):
                out[r["key"]] = r
    return out


def build_rows(model, n_pilot):
    samples = load_model_samples(model)
    if n_pilot:
        samples = pilot_subset(samples, n_pilot)
    llm = load_llm(model)
    rows = []
    for s in samples:
        rec = llm.get(s["key"])
        if rec is None:
            continue
        R = rule_parse(s)
        L = postprocess_llm(s, rec["segments"])   # 단위 환산은 코드가 (LLM 산술 배제)
        rows.append({
            "model": model, "key": s["key"], "idx": s["idx"], "gt_label": s["gt_label"],
            "gt": s["gt"], "rule_pred": R, "llm_pred": L,
            "llm_status": rec["status"], "json_fallback": rec.get("json_fallback", False),
            "count_diff": len(R) - len(L),
            "count_match": len(R) == len(L),
            "exact_match": seg_match(R, L),
            "scorer_text": s["scorer_text"], "llm_text": s["llm_text"],
            "duration": s["duration"],
        })
    return rows


def agg_compare(rows):
    n = len(rows)
    if n == 0:
        return None
    hist = {b: 0 for b in BUCKETS}
    for r in rows:
        hist[bucket(r["count_diff"])] += 1
    return {
        "n": n,
        "count_match_rate": sum(r["count_match"] for r in rows) / n,
        "exact_match_rate": sum(r["exact_match"] for r in rows) / n,
        "mean_count_diff": sum(r["count_diff"] for r in rows) / n,
        "hist": hist,
        "llm_json_error": sum(1 for r in rows if r["llm_status"] == "json_error"),
        "llm_api_error": sum(1 for r in rows if r["llm_status"] == "api_error"),
        "llm_refusal": sum(1 for r in rows if r["llm_status"] == "refusal"),
        "llm_json_fallback": sum(1 for r in rows if r["json_fallback"]),
        "rule_parse_fail": sum(1 for r in rows if not r["rule_pred"]),
        "llm_parse_fail": sum(1 for r in rows if not r["llm_pred"]),
    }


def fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def metric_table(per_model, full_run):
    """기존 → 재계산 (Δ) 표. Δ 는 동일 코드·동일 부분집합의 룰기반 값 대비."""
    lines = []
    head = "| 모델 | " + " | ".join(METRIC_COLS) + " |"
    lines += [head, "|" + "---|" * (len(METRIC_COLS) + 1)]
    deltas = []
    for m in MODELS:
        pm = per_model.get(m)
        if not pm:
            continue
        cells = []
        for k in METRIC_COLS:
            base, new = pm["rule"][k], pm["llm"][k]
            if base is None or new is None:
                cells.append("-")
                continue
            d = new - base
            deltas.append((m, k, abs(d)))
            nd = 2 if k.startswith(("mIoU", "F1")) else 4
            cells.append(f"{base:.{nd}f} → {new:.{nd}f} ({d:+.{nd}f})")
        lines.append(f"| {SHORT[m]} | " + " | ".join(cells) + " |")
    return "\n".join(lines), deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    n_pilot = 0 if args.full else args.pilot

    os.makedirs(OUT_ROOT, exist_ok=True)
    rnd = random.Random(args.seed)

    all_rows, per_model, cmp_stats = [], {}, {}
    for m in MODELS:
        rows = build_rows(m, n_pilot)
        if not rows:
            print(f"[skip] {SHORT[m]}: LLM 캐시 없음")
            continue
        all_rows += rows
        cmp_stats[m] = agg_compare(rows)
        per_model[m] = {
            "rule": compute_metrics([(r["gt"], r["rule_pred"]) for r in rows]),
            "llm": compute_metrics([(r["gt"], r["llm_pred"]) for r in rows]),
            "stored": stored_metrics(m),
        }

    if not all_rows:
        sys.exit("LLM 파싱 캐시가 비어 있습니다. 먼저 pv_llm.py 를 실행하세요.")

    full_run = all(v["n"] == 3455 for v in cmp_stats.values())
    scope = "전체 3455 샘플/모델" if full_run else f"파일럿 {cmp_stats[MODELS[0]]['n']} 샘플/모델"

    # ---------- per_sample.jsonl ----------
    with open(os.path.join(OUT_ROOT, "per_sample.jsonl"), "w") as f:
        for r in all_rows:
            f.write(json.dumps({k: v for k, v in r.items()
                                if k not in ("scorer_text", "llm_text")},
                               ensure_ascii=False) + "\n")

    # ---------- mismatch_samples.txt ----------
    with open(os.path.join(OUT_ROOT, "mismatch_samples.txt"), "w") as f:
        f.write(f"불일치 샘플 육안확인 덤프 ({scope}, tolerance {TOL}s)\n")
        f.write("R = 룰기반 파서(eval_miou.py) / L = 독립 LLM 파서\n" + "=" * 100 + "\n")
        for m in MODELS:
            rows = [r for r in all_rows if r["model"] == m]
            mis = [r for r in rows if not r["exact_match"]]
            f.write(f"\n\n{'#'*100}\n# {SHORT[m]} — 불일치 {len(mis)}/{len(rows)}건 중 최대 20건\n{'#'*100}\n")
            for r in (mis if len(mis) <= 20 else rnd.sample(mis, 20)):
                f.write(f"\n--- key={r['key']}  label={r['gt_label']}  "
                        f"count_diff={r['count_diff']:+d}  llm_status={r['llm_status']}\n")
                if r["duration"]:
                    f.write(f"    duration = {r['duration']}s\n")
                f.write(f"    GT   : {r['gt']}\n")
                f.write(f"    R    : {r['rule_pred']}\n")
                f.write(f"    L    : {r['llm_pred']}\n")
                f.write(f"    채점기가 본 텍스트 : {r['scorer_text'][:600]!r}\n")
                if r["llm_text"] != r["scorer_text"]:
                    f.write(f"    LLM 이 본 raw 원문 : {r['llm_text'][:900]!r}\n")

    # ---------- metrics_comparison.md ----------
    table, deltas = metric_table(per_model, full_run)
    miou_d = [d for _, k, d in deltas if k == "mIoU"]
    abs_all = [d for _, _, d in deltas]
    worst = max(deltas, key=lambda x: x[2])

    L = []
    L.append("# Baseline 파서 검증 — 룰기반 vs 독립 LLM 파서\n")
    L.append(f"- 범위: {scope}, UnAV-100 멀티세그 프롬프트, baseline 5종")
    L.append("- LLM 파서: `claude-sonnet-4-6`, 룰기반 파서 코드를 참조하지 않고 독립 구현")
    L.append("- 입력: **래퍼 전처리 이전의 진짜 raw generation** "
             "(MUSEG=`raw`, AVicuna=`raw_pred`+duration, 나머지=모델 원문)")
    L.append("- 지표 코드는 `eval_miou.py` 원본 그대로. 파서만 교체했다.")
    L.append("- 규약: mIoU=sample merge, F1@k=pairwise merge, USA=`CR_star`, OSA=`1-FMR`\n")

    L.append("## [2] 파서 일치도\n")
    L.append("| 모델 | n | count_match | exact_match(±0.05s) | mean_count_diff | R 파싱실패 | L 파싱실패 | JSON오류 | API오류 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        c = cmp_stats.get(m)
        if not c:
            continue
        L.append(f"| {SHORT[m]} | {c['n']} | {100*c['count_match_rate']:.2f}% | "
                 f"{100*c['exact_match_rate']:.2f}% | {c['mean_count_diff']:+.4f} | "
                 f"{c['rule_parse_fail']} | {c['llm_parse_fail']} | "
                 f"{c['llm_json_error']} | {c['llm_api_error']} |")
    tot_n = sum(c["n"] for c in cmp_stats.values())
    L.append(f"| **전체** | {tot_n} | "
             f"{100*sum(c['count_match_rate']*c['n'] for c in cmp_stats.values())/tot_n:.2f}% | "
             f"{100*sum(c['exact_match_rate']*c['n'] for c in cmp_stats.values())/tot_n:.2f}% | "
             f"{sum(c['mean_count_diff']*c['n'] for c in cmp_stats.values())/tot_n:+.4f} | "
             f"{sum(c['rule_parse_fail'] for c in cmp_stats.values())} | "
             f"{sum(c['llm_parse_fail'] for c in cmp_stats.values())} | "
             f"{sum(c['llm_json_error'] for c in cmp_stats.values())} | "
             f"{sum(c['llm_api_error'] for c in cmp_stats.values())} |")

    L.append("\n### count_diff 분포  (= len(R) − len(L); 양수 = 룰기반이 더 많이 뽑음)\n")
    L.append("| 모델 | " + " | ".join(BUCKETS) + " |")
    L.append("|" + "---|" * (len(BUCKETS) + 1))
    for m in MODELS:
        c = cmp_stats.get(m)
        if not c:
            continue
        L.append(f"| {SHORT[m]} | " + " | ".join(str(c["hist"][b]) for b in BUCKETS) + " |")

    L.append("\n## [3] 지표 재계산 — 기존(룰기반) → 재계산(LLM) (Δ)\n")
    L.append(table)
    L.append(f"\n- **절대 델타 최대값: {max(abs_all):.4f}** "
             f"({SHORT[worst[0]]} / {worst[1]})")
    L.append(f"- 절대 델타 평균: {sum(abs_all)/len(abs_all):.4f}  (전체 {len(abs_all)}개 셀)")
    L.append(f"- mIoU 만: 최대 {max(miou_d):.4f} / 평균 {sum(miou_d)/len(miou_d):.4f}")

    if full_run:
        L.append("\n### 참고: 논문에 실린 저장값과 룰기반 재계산의 재현오차\n")
        L.append("| 모델 | " + " | ".join(METRIC_COLS) + " |")
        L.append("|" + "---|" * (len(METRIC_COLS) + 1))
        for m in MODELS:
            pm = per_model.get(m)
            if not pm:
                continue
            cs = [fmt(abs(pm["rule"][k] - pm["stored"][k]), 6)
                  if pm["rule"][k] is not None and pm["stored"][k] is not None else "-"
                  for k in METRIC_COLS]
            L.append(f"| {SHORT[m]} | " + " | ".join(cs) + " |")

    L.append("\n## 한 문장 요약 (rebuttal 용)\n")
    overall_cm = 100 * sum(c["count_match_rate"] * c["n"] for c in cmp_stats.values()) / tot_n
    L.append(f"> 독립 LLM 파서로 재파싱했을 때 구간 개수가 {overall_cm:.1f}% 일치했고, "
             f"재계산한 baseline 지표는 mIoU 기준 최대 ±{max(miou_d):.2f} 이내로 동일했다.")

    with open(os.path.join(OUT_ROOT, "metrics_comparison.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\n[saved] {OUT_ROOT}/{{metrics_comparison.md, mismatch_samples.txt, per_sample.jsonl}}")


if __name__ == "__main__":
    main()

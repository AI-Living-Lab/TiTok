#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_qwen3omni.py — Qwen3-Omni-30B-A3B-Instruct 추론 결과 채점 래퍼.
eval_chronus.py 와 동일한 패턴 (결과 스키마가 {"id","question","output"} 이라
pred/gt_segments 를 embed 해서 eval_miou.py 에 넘기는 것까지 동일).

사용:
  python3 eval_qwen3omni.py \
     --results_dir /workspace/outputs/base/Qwen3Omni/unav100_qwen3omni/results \
     --test_json   /workspace/data/test/unav100_qwen3omni.json \
     [--eval_dir <출력폴더>] [--label Qwen3-Omni-30B-A3B-Instruct] [--testset unav100_qwen3omni]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_RESULTS = "/workspace/outputs/base/Qwen3Omni/unav100_qwen3omni/results"
DEF_TESTJSON = "/workspace/data/test/unav100_qwen3omni.json"


def _load(p):
    with open(p) as f:
        return json.load(f)


def merge_results(results_dir):
    if os.path.isfile(results_dir):
        return _load(results_dir)
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))
    if not chunks:
        raise SystemExit(f"[eval_qwen3omni] chunk_*.json 없음: {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_qwen3omni] merged {len(chunks)} chunks -> {len(out)} samples")
    return out


def transform(results, test_json):
    gt = {x["id"]: x for x in _load(test_json)}
    conv, miss = [], 0
    for r in results:
        g = gt.get(r["id"])
        if g is None:
            miss += 1
            continue
        conv.append({
            "id": r["id"],
            "gt_label": g.get("gt_label", ""),
            "gt_segments": g.get("gt_segments", []),
            "pred": r.get("output", ""),
        })
    if miss:
        print(f"[eval_qwen3omni] [WARN] {miss} 샘플 id 매칭 실패 -> 제외")
    print(f"[eval_qwen3omni] transformed {len(conv)} samples (output->pred, gt embedded)")
    return conv


def main():
    ap = argparse.ArgumentParser(description="Qwen3-Omni 결과 채점 -> table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--test_json", default=DEF_TESTJSON)
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--label", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--testset", default="unav100_qwen3omni")
    args = ap.parse_args()

    base = args.results_dir if os.path.isdir(args.results_dir) \
        else os.path.dirname(args.results_dir)
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(base), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    results = merge_results(args.results_dir)
    conv = transform(results, args.test_json)
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    print(f"[eval_qwen3omni] wrote {rank0}")

    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_qwen3omni] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_qwen3omni] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # --- 파싱실패율 리포트 (eval_miou.py --natural 파서 기준) -------------------
    #  parse_fail = output(=pred) 문자열은 있으나 자연어 파서가 시간구간을 하나도
    #  못 뽑은 샘플. infer 단계의 빈 output(추론실패)도 여기 포함되므로 함께 표기.
    pw = os.path.join(eval_dir, "pairwise_miou_summary.json")
    if os.path.exists(pw):
        s = _load(pw)
        n = s.get("n_samples", 0)
        empty_pred = sum(1 for r in results if not str(r.get("output", "")).strip())
        print("\n" + "=" * 60)
        print("  파싱 리포트 (natural parser)")
        print(f"    samples            : {n}")
        print(f"    parse_ok           : {s.get('parse_ok')}")
        print(f"    parse_fail         : {s.get('parse_fail')}")
        print(f"    parse-fail rate    : {s.get('parse_fail_rate_%')}%")
        print(f"    (그중 빈 output)   : {empty_pred}  "
              f"({100.0 * empty_pred / max(n, 1):.2f}%)")
        print("=" * 60)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_qwen3omni] DONE -> {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()

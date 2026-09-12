#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_museg_charades.py — MUSEG(Charades-STA, 청크 추론) 결과 채점 래퍼.

eval_museg.py(unav100용) 와 완전히 동일한 패턴이며 charades 로컬 기본 경로만 다르다.

infer_charades_chunks.py 결과 = results_dir/chunk_*.json, 각 항목 {"id","video","question","output"}.
파이프라인:
  1) results_dir 의 chunk_*.json 병합
  2) id 로 test_json(charades_sta_museg.json) 매칭 → gt_label/gt_segments embed, output→pred
  3) eval_dir/test_results_rank0.json 저장
  4) eval_miou.py --natural  (museg 형식 X.XX-X.XX 소수초, finditer 관대 파싱)
  5) maketable.py → eval_dir/table.txt

사용(인자 없이도 표준 경로로 동작):
  python3 eval_museg_charades.py
"""
import argparse, glob, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEF_RESULTS = "/workspace/outputs/base/MUSEG/charades_sta/results"
DEF_TESTJSON = "/workspace/data/test/charades_sta_museg.json"


def _load(p):
    with open(p) as f:
        return json.load(f)


def merge_results(results_dir):
    if os.path.isfile(results_dir):
        return _load(results_dir)
    chunks = sorted(glob.glob(os.path.join(results_dir, "chunk_*.json")))
    if not chunks:
        raise SystemExit(f"[eval_museg_charades] chunk_*.json 없음: {results_dir}")
    out = []
    for c in chunks:
        out.extend(_load(c))
    print(f"[eval_museg_charades] merged {len(chunks)} chunks → {len(out)} samples")
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
        print(f"[eval_museg_charades] [WARN] {miss} 샘플 id 매칭 실패 → 제외")
    print(f"[eval_museg_charades] transformed {len(conv)} samples (output→pred, gt embedded)")
    return conv


def main():
    ap = argparse.ArgumentParser(description="MUSEG(Charades-STA) 결과 채점 → table.txt")
    ap.add_argument("--results_dir", default=DEF_RESULTS)
    ap.add_argument("--test_json", default=DEF_TESTJSON)
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--label", default="MUSEG-7B")
    ap.add_argument("--testset", default="charades_sta")
    args = ap.parse_args()

    base = args.results_dir if os.path.isdir(args.results_dir) else os.path.dirname(args.results_dir)
    eval_dir = args.eval_dir or os.path.join(os.path.dirname(base), "eval")
    os.makedirs(eval_dir, exist_ok=True)

    results = merge_results(args.results_dir)
    conv = transform(results, args.test_json)
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    with open(rank0, "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)
    print(f"[eval_museg_charades] wrote {rank0}")

    cmd = [sys.executable, os.path.join(HERE, "eval_miou.py"), rank0,
           "--natural", "--label", args.label, "--testset", args.testset]
    print("[eval_museg_charades] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    cmd = [sys.executable, os.path.join(HERE, "maketable.py"), eval_dir]
    print("[eval_museg_charades] $", " ".join(cmd))
    subprocess.run(cmd, check=True)

    table = os.path.join(eval_dir, "table.txt")
    print(f"\n[eval_museg_charades] DONE → {table}\n")
    if os.path.exists(table):
        with open(table) as f:
            print(f.read())


if __name__ == "__main__":
    main()

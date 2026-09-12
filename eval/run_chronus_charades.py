#!/usr/bin/env python3
"""ChronusOmni charades LoRA(chronus_charades) 청크 추론 + 채점 — infer_chronus_sft.py charades 버전.

infer_chronus_sft.py(unav100용)와 완전히 동일한 파이프라인(모델 1회 로드 → 청크 순차
추론 → 청크마다 원자적 저장 + 즉시 채점 → 마지막에 최종 채점)이며, 기본값만
charades_sta_chronus.json / chronus_charades 체크포인트로 바뀐 버전이다. 실제 추론 로직
(run_one/split_chunks/run_scoring)은 이 파일이 아니라 infer_chronus_sft.py 것을 그대로
import 해서 쓴다 — 로직 중복 없이 unav100/charades 두 eval이 항상 같은 코드로 채점됨.

charades_sta_chronus.json 은 build_charades_chronus_eval_data.py 로
/workspace/data/test/charades_sta_museg.json 에서 만든 것 (question을 Chronus 포맷
"second{start}-second{end}" 로 재작성, 오디오 미사용이라 audio 필드 없음, charades는
전부 단일세그라 멀티세그 안내문 없음).

────────────────────────────────────────────────────────────────
사용 (반드시 Chronus128 conda env 의 python 으로 실행 — chronusomni env 는
Blackwell GPU(sm_120) 미지원 torch라 여기선 안 씀, scripts/finetune_lora_charades_sta.sh
학습 때 쓴 것과 같은 env):

  source /workspace/setup.sh && conda activate Chronus128
  cd /workspace/Team4/eval
  python3 run_chronus_charades.py

인자로 오버라이드도 가능 (infer_chronus_sft.py 와 동일한 인자셋):
  python3 run_chronus_charades.py --chunk_size 500 --no_score

학습 전/후 비교용으로 LoRA 없이 base만 평가하고 싶으면:
  python3 run_chronus_charades.py --lora_ckpt base \
      --out_dir /workspace/outputs/base/ChronusOmni/charades_sta_chronus

전제: --lora_ckpt (기본 /workspace/checkpoints/sft/chronus_charades) 는 학습이 100% 끝난
뒤의 output_dir 루트여야 함 (non_lora_trainables.bin 이 학습 완전 종료 시 딱 한 번만
그 위치에 저장됨 — checkpoint-375/checkpoint-750 같은 중간 서브폴더에는 없음).

결과물 (OUT_DIR = 기본 /workspace/outputs/sft/chronus_charades/charades_sta_chronus):
  inputs/chunk_XXXX.json   : 쪼개진 입력
  results/chunk_XXXX.json  : 청크별 추론 결과
  eval/table.txt           : ★ 최종 산출물 (sample_mIoU 등 집계 1행, maketable.py 형식)
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# infer_chronus_sft 를 import 하는 순간 os.environ 비디오 전처리 매크로 세팅 +
# sys.path에 Chronus repo 추가 + os.chdir(CHRONUS_ROOT) 가 전부 일어난다
# (whisper/BEATs 가중치를 상대경로로 찾으므로 cwd가 Chronus repo 루트여야 함).
import infer_chronus_sft as base

import argparse
import json
import subprocess
import time
import traceback

from tqdm import tqdm

DEF_LORA_CKPT = "/workspace/checkpoints/sft/chronus_charades"
DEF_MODEL_BASE = "/workspace/checkpoints/base/ChronusOmni"
DEF_TEST_JSON = "/workspace/data/test/charades_sta_chronus.json"
DEF_OUT_DIR = "/workspace/outputs/sft/chronus_charades/charades_sta_chronus"
DEF_LABEL = "ChronusOmni-charades-SFT"
DEF_TESTSET = "charades_sta_chronus"

COUNTF1_PY = os.path.join(SCRIPT_DIR, "eval_countf1_unav100.py")
MAKETABLE_PY = os.path.join(SCRIPT_DIR, "maketable.py")


def run_countf1_scoring(eval_dir, label, quiet=True):
    """maketable.py 의 USA(uv)/OSA(uv)/CountF1(uv)/b(uv)/parsefail(uv) 열을 채우는 보조 채점.

    eval_chronus.py(run_scoring)가 이미 만들어둔 test_results_rank0.json(id, gt_segments,
    pred)을 eval_countf1_unav100.py 가 요구하는 (video_id, query) 키 스키마의 GT/pred jsonl
    로 변환해 돌린 뒤, countf1_unav100_summary.json 을 eval_dir 에 저장하고 maketable.py 를
    다시 돌려 table.txt 에 반영한다. charades_sta 는 전부 단일세그(N_gt==1)라 USA/CountF1/b
    는 정의상 '-'로 남고(멀티세그 서브셋이 비어서), OSA(=싱글 GT를 안 쪼갠 비율)만 채워진다
    (eval_countf1_unav100.py 의 정의 참고).
    """
    rank0 = os.path.join(eval_dir, "test_results_rank0.json")
    if not os.path.exists(rank0):
        return
    with open(rank0) as f:
        rows = json.load(f)

    gt_path = os.path.join(eval_dir, "_countf1_gt.jsonl")
    pred_path = os.path.join(eval_dir, "_countf1_pred.jsonl")
    with open(gt_path, "w") as gf, open(pred_path, "w") as pf:
        for r in rows:
            vid = r["id"]
            gf.write(json.dumps({"video_id": vid, "query": "", "segments": r.get("gt_segments", [])}, ensure_ascii=False) + "\n")
            pf.write(json.dumps({"video_id": vid, "query": "", "raw_output": r.get("pred", "")}, ensure_ascii=False) + "\n")

    summary_path = os.path.join(eval_dir, "countf1_unav100_summary.json")
    cmd = [sys.executable, COUNTF1_PY, "--gt", gt_path, "--pred", f"{label}={pred_path}",
           "--format", "auto", "--summary-out", f"{label}={summary_path}"]
    kw = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if quiet else {}
    try:
        subprocess.run(cmd, check=True, **kw)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] countf1(uv) 채점 실패(무시하고 계속 진행): {e}")
        return

    # countf1_unav100_summary.json 을 새로 만들었으니 table.txt 도 다시 만들어 반영
    subprocess.run([sys.executable, MAKETABLE_PY, eval_dir], check=False, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora_ckpt", default=DEF_LORA_CKPT,
                     help="LoRA 체크포인트 디렉토리(non_lora_trainables.bin 있는 학습 완료 후 output_dir 루트). base/no 면 LoRA 미사용")
    ap.add_argument("--model_base", default=DEF_MODEL_BASE)
    ap.add_argument("--test_json", default=DEF_TEST_JSON)
    ap.add_argument("--out_dir", default=DEF_OUT_DIR)
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--frame_num", type=int, default=64)
    ap.add_argument("--label", default=DEF_LABEL)
    ap.add_argument("--testset", default=DEF_TESTSET)
    ap.add_argument("--no_score", action="store_true", help="채점 단계를 건너뛰고 추론만 한다")
    args = ap.parse_args()

    is_lora = args.lora_ckpt.lower() not in ("base", "no", "none", "")
    if is_lora and not os.path.exists(os.path.join(args.lora_ckpt, "non_lora_trainables.bin")):
        raise SystemExit(
            f"[run_chronus_charades] {args.lora_ckpt}/non_lora_trainables.bin 이 없습니다.\n"
            "  학습이 아직 안 끝났거나(체크포인트만 있고 최종 저장 전), --lora_ckpt 경로가\n"
            "  checkpoint-XXX 같은 중간 서브폴더를 가리키고 있는지 확인하세요.\n"
            "  (이 파일은 train.py 가 학습 루프를 100% 마친 뒤 output_dir 루트에만 씁니다.)"
        )

    inputs_dir = os.path.join(args.out_dir, "inputs")
    results_dir = os.path.join(args.out_dir, "results")
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.test_json) as f:
        data = json.load(f)
    n_chunks = base.split_chunks(data, args.chunk_size, inputs_dir)
    print(f"[run_chronus_charades] {len(data)} samples -> {n_chunks} chunks of {args.chunk_size} (inputs: {inputs_dir})")

    if is_lora:
        print(f"[run_chronus_charades] loading base={args.model_base}  LoRA={args.lora_ckpt}  (merge_and_unload in-memory, flash_attn=True)")
        tokenizer, model, image_processor, _ = base.load_pretrained_model(
            args.lora_ckpt, args.model_base, is_lora=True, use_flash_attn=True)
    else:
        print(f"[run_chronus_charades] loading base only: {args.model_base} (flash_attn=True)")
        tokenizer, model, image_processor, _ = base.load_pretrained_model(
            args.model_base, None, use_flash_attn=True)
    model = model.to('cuda').eval()
    model = model.bfloat16()
    print("[run_chronus_charades] model ready.")

    for i in range(n_chunks):
        chunk_path = os.path.join(inputs_dir, f"chunk_{i:04d}.json")
        result_path = os.path.join(results_dir, f"chunk_{i:04d}.json")
        if os.path.exists(result_path):
            print(f"[chunk {i + 1}/{n_chunks}] skip (already done) -> {result_path}")
            continue
        with open(chunk_path) as f:
            chunk_samples = json.load(f)

        t0 = time.time()
        chunk_results = []
        n_fail = 0
        for sample in tqdm(chunk_samples, desc=f"chunk {i + 1}/{n_chunks}"):
            try:
                out = base.run_one(sample, tokenizer, model, image_processor, args.frame_num)
            except Exception as e:
                n_fail += 1
                traceback.print_exc()
                out = {"id": sample["id"], "question": sample.get("question", ""), "output": "", "error": str(e)}
            chunk_results.append(out)

        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(chunk_results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, result_path)
        print(f"[chunk {i + 1}/{n_chunks}] done: {len(chunk_results)} samples "
              f"({n_fail} failed) in {time.time() - t0:.1f}s -> {result_path}")

        if not args.no_score:
            print(f"[chunk {i + 1}/{n_chunks}] 중간 채점 중... (GPU 안 씀, 지금까지 결과만 반영)")
            base.run_scoring(results_dir, args.test_json, eval_dir, args.label, args.testset, quiet=True)
            run_countf1_scoring(eval_dir, args.label, quiet=True)

    print(f"\n[run_chronus_charades] 추론 ALL DONE -> {results_dir}")

    if not args.no_score:
        print("[run_chronus_charades] 최종 채점 실행...")
        base.run_scoring(results_dir, args.test_json, eval_dir, args.label, args.testset, quiet=False)
        run_countf1_scoring(eval_dir, args.label, quiet=False)
        table = os.path.join(eval_dir, "table.txt")
        print(f"\n[run_chronus_charades] ★ 최종 결과 -> {table}")
    else:
        print("[run_chronus_charades] --no_score 지정됨: 채점 생략. 나중에 수동으로:")
        print(f"  python3 {base.EVAL_CHRONUS_PY} --results_dir {results_dir} "
              f"--test_json {args.test_json} --eval_dir {eval_dir} --label {args.label} --testset {args.testset}")


if __name__ == "__main__":
    main()

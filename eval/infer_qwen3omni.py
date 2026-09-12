#!/usr/bin/env python3
"""Qwen3-Omni-30B-A3B-Instruct UnAV-100 청크 추론 — resume 안전, 원샷 실행.

infer_chronus_sft.py 와 동일한 골격(청크 분할 -> 순차 추론 -> 매 청크 채점 -> 최종 table.txt)
이지만 모델 호출부만 Qwen3-Omni 공식 API(Qwen3OmniMoeForConditionalGeneration +
Qwen3OmniMoeProcessor + qwen_omni_utils.process_mm_info)로 바뀐다.

입력: data/test/unav100_qwen3omni.json (build_unav100_qwen3omni.py 로 생성).
      각 샘플의 "messages" content 는 [video, audio, text] 3항목:
        - video : /workspace/datasets/unav_100/videos/<id>.mp4
        - audio : /workspace/datasets/unav_100/audio/<id>.wav  (큐레이션된 이벤트 .wav)
        - text  : 멀티세그 힌트 프롬프트

오디오 투입 방식 (--use_audio_in_video):
  0 (기본) : messages 의 명시적 .wav(audio 항목)를 그대로 process_mm_info 로 넣는다.
             경로 기반이라 재현성이 좋고, UnAV-100 큐레이션 .wav 를 정확히 사용한다.
  1        : audio 항목을 떼고 mp4 컨테이너 트랙을 use_audio_in_video=True 로 시간정렬
             (TMRoPE)해서 넣는다. "언제" 질의에 이론상 유리하나 mp4 트랙 품질에 의존.
  두 방식 다 동시에 켜면 오디오가 이중 투입되므로, 이 스크립트가 배타적으로 처리한다.

메모리 안전장치 (이 GPU = RTX PRO 6000 Blackwell, ~95.6GiB):
  - model.disable_talker() : talker(오디오 출력) 미사용. 공식 표 기준 Instruct 는
    talker 포함 시 60초 영상 107.74GB(OOM), talker 제외 시 ~95.76GB. UnAV-100 은
    대부분 짧지만(<=60s) 긴 샘플은 아슬아슬하므로 talker 는 반드시 끈다.
  - attn_implementation="flash_attention_2" : 공식 권장.
  - per-sample OOM 시 (fps, max_pixels) 를 단계적으로 낮춰(ladder) 자동 재시도하고,
    그래도 실패하면 그 샘플만 {"output": "", "error": ...} 로 남기고 계속 진행한다
    (청크 전체가 죽지 않음).

리포트:
  - 각 청크 종료 시: 그 청크 실패 수 + 누적 추론실패율(infer-fail rate).
  - 매 청크 채점(eval_qwen3omni.py)이 파싱실패율(parse-fail rate)까지 출력.
  - 최종: 누적 추론실패율 + 최종 채점(table.txt, parse-fail rate 포함).

사용 (보통은 run_qwen3omni_infer.sh 를 통해 실행):
  python3 infer_qwen3omni.py \
    --model_path /workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct \
    --test_json  /workspace/data/test/unav100_qwen3omni.json \
    --out_dir    /workspace/outputs/base/Qwen3Omni/unav100_qwen3omni \
    --chunk_size 500
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import glob
import json
import time
import copy
import argparse
import traceback
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_QWEN3OMNI_PY = os.path.join(SCRIPT_DIR, "eval_qwen3omni.py")

# OOM 재시도 사다리: (fps, max_pixels).  None = 라이브러리 기본값.
#   기본(2fps, 기본해상도) -> 1fps -> 1fps+해상도축소 -> 0.5fps+더 축소
OOM_LADDER = [
    (None, None),
    (1.0, None),
    (1.0, 256 * 28 * 28),
    (0.5, 128 * 28 * 28),
]


# ----------------------------------------------------------------------------- #
# 청크 분할
# ----------------------------------------------------------------------------- #
def split_chunks(data, chunk_size, chunks_dir):
    os.makedirs(chunks_dir, exist_ok=True)
    n_chunks = (len(data) + chunk_size - 1) // chunk_size if data else 0
    for i in range(n_chunks):
        p = os.path.join(chunks_dir, f"chunk_{i:04d}.json")
        if not os.path.exists(p):
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data[i * chunk_size:(i + 1) * chunk_size], f,
                          ensure_ascii=False, indent=2)
            os.replace(tmp, p)
    return n_chunks


# ----------------------------------------------------------------------------- #
# 샘플 1건 추론
# ----------------------------------------------------------------------------- #
def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error" in msg or "cublas" in msg


def _prep_messages(sample, use_audio_in_video, fps, max_pixels):
    """messages 를 복제하고 fps/max_pixels 적용, 오디오 투입 방식에 맞게 항목 정리."""
    messages = copy.deepcopy(sample["messages"])
    content = messages[0]["content"]

    # video 항목에 디코딩 파라미터 주입
    for item in content:
        if item.get("type") == "video":
            if fps is not None:
                item["fps"] = fps
            if max_pixels is not None:
                item["max_pixels"] = max_pixels

    if use_audio_in_video:
        # mp4 트랙을 쓰므로 명시적 audio 항목은 제거 (이중 투입 방지)
        messages[0]["content"] = [it for it in content if it.get("type") != "audio"]
    return messages


def _extract_question(sample) -> str:
    for it in sample["messages"][0]["content"]:
        if it.get("type") == "text":
            return it["text"]
    return sample.get("question", "")


def run_one(sample, model, processor, use_audio_in_video, max_new_tokens,
            torch, process_mm_info):
    question = _extract_question(sample)
    last_err = None
    for fps, max_pixels in OOM_LADDER:
        messages = _prep_messages(sample, use_audio_in_video, fps, max_pixels)
        try:
            with torch.inference_mode():
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(
                    messages, use_audio_in_video=use_audio_in_video)
                inputs = processor(
                    text=text, audio=audios, images=images, videos=videos,
                    return_tensors="pt", padding=True,
                    use_audio_in_video=use_audio_in_video,
                    # transformers 5.x: qwen-vl-utils 레퍼런스와 동일하게 프레임당 픽셀 상한
                    # 적용(안 하면 일부 영상이 토큰/메모리를 훨씬 많이 먹음). v5.22부터 기본값.
                    cap_pixels_per_frame=True,
                )
                inputs = inputs.to(model.device).to(model.dtype)
                gen_out = model.generate(
                    **inputs,
                    thinker_return_dict_in_generate=True,
                    thinker_max_new_tokens=max_new_tokens,
                    thinker_do_sample=False,
                    return_audio=False,
                    use_audio_in_video=use_audio_in_video,
                )
                # generate 는 (text_ids, audio) 튜플 또는 dict-like 를 반환할 수 있음
                if isinstance(gen_out, (tuple, list)):
                    gen_out = gen_out[0]
                seq = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
                output = processor.batch_decode(
                    seq[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )[0].strip()
            rec = {"id": sample["id"], "question": question, "output": output}
            if (fps, max_pixels) != OOM_LADDER[0]:
                rec["downscaled"] = {"fps": fps, "max_pixels": max_pixels}
            return rec
        except Exception as e:  # noqa: BLE001
            last_err = e
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            if not _is_oom(e):
                break
            print(f"[infer_qwen3omni] OOM on {sample['id']} at fps={fps} "
                  f"max_pixels={max_pixels}, 재시도(더 축소)...", flush=True)
    traceback.print_exc()
    return {"id": sample["id"], "question": question, "output": "",
            "error": str(last_err)}


# ----------------------------------------------------------------------------- #
# 누적 추론 실패율
# ----------------------------------------------------------------------------- #
def scan_infer_failures(results_dir):
    done = err = 0
    for p in sorted(glob.glob(os.path.join(results_dir, "chunk_*.json"))):
        try:
            with open(p) as f:
                rows = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for r in rows:
            done += 1
            if r.get("error") or not r.get("output", "").strip():
                err += 1
    rate = 100.0 * err / done if done else 0.0
    return done, err, rate


# ----------------------------------------------------------------------------- #
# 채점 래퍼
# ----------------------------------------------------------------------------- #
def run_scoring(results_dir, test_json, eval_dir, label, testset, quiet=False):
    cmd = [sys.executable, EVAL_QWEN3OMNI_PY,
           "--results_dir", results_dir, "--test_json", test_json,
           "--eval_dir", eval_dir, "--label", label, "--testset", testset]
    try:
        if quiet:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] 채점 실패(무시하고 계속 진행): {e}")
        return False


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--test_json",
                    default="/workspace/data/test/unav100_qwen3omni.json")
    ap.add_argument("--out_dir",
                    default="/workspace/outputs/base/Qwen3Omni/unav100_qwen3omni")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_audio_in_video", type=int, default=0,
                    help="0=명시적 .wav 항목 사용(기본), 1=mp4 트랙을 시간정렬 투입")
    ap.add_argument("--label", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--testset", default=None)
    ap.add_argument("--no_score", action="store_true")
    args = ap.parse_args()

    testset = args.testset or os.path.splitext(os.path.basename(args.test_json))[0]
    use_audio_in_video = bool(args.use_audio_in_video)

    inputs_dir = os.path.join(args.out_dir, "inputs")
    results_dir = os.path.join(args.out_dir, "results")
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.test_json) as f:
        data = json.load(f)
    n_chunks = split_chunks(data, args.chunk_size, inputs_dir)
    print(f"[infer_qwen3omni] {len(data)} samples -> {n_chunks} chunks "
          f"of {args.chunk_size} (inputs: {inputs_dir})")
    print(f"[infer_qwen3omni] use_audio_in_video={use_audio_in_video} "
          f"({'mp4 트랙' if use_audio_in_video else '명시적 .wav'})")

    import torch
    from transformers import (Qwen3OmniMoeForConditionalGeneration,
                              Qwen3OmniMoeProcessor)
    from qwen_omni_utils import process_mm_info

    print(f"[infer_qwen3omni] loading {args.model_path} "
          f"(flash_attention_2, disable_talker)")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.disable_talker()
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print("[infer_qwen3omni] model ready.")

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
        for idx, sample in enumerate(chunk_samples):
            out = run_one(sample, model, processor, use_audio_in_video,
                          args.max_new_tokens, torch, process_mm_info)
            if out.get("error") or not out.get("output", "").strip():
                n_fail += 1
            chunk_results.append(out)
            if (idx + 1) % 20 == 0:
                print(f"[chunk {i + 1}/{n_chunks}] {idx + 1}/{len(chunk_samples)} "
                      f"done ({n_fail} fail)", flush=True)

        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(chunk_results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, result_path)

        done, err, rate = scan_infer_failures(results_dir)
        print(f"[chunk {i + 1}/{n_chunks}] done: {len(chunk_results)} samples "
              f"({n_fail} fail) in {time.time() - t0:.1f}s -> {result_path}")
        print(f"[infer_qwen3omni] 누적 추론실패율(infer-fail): "
              f"{err}/{done} = {rate:.2f}%")

        if not args.no_score:
            run_scoring(results_dir, args.test_json, eval_dir, args.label,
                        testset, quiet=True)

    done, err, rate = scan_infer_failures(results_dir)
    print(f"\n[infer_qwen3omni] 추론 ALL DONE -> {results_dir}")
    print(f"[infer_qwen3omni] 최종 추론실패율(infer-fail): {err}/{done} = {rate:.2f}%")

    if not args.no_score:
        print("[infer_qwen3omni] 최종 채점 실행...")
        run_scoring(results_dir, args.test_json, eval_dir, args.label,
                    testset, quiet=False)
        print(f"\n[infer_qwen3omni] * 최종 결과 -> "
              f"{os.path.join(eval_dir, 'table.txt')}")
    else:
        print(f"[infer_qwen3omni] --no_score 지정됨. 나중에 수동으로:\n"
              f"  python3 {EVAL_QWEN3OMNI_PY} --results_dir {results_dir} "
              f"--test_json {args.test_json} --eval_dir {eval_dir} "
              f"--label {args.label} --testset {testset}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Qwen3-Omni-30B-A3B-Instruct UnAV-100 추론 — vLLM 백엔드 (offline batched).

infer_qwen3omni.py(HF transformers 백엔드)와 입출력 규약이 완전히 동일하다:
  - 입력  : data/test/unav100_qwen3omni.json  (messages content = [video, audio(.wav), text])
  - 청크  : <out_dir>/inputs/chunk_XXXX.json   (HF 런과 공유 — 같은 내용)
  - 결과  : <out_dir>/results/chunk_XXXX.json  ([{id, question, output}, ...])
  - 채점  : eval_qwen3omni.py 를 그대로 호출 (--natural, parse-fail rate 리포트 포함)
따라서 HF 런이 만들어 둔 청크/결과와 resume 호환된다. (엔진 일관성을 위해선
results/ 를 비우고 vLLM 으로 전체를 다시 도는 걸 권장 — run_qwen3omni_vllm.sh 참고.)

왜 vLLM:
  HF transformers 의 MoE 추론은 1샘플씩 + GPU가 CPU 전처리를 기다려 매우 느리다
  (~7s/sample, GPU util <20%). vLLM 은 연속배칭(continuous batching)으로 여러 시퀀스를
  동시에 굴리고 프리필/디코드를 파이프라인해 GPU를 포화시킨다. Qwen3-Omni README 도
  대량 추론엔 vLLM 을 명시적으로 권장.

구조:
  - LLM(...) 1회 로드 (gpu_memory_utilization 높게, tp=1, max_model_len=32768).
  - 청크(기본 500)별로:
      * build_batch(기본 128) 단위로 process_mm_info 를 스레드풀로 병렬 디코드
        (decord/soundfile 는 GIL 을 상당부분 해제) → vLLM 입력 dict 구성
      * llm.generate(batch_inputs, sampling_params) — vLLM 내부에서 max_num_seqs 만큼
        동시 처리
      * 결과를 청크 리스트에 누적, 청크 끝나면 원자적 저장
  - 실패(빈 출력/예외) 샘플은 {"output": "", "error": ...} 로 남기고 계속.
  - 매 청크 후 채점(eval_qwen3omni.py) → eval/table.txt 갱신 + 파싱실패율 출력.

오디오 (--use_audio_in_video):
  0 (기본) : messages 의 명시적 .wav 를 audio 모달리티로 투입 (use_audio_in_video=False).
  1        : audio 항목 제거하고 mp4 트랙을 mm_processor_kwargs 로 시간정렬 투입.

사용 (보통 run_qwen3omni_vllm.sh 로 실행):
  python3 infer_qwen3omni_vllm.py \
    --model_path /workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct \
    --test_json  /workspace/data/test/unav100_qwen3omni.json \
    --out_dir    /workspace/outputs/base/Qwen3Omni/unav100_qwen3omni_vllm \
    --chunk_size 500
"""
import os
import sys
import sysconfig

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

# 이 인터프리터의 env bin 을 PATH 최상단에 (conda activate 없이 절대경로 python 으로
# 실행돼도 flashinfer JIT 가 'ninja' 를 찾을 수 있도록).
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

# --- Blackwell(sm_120) + vLLM/flashinfer JIT 우회 -------------------------------
#  flashinfer 는 CUDA_HOME 의 nvcc 로 CUDA 버전을 판별한다. 이 서버의
#  /usr/local/cuda 는 12.8 이고, flashinfer 는 SM 12.x 에 CUDA>=12.9 를 요구하므로
#  "No supported CUDA architectures found for major versions [12]" 로 엔진 로드가 죽는다.
#  pip 로 설치된 nvidia/cu13 (nvcc 13.3) 을 CUDA_HOME 으로 지정하면 해결된다.
_CU13 = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cu13")
if os.path.isfile(os.path.join(_CU13, "bin", "nvcc")):
    os.environ.setdefault("CUDA_HOME", _CU13)
    os.environ.setdefault("CUDA_PATH", _CU13)
    os.environ["PATH"] = os.path.join(_CU13, "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
os.environ.setdefault("FLASHINFER_NVCC_THREADS", "8")
# pip 의 nvidia CUDA wheel 세트가 내부 불일치(nvcc 13.3 vs cudart 13.0)라 cccl 의
# compiler/toolkit 호환성 검사가 실패한다. 해당 검사만 끈다(minor 차이라 안전).
os.environ.setdefault("CCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "1")
os.environ.setdefault("NVCC_APPEND_FLAGS", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK")

import sys
import glob
import json
import time
import copy
import argparse
import traceback
import subprocess
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_QWEN3OMNI_PY = os.path.join(SCRIPT_DIR, "eval_qwen3omni.py")


# ----------------------------------------------------------------------------- #
# 청크 분할 (infer_qwen3omni.py 와 동일)
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
# 멀티모달 전처리 → vLLM 입력 dict
# ----------------------------------------------------------------------------- #
def _extract_question(sample) -> str:
    for it in sample["messages"][0]["content"]:
        if it.get("type") == "text":
            return it["text"]
    return sample.get("question", "")


def build_vllm_input(sample, processor, process_mm_info, use_audio_in_video):
    """(input_dict | None, question, err) 반환. None 이면 전처리 실패."""
    question = _extract_question(sample)
    try:
        messages = copy.deepcopy(sample["messages"])
        if use_audio_in_video:
            messages[0]["content"] = [
                it for it in messages[0]["content"] if it.get("type") != "audio"
            ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video)

        mm = {}
        if images is not None:
            mm["image"] = images
        if videos is not None:
            mm["video"] = videos
        if audios is not None:
            mm["audio"] = audios

        inp = {
            "prompt": text,
            "multi_modal_data": mm,
            "mm_processor_kwargs": {"use_audio_in_video": use_audio_in_video},
        }
        return inp, question, None
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return None, question, str(e)


# ----------------------------------------------------------------------------- #
# 누적 추론 실패율 (infer_qwen3omni.py 와 동일)
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
                    default="/workspace/outputs/base/Qwen3Omni/unav100_qwen3omni_vllm")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--build_batch", type=int, default=128,
                    help="한 번에 전처리+generate 로 넘길 샘플 수 (RAM/스케줄링 균형)")
    ap.add_argument("--decode_workers", type=int, default=8,
                    help="process_mm_info 병렬 디코드 스레드 수")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0=greedy (grounding 평가 재현성). README 기본은 0.6.")
    ap.add_argument("--use_audio_in_video", type=int, default=0)
    # vLLM 엔진 파라미터
    ap.add_argument("--gpu_mem_util", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=32768)
    ap.add_argument("--max_num_seqs", type=int, default=8)
    ap.add_argument("--moe_backend", default="triton",
                    help="'triton'=nvcc JIT 불필요(Blackwell+CUDA13 셋업에서 안전). "
                         "'auto'=flashinfer cutlass(빠르나 sm120 JIT 이슈 가능).")
    ap.add_argument("--enforce_eager", type=int, default=1,
                    help="1=torch.compile/CUDA graph 캡처 생략(첫 로드 빠름, 안정). "
                         "0=컴파일(더 빠른 정상상태, 첫 로드 김).")
    ap.add_argument("--limit_mm_video", type=int, default=1)
    ap.add_argument("--limit_mm_audio", type=int, default=1)
    ap.add_argument("--limit_mm_image", type=int, default=1)
    ap.add_argument("--label", default="Qwen3-Omni-30B-A3B-Instruct-vllm")
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
    print(f"[vllm] {len(data)} samples -> {n_chunks} chunks of {args.chunk_size}")
    print(f"[vllm] use_audio_in_video={use_audio_in_video} "
          f"({'mp4 트랙' if use_audio_in_video else '명시적 .wav'})  "
          f"greedy={args.temperature == 0.0}  max_new_tokens={args.max_new_tokens}")

    import torch
    from vllm import LLM, SamplingParams
    from transformers import Qwen3OmniMoeProcessor
    from qwen_omni_utils import process_mm_info

    print(f"[vllm] loading engine: {args.model_path} "
          f"(moe_backend={args.moe_backend}, enforce_eager={bool(args.enforce_eager)})")
    llm_kwargs = dict(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=torch.cuda.device_count(),
        limit_mm_per_prompt={
            "image": args.limit_mm_image,
            "video": args.limit_mm_video,
            "audio": args.limit_mm_audio,
        },
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=bool(args.enforce_eager),
        seed=1234,
    )
    if args.moe_backend and args.moe_backend != "auto":
        llm_kwargs["kernel_config"] = {"moe_backend": args.moe_backend}
    llm = LLM(**llm_kwargs)
    sp_kwargs = dict(max_tokens=args.max_new_tokens)
    if args.temperature and args.temperature > 0:
        sp_kwargs.update(temperature=args.temperature, top_p=0.95, top_k=20)
    else:
        sp_kwargs.update(temperature=0.0)
    sampling_params = SamplingParams(**sp_kwargs)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)
    print("[vllm] engine ready.")

    for i in range(n_chunks):
        chunk_path = os.path.join(inputs_dir, f"chunk_{i:04d}.json")
        result_path = os.path.join(results_dir, f"chunk_{i:04d}.json")
        if os.path.exists(result_path):
            print(f"[chunk {i + 1}/{n_chunks}] skip (already done)")
            continue
        with open(chunk_path) as f:
            chunk_samples = json.load(f)

        t0 = time.time()
        chunk_results = []
        n_fail = 0
        with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
            for b0 in range(0, len(chunk_samples), args.build_batch):
                batch = chunk_samples[b0:b0 + args.build_batch]
                built = list(pool.map(
                    lambda s: build_vllm_input(
                        s, processor, process_mm_info, use_audio_in_video),
                    batch))

                gen_inputs, gen_meta = [], []
                for sample, (inp, question, err) in zip(batch, built):
                    if inp is None:
                        chunk_results.append({"id": sample["id"], "question": question,
                                              "output": "", "error": err})
                        n_fail += 1
                    else:
                        gen_inputs.append(inp)
                        gen_meta.append((sample["id"], question))

                if gen_inputs:
                    try:
                        outs = llm.generate(gen_inputs, sampling_params=sampling_params)
                    except Exception as e:  # noqa: BLE001
                        traceback.print_exc()
                        for sid, q in gen_meta:
                            chunk_results.append({"id": sid, "question": q,
                                                  "output": "", "error": str(e)})
                            n_fail += 1
                        continue
                    for (sid, q), o in zip(gen_meta, outs):
                        txt = o.outputs[0].text.strip() if o.outputs else ""
                        rec = {"id": sid, "question": q, "output": txt}
                        if not txt:
                            rec["error"] = "empty_output"
                            n_fail += 1
                        chunk_results.append(rec)

                print(f"[chunk {i + 1}/{n_chunks}] {min(b0 + args.build_batch, len(chunk_samples))}"
                      f"/{len(chunk_samples)} built+gen ({n_fail} fail)", flush=True)

        # id 순서를 원래 청크 순서에 맞춤
        order = {s["id"]: k for k, s in enumerate(chunk_samples)}
        chunk_results.sort(key=lambda r: order.get(r["id"], 1 << 30))

        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(chunk_results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, result_path)

        done, err, rate = scan_infer_failures(results_dir)
        dt = time.time() - t0
        print(f"[chunk {i + 1}/{n_chunks}] done: {len(chunk_results)} samples "
              f"({n_fail} fail) in {dt:.1f}s ({dt / max(len(chunk_samples), 1):.2f}s/sample)")
        print(f"[vllm] 누적 추론실패율(infer-fail): {err}/{done} = {rate:.2f}%")

        if not args.no_score:
            run_scoring(results_dir, args.test_json, eval_dir, args.label,
                        testset, quiet=True)

    done, err, rate = scan_infer_failures(results_dir)
    print(f"\n[vllm] 추론 ALL DONE -> {results_dir}")
    print(f"[vllm] 최종 추론실패율(infer-fail): {err}/{done} = {rate:.2f}%")

    if not args.no_score:
        print("[vllm] 최종 채점 실행...")
        run_scoring(results_dir, args.test_json, eval_dir, args.label, testset, quiet=False)
        print(f"\n[vllm] * 최종 결과 -> {os.path.join(eval_dir, 'table.txt')}")
    else:
        print(f"[vllm] --no_score. 나중에 수동 채점:\n"
              f"  python3 {EVAL_QWEN3OMNI_PY} --results_dir {results_dir} "
              f"--test_json {args.test_json} --eval_dir {eval_dir} "
              f"--label {args.label} --testset {testset}")


if __name__ == "__main__":
    main()

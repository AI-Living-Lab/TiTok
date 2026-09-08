#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pv_llm.py — 로컬 Qwen2.5-7B-Instruct 로 raw 출력에서 시간구간을 뽑는 독립 파서.

룰기반 파서(eval_miou.py)의 정규식·보정·merge 로직을 일절 참조하지 않는다.
지시는 "이 텍스트가 주장하는 시간 구간을 모두 초 단위로 추출하라" 하나뿐이고,
출력은 {"segments": [[s,e], ...]} JSON 으로만 받는다.

  - temperature=0 (greedy), seed 고정 → 재현 가능
  - 배치 추론 (left padding + generate)
  - JSON 파싱 실패 시 1회 재시도(더 강한 지시 + 소폭 긴 예산). 실패해도 샘플은
    버리지 않고 status='json_error' 로 기록
  - per_sample.jsonl 캐시 → 중단돼도 이어서 실행

GPU 메모리: 이 서버는 A100 1대를 학습 작업들과 공유하므로 4-bit NF4 로 로드한다
(가중치 ~5GB). --dtype bf16 으로 전환 가능하지만 여유 VRAM 이 16GB 아래면 위험.
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pv_data import WS, MODELS_ALL, SHORT, load_model_samples, pilot_subset  # noqa: E402

HF_MODEL_DEFAULT = "Qwen/Qwen2.5-14B-Instruct"
SEED = 1234
OUT_ROOT = os.path.join(WS, "results", "parser_verification")
CACHE_DIR = os.path.join(OUT_ROOT, "cache")

SYSTEM = (
    "You extract time intervals from text.\n"
    "The text is one model's answer to a question about when an event happens in a video.\n"
    "Report every time interval that the text asserts the event occurs in.\n"
    "\n"
    "Rules:\n"
    "- Times may be written in any notation. Convert everything to seconds.\n"
    "- If the text contains reasoning followed by a final answer, report the intervals the "
    "text ultimately asserts, not ones it considered and rejected.\n"
    "- If the text asserts no interval at all (it refuses, says the event is absent, or never "
    "commits to a time range), report an empty list.\n"
    "- Never invent an interval the text does not state.\n"
    "- Report intervals as written: do not merge, split, reorder, or clip them.\n"
    "- Output one entry per interval the text states, in the order stated. Keep an interval "
    "even if it repeats one already listed, and keep adjacent or touching intervals separate "
    "(0-10 followed by 10-18 is two intervals, not one).\n"
    "\n"
    'Reply with JSON and nothing else, exactly: {"segments": [[start, end], ...]}\n'
    "start and end are numbers in seconds, with at most 2 decimal places. "
    "Use [] if there are none.\n"
    "\n"
    "Examples of a correct reply:\n"
    '{"segments": [[0.0, 12.5], [30.0, 41.2]]}\n'
    '{"segments": []}'
)
RETRY_SUFFIX = (
    "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object, "
    'no explanation, no code fence. Example: {"segments": [[1.5, 4.0]]}'
)

USER_SECONDS = "Text:\n<<<\n{text}\n>>>"
# AVicuna 는 0~99 로 정규화된 시간을 쓴다. 산술은 LLM 에게 맡기지 않고
# 적힌 숫자를 그대로 받아 pv_data.postprocess_llm 이 pos/100*duration 으로 환산한다.
USER_AS_WRITTEN = (
    "In this text the times are written on a 0-to-99 scale relative to the whole video, "
    "not in seconds. Report the numbers exactly as they are written. Do NOT convert them "
    "to seconds and do NOT do any arithmetic.\n\nText:\n<<<\n{text}\n>>>"
)


def build_user(sample):
    if sample["unit"] == "percent_as_written":
        return USER_AS_WRITTEN.format(text=sample["llm_text"])
    return USER_SECONDS.format(text=sample["llm_text"])


# ------------------------------ 응답 파싱 ------------------------------
def coerce_segments(obj):
    if not isinstance(obj, dict) or "segments" not in obj:
        raise ValueError("no 'segments' key")
    segs = obj["segments"]
    if not isinstance(segs, list):
        raise ValueError("'segments' is not a list")
    out = []
    for it in segs:
        if not isinstance(it, (list, tuple)) or len(it) != 2:
            raise ValueError(f"bad segment {it!r}")
        out.append([float(it[0]), float(it[1])])
    return out


_LEADING_ZERO = re.compile(r"(?<![\w.])0+(?=\d)")


def parse_response(text):
    """엄격 파싱 우선 → 실패 시 첫 JSON 객체만 떼어 재시도. (fallback_used, segments)"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    t = _LEADING_ZERO.sub("", t)   # [[00, 99]] → [[0, 99]] (0.5 는 건드리지 않음)
    try:
        return False, coerce_segments(json.loads(t))
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        return True, coerce_segments(json.loads(m.group(0)))
    raise ValueError("no JSON object in response")


# ------------------------------ 캐시 ------------------------------
def cache_path(model):
    return os.path.join(CACHE_DIR, f"{model}.jsonl")


def load_cache(model):
    p, done = cache_path(model), {}
    if not os.path.exists(p):
        return done
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                       # 중단으로 잘린 마지막 줄
            done[r["key"]] = r                 # 실패도 기록으로 남기되
    return {k: v for k, v in done.items() if v.get("status") == "ok"}   # 재실행 대상은 성공만 skip


# ------------------------------ 추론 ------------------------------
class Runner:
    def __init__(self, dtype="4bit", max_new_tokens=384, hf_model=HF_MODEL_DEFAULT):
        # 이 env 에는 deepspeed 가 깔려 있어 transformers 가 import 할 때 CUDA_HOME 을 요구한다.
        # nvcc 가 env 안에 있으므로 없으면 채워 준다 (이 프로세스에만 적용).
        if not os.environ.get("CUDA_HOME"):
            cand = os.path.dirname(os.path.dirname(sys.executable))
            if os.path.exists(os.path.join(cand, "bin/nvcc")):
                os.environ["CUDA_HOME"] = cand
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(SEED)
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.hf_model = hf_model

        self.tok = AutoTokenizer.from_pretrained(hf_model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda:0"}  # transformers 4.51 은 torch_dtype
        if dtype == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        print(f"[load] {hf_model}  dtype={dtype}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model, **kw).eval()
        free, total = torch.cuda.mem_get_info()
        print(f"[load] 완료. GPU 여유 {free/2**30:.1f}GB / {total/2**30:.1f}GB", flush=True)

    def _prompt(self, sample, retry=False):
        msgs = [{"role": "system", "content": SYSTEM + (RETRY_SUFFIX if retry else "")},
                {"role": "user", "content": build_user(sample)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, samples, retry=False):
        prompts = [self._prompt(s, retry) for s in samples]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=3072).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


def run_model(runner, model, samples, batch_size):
    os.makedirs(CACHE_DIR, exist_ok=True)
    done = load_cache(model)
    todo = [s for s in samples if s["key"] not in done]
    print(f"[{SHORT[model]}] 대상 {len(samples)}  캐시적중 {len(done)}  추론 {len(todo)}", flush=True)
    if not todo:
        return done

    # 긴 입력이 짧은 입력과 한 배치에 섞이면 패딩 낭비 → 길이순 정렬
    todo.sort(key=lambda s: len(s["llm_text"]))
    fh = open(cache_path(model), "a")
    t0, n = time.time(), 0

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        texts = runner.generate(chunk)
        recs, retry_idx = [], []
        for j, (s, txt) in enumerate(zip(chunk, texts)):
            rec = {"key": s["key"], "idx": s["idx"], "model": model, "segments": [],
                   "status": "ok", "attempts": 1, "json_fallback": False,
                   "error": None, "response": txt[:2000]}
            try:
                fb, segs = parse_response(txt)
                rec["segments"], rec["json_fallback"] = segs, fb
            except Exception as e:
                rec["status"], rec["error"] = "json_error", f"{type(e).__name__}: {e}"
                rec["first_response"] = txt[:2000]
                retry_idx.append(j)
            recs.append(rec)

        if retry_idx:                                   # JSON 실패분 1회 재시도
            rs = [chunk[j] for j in retry_idx]
            rtexts = runner.generate(rs, retry=True)
            for j, txt in zip(retry_idx, rtexts):
                rec = recs[j]
                rec["attempts"] = 2
                rec["response"] = txt[:2000]
                try:
                    fb, segs = parse_response(txt)
                    rec["segments"], rec["json_fallback"] = segs, fb
                    rec["status"], rec["error"] = "ok", None
                except Exception as e:
                    rec["status"], rec["error"] = "json_error", f"{type(e).__name__}: {e}"

        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[rec["key"]] = rec
        fh.flush()
        n += len(chunk)
        rate = n / max(time.time() - t0, 1e-9)
        print(f"  [{SHORT[model]}] {n}/{len(todo)}  {rate:.1f} samp/s  "
              f"retry {len(retry_idx)}", flush=True)

    fh.close()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="모델당 N개 랜덤 샘플")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", choices=["4bit", "bf16"], default="bf16")
    ap.add_argument("--hf-model", default=HF_MODEL_DEFAULT)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--models", nargs="*", default=MODELS_ALL)
    args = ap.parse_args()
    if not args.pilot and not args.full:
        ap.error("--pilot N 또는 --full 중 하나가 필요합니다")

    runner = Runner(dtype=args.dtype, max_new_tokens=args.max_new_tokens,
                    hf_model=args.hf_model)
    for m in args.models:
        ss = load_model_samples(m)
        if args.pilot:
            ss = pilot_subset(ss, args.pilot)
        done = run_model(runner, m, ss, args.batch_size)
        bad = [r for r in done.values() if r["status"] != "ok"]
        print(f"[{SHORT[m]}] 완료 {len(done)}  JSON 실패 {len(bad)}", flush=True)


if __name__ == "__main__":
    main()

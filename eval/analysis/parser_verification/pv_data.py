#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pv_data.py — 파서 검증용 데이터 로더.

리뷰어 대응 실험의 입력 정의를 한 곳에 모은다.
핵심 결정: LLM 파서에는 **래퍼 전처리 이전의 진짜 raw generation** 을 먹인다.
  - MUSEG   : results/chunk_*.json 의 `raw` (<think> 포함 원문). eval 의 pred 는
              래퍼가 <answer> 를 뽑은 결과라 127건이 빈 문자열이다.
  - AVicuna : `raw_pred` (0~99 정규화 시간) + `duration`. eval 의 pred 는 래퍼가
              이미 초로 환산해 둔 값.
  - 나머지 3종 : pred 가 곧 모델 원문이라 그대로 사용.

룰기반 파서(R)는 항상 채점기가 실제로 본 텍스트(eval 의 pred)를 파싱한다 —
즉 논문 수치를 만든 경로 그대로.
"""
import glob
import json
import os
import random
import re
import sys

# .../Team4/eval/analysis/parser_verification/pv_data.py -> 5단계 위가 workspace
WS = os.environ.get("WORKSPACE",
     os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
         os.path.dirname(os.path.abspath(__file__)))))))  # workspace
EVAL_DIR = os.path.join(WS, "Team4/eval")
BASE = os.path.join(WS, "outputs/base")
sys.path.insert(0, EVAL_DIR)

import eval_miou as EM  # noqa: E402

MAX_T = EM.MAX_TIME_DEFAULT

MODELS = ["ChronusOmni", "AVicuna", "ARC-Hunyuan-Video-7B", "Qwen2.5-Omni", "MUSEG"]

# TiTok(ours). baseline 과 두 가지가 다르다:
#   1) pred 가 전용 시간토큰(<t0><t9><tdot><t2> = 9.2초) → 채점기가 parse_tokens 사용(natural=False)
#   2) GT 가 gt_segments 가 아니라 ref 에 같은 토큰으로 들어있음 (gt_source='ref(auto)')
TITOK = "TiTok"
TITOK_EVAL = os.path.join(
    WS, "outputs/gdpo/sft_7b_unav_v8_rl_rMsep3_unpucha_batch4_noscaling",
    "checkpoint-2000/fps5_tti/unav100_titok__BUGGY_audiolen_dropped/test_results_rank0.json")
MODELS_ALL = MODELS + [TITOK]

# 표기용 짧은 이름
SHORT = {
    "ChronusOmni": "ChronusOmni",
    "AVicuna": "AVicuna",
    "ARC-Hunyuan-Video-7B": "ARC-Hunyuan-7B",
    "Qwen2.5-Omni": "Qwen2.5-Omni",
    "MUSEG": "MUSEG",
    "TiTok": "TiTok(ours)",
}


# ---- TiTok 시간토큰 디토크나이저 -------------------------------------------
# 고정 인코딩(파싱 휴리스틱이 아님): <t0>..<t9> 는 숫자, <tdot> 는 소수점.
#   <t0><t0><t9><tdot><t2> -> 9.2   /   <t0><t4><t3><tdot><t3> -> 43.3
# 토큰을 평문 숫자로만 치환하고 문장 구조는 건드리지 않는다 → LLM 은 baseline 과
# 완전히 같은 종류의 일(평문에서 구간 찾기)만 하게 된다.
_TOK_RUN = re.compile(r"(?:<t\d>|<tdot>)+")
_TOK_DIGITS = re.compile(r"<t(\d)>")


def detokenize_time_tokens(text):
    def rep(m):
        whole = m.group(0)
        ip, _, dp = whole.partition("<tdot>")
        i = "".join(_TOK_DIGITS.findall(ip))
        d = "".join(_TOK_DIGITS.findall(dp))
        if not i and not d:
            return whole
        head = str(int(i)) if i else "0"
        return f"{head}.{d}" if d else head
    return _TOK_RUN.sub(rep, text or "")


def postprocess_llm(sample, segs):
    """LLM 이 '적힌 그대로' 낸 숫자에 모델별 단위 환산을 적용한다.

    AVicuna 는 0~99 정규화 시간을 쓰므로 파이프라인과 동일하게 pos/100*duration.
    (eval_miou._scale_percent 와 같은 상수 — LLM 에게 산술을 맡기지 않는다.)
    """
    if sample["unit"] == "percent_as_written" and sample["duration"]:
        d = sample["duration"]
        return [[a / 100.0 * d, b / 100.0 * d] for a, b in segs]
    return [[float(a), float(b)] for a, b in segs]


def _eval_path(model):
    if model == TITOK:
        return TITOK_EVAL
    return os.path.join(BASE, model, "unav100_multiseg/eval/test_results_rank0.json")


def _raw_chunks(model):
    return sorted(glob.glob(os.path.join(BASE, model, "unav100_multiseg/results/chunk_*.json")))


def _load_museg_raw():
    """MUSEG 원문(raw)을 id → text 로."""
    out = {}
    for f in _raw_chunks("MUSEG"):
        for r in json.load(open(f)):
            out[r["id"]] = str(r.get("raw", "") or "")
    return out


def load_model_samples(model):
    """샘플 리스트 반환.

    각 항목:
      key        : 샘플 식별자 (id 가 없는 모델은 'idx:<n>')
      gt         : [[s,e], ...]
      scorer_text: 채점기가 본 텍스트 (룰기반 파서 R 의 입력)
      llm_text   : LLM 파서 L 의 입력 (진짜 raw generation)
      duration   : 있으면 float, 없으면 None
      unit       : 'seconds' | 'percent_0_99'  — llm_text 의 시간 단위 규약
    """
    rows = json.load(open(_eval_path(model)))
    museg_raw = _load_museg_raw() if model == "MUSEG" else None

    samples = []
    for i, r in enumerate(rows):
        key = r.get("id") or f"idx:{i}"
        scorer_text = str(r.get("pred", "") or "")

        if model == TITOK:
            # 토큰 -> 평문 숫자. LLM 은 baseline 과 동일한 평문 추출만 한다.
            llm_text, unit, dur = detokenize_time_tokens(scorer_text), "seconds", None
        elif model == "MUSEG":
            llm_text, unit, dur = museg_raw.get(r["id"], ""), "seconds", None
        elif model == "AVicuna":
            # 0~99 정규화 값을 '적힌 그대로' 뽑게 하고 환산은 postprocess_llm 이 한다.
            llm_text = str(r.get("raw_pred", "") or "")
            unit = "percent_as_written"
            dur = float(r["duration"]) if r.get("duration") else None
        else:
            llm_text, unit, dur = scorer_text, "seconds", None

        samples.append({
            "key": key,
            "idx": i,
            "model": model,
            "gt": (EM.parse_tokens(str(r.get("ref", "")), MAX_T) if model == TITOK
                   else EM._coerce_segments(r.get("gt_segments"))),
            "gt_label": r.get("gt_label", ""),
            "scorer_text": scorer_text,
            "llm_text": llm_text,
            "duration": dur,
            "unit": unit,
        })
    return samples


def rule_parse(sample):
    """룰기반 파서 R — 채점기와 완전히 동일한 경로.

    TiTok 은 채점기가 natural=False 로 돌았으므로 parse_tokens 를 쓴다.
    """
    scope = EM.extract_answer_scope(sample["scorer_text"])
    if sample["model"] == TITOK:
        return EM.parse_tokens(scope, MAX_T)
    return EM.parse_natural(scope, MAX_T)


def pilot_subset(samples, n=100, seed=42):
    rnd = random.Random(f"{seed}:{samples[0]['model']}" if samples else seed)
    if len(samples) <= n:
        return list(samples)
    return sorted(rnd.sample(samples, n), key=lambda s: s["idx"])


if __name__ == "__main__":
    for m in MODELS_ALL:
        ss = load_model_samples(m)
        empty_scorer = sum(1 for s in ss if not s["scorer_text"].strip())
        empty_llm = sum(1 for s in ss if not s["llm_text"].strip())
        avg = sum(len(s["llm_text"]) for s in ss) / len(ss)
        print(f"{SHORT[m]:<16} n={len(ss):<6} unit={ss[0]['unit']:<14} "
              f"빈 scorer_text={empty_scorer:<4} 빈 llm_text={empty_llm:<4} "
              f"llm_text 평균 {avg:.0f}자")

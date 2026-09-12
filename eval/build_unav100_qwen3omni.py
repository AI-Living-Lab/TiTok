#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_unav100_qwen3omni.py — unav100_chronus.json 를 Qwen3-Omni 네이티브 입력 포맷으로 변환.

다른 비교모델(avicuna/museg/chronus/arc/titok)과 동일하게 소스는
data/test/unav100_chronus.json (3455 샘플, id/gt_label/gt_segments/video/audio/question)
이고 여기서 바뀌는 건 "question"의 Answer format 블록뿐이다 (COMPARISON_MODELS_HANDOFF.md 참고).

Qwen3-Omni는 자체 conversation 포맷(role/content list, apply_chat_template)을 쓰므로
question 텍스트를 감싸는 <video> 같은 토큰 플레이스홀더가 필요 없다. 대신 미디어를
content list 안에 별도 항목으로 넣는다:

  content = [
    {"type": "video", "video": <mp4 경로>},
    {"type": "audio", "audio": <wav 경로>},   # ★ UnAV-100 큐레이션된 .wav 를 명시적으로 투입
    {"type": "text",  "text":  <question>},
  ]

왜 별도 audio 항목인가:
  - UnAV-100 은 이벤트별로 잘라둔 .wav(/workspace/datasets/unav_100/audio/*.wav)가 있고,
    평가 취지가 "video 와 audio 양쪽에서" 이벤트가 언제 일어나는지이므로 그 .wav 를
    그대로 모델에 넣는 게 맞다. mp4 컨테이너 오디오 트랙에 의존(use_audio_in_video=True)
    하지 않고 경로를 직접 실어 재현성을 확보한다.
  - Qwen3-Omni 는 한 메시지 안에 video + audio 를 동시에 받는 패턴을 공식 지원한다
    (README "Best Practices for the Thinking Model" 의 audio+image+video 예시,
     conversation4 mixed-media 예시). qwen_omni_utils.process_mm_info 가 이 audio 항목을
     audios 리스트로 뽑아준다.
  - 추론기(infer_qwen3omni.py)는 기본 use_audio_in_video=False 로 이 .wav 를 쓴다.
    --use_audio_in_video 1 로 주면 추론기가 이 audio 항목을 떼고 mp4 트랙을 시간정렬
    (TMRoPE)해서 넣는다. 둘 중 하나만 쓰도록(중복 방지) 추론기가 처리한다.

Answer format 은 chronus 전용 "second{}" 래퍼 대신, eval_miou.py --natural 파서가
그대로 수용하는 평범한 소수초 "start-end" 표기로 바꿔 모델이 자연스럽게 답하도록 유도한다.

출력 각 샘플 스키마:
  {"id", "gt_label", "gt_segments", "video", "audio", "question", "messages": [...]}
messages 는 Qwen3OmniMoeProcessor.apply_chat_template 에 그대로 넣을 수 있는 형식이다.

사용:
  python3 build_unav100_qwen3omni.py \
    --src /workspace/data/test/unav100_chronus.json \
    --dst /workspace/data/test/unav100_qwen3omni.json
"""
import argparse
import json
import os
import re

DEFAULT_SRC = "/workspace/data/test/unav100_chronus.json"
DEFAULT_DST = "/workspace/data/test/unav100_qwen3omni.json"

ANSWER_FORMAT_RE = re.compile(r"\n\nAnswer format:.*", re.DOTALL)


def rewrite_question(question: str) -> str:
    """공통 stem은 유지하고 Answer format 블록만 Qwen3 네이티브(소수초) 표기로 교체."""
    stem = ANSWER_FORMAT_RE.sub("", question).strip()
    return (
        f"{stem}\n\n"
        "Answer format:\n"
        '"start-end" (seconds, decimal allowed, e.g. "12.3-20.66")\n\n'
        "For multiple segments, separate them with a period and space, like:\n"
        '"start-end. start-end. ..."'
    )


def build(sample: dict, strict_paths: bool) -> dict:
    question = rewrite_question(sample["question"])
    video = sample["video"]
    audio = sample.get("audio", "")

    if strict_paths:
        if not os.path.isfile(video):
            raise FileNotFoundError(f"video 없음: {video} (id={sample.get('id')})")
        if not audio:
            raise KeyError(f"audio 필드 없음 (id={sample.get('id')})")
        if not os.path.isfile(audio):
            raise FileNotFoundError(f"audio 없음: {audio} (id={sample.get('id')})")

    content = [{"type": "video", "video": video}]
    if audio:
        content.append({"type": "audio", "audio": audio})
    content.append({"type": "text", "text": question})

    return {
        "id": sample["id"],
        "gt_label": sample.get("gt_label", ""),
        "gt_segments": sample.get("gt_segments", []),
        "video": video,
        "audio": audio,
        "question": question,
        "messages": [{"role": "user", "content": content}],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--no-check", action="store_true",
                    help="video/audio 파일 존재 검사를 건너뛴다 (기본은 검사).")
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)

    strict = not args.no_check
    out = [build(s, strict) for s in data]

    n_audio = sum(1 for x in out if x["audio"])
    tmp = args.dst + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.dst)

    print(f"[build_unav100_qwen3omni] {len(data)} samples -> {args.dst}")
    print(f"[build_unav100_qwen3omni] audio 경로 포함: {n_audio}/{len(out)} "
          f"(누락 {len(out) - n_audio})")
    print("---- sample[0] ----")
    print(json.dumps(out[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

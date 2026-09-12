#!/bin/bash
# ============================================================
# run_qwen3omni_infer.sh
#   Qwen/Qwen3-Omni-30B-A3B-Instruct 를 UnAV-100(멀티세그 프롬프트, Qwen3 네이티브
#   conversation 포맷)에 대해 "추론 + 청크채점 + 최종 table.txt" 까지 한 번에 돌리는
#   실행기. run_chronus_sft_infer.sh 와 동일한 골격.
#
#   사전 준비 (한 번만) — 2026-09-02 기준 아래 3개 전부 완료됨:
#     1) 모델 다운로드 -> /workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct (66G, 15 shard 검증 OK)
#     2) conda env "qwen3omni" = Chronus128 클론 + 아래 추가 설치 완료:
#          pip install -U "transformers>=5.2.0" accelerate "qwen-omni-utils[decord]"
#        => transformers 5.16.1 / accelerate 1.14.0 / qwen-omni-utils 0.0.9,
#           Qwen3OmniMoeForConditionalGeneration import 확인.
#        이 GPU(RTX PRO 6000 Blackwell, sm_120)는 torch가 cu128 이상이어야 커널을 찾는다
#        (클론된 torch 2.8.0+cu128 이 조건 충족). chronusomni env(torch 2.3+cu118)는 안 돈다.
#     3) 테스트셋: build_unav100_qwen3omni.py 로 생성한 data/test/unav100_qwen3omni.json
#        (messages content = [video, audio(.wav), text] 3항목)
#
#   사용법:
#     bash run_qwen3omni_infer.sh
#     CHUNK_SIZE=250 bash run_qwen3omni_infer.sh          # 메모리 이슈 시 청크를 잘게
#     bash run_qwen3omni_infer.sh --no_score              # 채점 생략, 추론만
#     USE_AUDIO_IN_VIDEO=1 bash run_qwen3omni_infer.sh    # .wav 대신 mp4 트랙을 시간정렬 투입
#
#   재실행하면 이미 끝난 청크는 자동 skip(resume).
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct}"
TEST_JSON="${TEST_JSON:-/workspace/data/test/unav100_qwen3omni.json}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/base/Qwen3Omni/unav100_qwen3omni}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
USE_AUDIO_IN_VIDEO="${USE_AUDIO_IN_VIDEO:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

mkdir -p "$OUT_DIR"

[ -f /workspace/setup.sh ] && source /workspace/setup.sh
if ! declare -F conda >/dev/null 2>&1; then
    for _c in "$CONDA_HOME" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
    done
fi
conda activate qwen3omni

# python3 를 PATH 검색 대신 env 안의 절대경로로 직접 호출
# (run_chronus_sft_infer.sh 에서 겪은 것과 동일한 함정 회피: 오래 산 셸에서
#  setup.sh 를 여러 번 source 하면 PATH 맨 앞이 base conda 로 덮여써질 수 있음)
QWEN3OMNI_PY="${CONDA_HOME:-/workspace/home/miniconda3}/envs/qwen3omni/bin/python3"
if [ ! -x "$QWEN3OMNI_PY" ]; then
    echo "[에러] qwen3omni python 바이너리를 못 찾음: $QWEN3OMNI_PY"
    exit 1
fi
echo "[python] $QWEN3OMNI_PY ($("$QWEN3OMNI_PY" -c 'import torch,sys;print("torch",torch.__version__,"@",sys.executable)'))"

if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo "[에러] 모델 경로가 비어있음: $MODEL_PATH (다운로드가 아직 안 끝났을 수 있음)"
    exit 1
fi
if [ ! -f "$TEST_JSON" ]; then
    echo "[에러] 테스트셋이 없음: $TEST_JSON (build_unav100_qwen3omni.py 먼저 실행)"
    exit 1
fi

echo "=================================================="
echo "  Qwen3-Omni-30B-A3B-Instruct UnAV-100 추론 + 청크채점"
echo "  MODEL_PATH : $MODEL_PATH"
echo "  TEST_JSON  : $TEST_JSON"
echo "  OUT_DIR    : $OUT_DIR"
echo "  CHUNK_SIZE : $CHUNK_SIZE"
echo "  GPU        : $CUDA_VISIBLE_DEVICES"
echo "=================================================="

"$QWEN3OMNI_PY" "$SCRIPT_DIR/infer_qwen3omni.py" \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --out_dir "$OUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --use_audio_in_video "$USE_AUDIO_IN_VIDEO" \
    "$@" 2>&1 | tee -a "$OUT_DIR/inference.log"

echo ""
echo "[완료] 최종 결과: $OUT_DIR/eval/table.txt"
[ -f "$OUT_DIR/eval/table.txt" ] && cat "$OUT_DIR/eval/table.txt"

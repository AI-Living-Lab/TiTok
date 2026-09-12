#!/bin/bash
# ============================================================
# run_qwen3omni_vllm.sh
#   Qwen/Qwen3-Omni-30B-A3B-Instruct 를 UnAV-100 에 대해 vLLM(offline batched)로
#   "추론 + 청크채점 + 최종 table.txt" 까지 한 번에. run_qwen3omni_infer.sh(HF판)의
#   vLLM 버전. HF판 대비 GPU 포화로 수 배 빠름.
#
#   사전 준비 (한 번만):
#     1) 모델: /workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct
#     2) conda env "qwen3vllm" = python 3.11 새 env + 아래 설치:
#          pip install vllm "qwen-omni-utils[decord]"
#        (README 권장: vLLM 은 런타임 충돌 회피 위해 반드시 새 env)
#        이 GPU(RTX PRO 6000 Blackwell, sm_120)는 vllm 이 딸려오는 torch 가 cu128 빌드
#        여야 커널을 찾는다. 최신 vllm(0.11+, torch 2.8+cu128)이면 충족.
#     3) 테스트셋: data/test/unav100_qwen3omni.json (build_unav100_qwen3omni.py 로 생성)
#
#   ★ GPU 는 1장뿐이라 HF판(run_qwen3omni_infer.sh)과 동시에 못 돈다.
#     vLLM 은 gpu_memory_utilization 만큼(기본 0.92 ≈ 90GB) 선점하므로,
#     HF 런을 먼저 멈춰야 한다:  tmux kill-session -t qwen3omni
#
#   사용법:
#     bash run_qwen3omni_vllm.sh
#     GPU_MEM_UTIL=0.88 bash run_qwen3omni_vllm.sh        # OOM 시 낮춤
#     MAX_NUM_SEQS=16   bash run_qwen3omni_vllm.sh        # 더 공격적 배칭(메모리 여유 시)
#     USE_AUDIO_IN_VIDEO=1 bash run_qwen3omni_vllm.sh     # .wav 대신 mp4 트랙
#     bash run_qwen3omni_vllm.sh --no_score               # 채점 생략
#
#   재실행하면 이미 끝난 청크는 자동 skip(resume).
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/workspace/checkpoints/base/Qwen3-Omni-30B-A3B-Instruct}"
TEST_JSON="${TEST_JSON:-/workspace/data/test/unav100_qwen3omni.json}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/base/Qwen3Omni/unav100_qwen3omni_vllm}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BUILD_BATCH="${BUILD_BATCH:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
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
conda activate qwen3vllm

VLLM_PY="${CONDA_HOME:-/workspace/home/miniconda3}/envs/qwen3vllm/bin/python3"
if [ ! -x "$VLLM_PY" ]; then
    echo "[에러] qwen3vllm python 을 못 찾음: $VLLM_PY"
    exit 1
fi
echo "[python] $VLLM_PY ($("$VLLM_PY" -c 'import vllm,torch;print("vllm",vllm.__version__,"torch",torch.__version__)'))"

if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo "[에러] 모델 경로가 비어있음: $MODEL_PATH"
    exit 1
fi
if [ ! -f "$TEST_JSON" ]; then
    echo "[에러] 테스트셋이 없음: $TEST_JSON (build_unav100_qwen3omni.py 먼저)"
    exit 1
fi

# GPU 점유 경고
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${USED_MIB:-0}" -gt 5000 ]; then
    echo "[경고] GPU 가 이미 ${USED_MIB} MiB 사용 중 — HF 런이 살아있으면 vLLM 이 OOM 난다."
    echo "       먼저:  tmux kill-session -t qwen3omni    (그리고 몇 초 대기)"
    echo "       계속하려면 5초 내 Ctrl-C 안 누르면 진행함..."
    sleep 5
fi

echo "=================================================="
echo "  Qwen3-Omni-30B-A3B-Instruct UnAV-100 추론 (vLLM)"
echo "  MODEL_PATH   : $MODEL_PATH"
echo "  TEST_JSON    : $TEST_JSON"
echo "  OUT_DIR      : $OUT_DIR"
echo "  CHUNK_SIZE   : $CHUNK_SIZE   BUILD_BATCH: $BUILD_BATCH"
echo "  GPU_MEM_UTIL : $GPU_MEM_UTIL  MAX_NUM_SEQS: $MAX_NUM_SEQS  MAX_MODEL_LEN: $MAX_MODEL_LEN"
echo "  AUDIO        : use_audio_in_video=$USE_AUDIO_IN_VIDEO"
echo "=================================================="

"$VLLM_PY" "$SCRIPT_DIR/infer_qwen3omni_vllm.py" \
    --model_path "$MODEL_PATH" \
    --test_json "$TEST_JSON" \
    --out_dir "$OUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    --build_batch "$BUILD_BATCH" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --use_audio_in_video "$USE_AUDIO_IN_VIDEO" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --max_model_len "$MAX_MODEL_LEN" \
    "$@" 2>&1 | tee -a "$OUT_DIR/inference.log"

echo ""
echo "[완료] 최종 결과: $OUT_DIR/eval/table.txt"
[ -f "$OUT_DIR/eval/table.txt" ] && cat "$OUT_DIR/eval/table.txt"

#!/bin/bash
# ============================================================
# run_chronus_sft_infer.sh
#   ChronusOmni SFT(LoRA) 를 UnAV-100 테스트셋에 대해
#   "추론 + 청크채점 + 최종 table.txt" 까지 전부 한 번에 돌리는 실행기.
#   이 스크립트 하나만 실행하면 됨 (infer_chronus_sft.py 를 직접 안 건드려도 됨).
#
#   내부적으로 하는 일 (infer_chronus_sft.py 에 위임):
#     1) TEST_JSON(기본 unav100_chronus.json, 3455개) 을 CHUNK_SIZE(기본 500)개씩
#        7개 청크로 쪼갠다 -> OUT_DIR/inputs/chunk_0000..0006.json
#     2) base 체크포인트 + LoRA 체크포인트를 "인메모리 병합"해서 모델을 1회 로드
#        (디스크에 별도 병합 모델을 저장하지 않음 -> 항상 최신 체크포인트로 다시 병합해서 씀)
#     3) 청크를 0번부터 순서대로 추론 -> 끝난 청크는 즉시 OUT_DIR/results/chunk_XXXX.json
#        에 저장(원자적 쓰기). 이미 저장된 청크는 재실행 시 건너뜀(resume).
#     4) 청크 하나가 끝날 때마다 "청크채점"을 자동 실행: 그 시점까지 쌓인 결과로
#        eval_chronus.py -> eval_miou.py -> maketable.py 를 돌려서
#        OUT_DIR/eval/table.txt 를 계속 갱신한다. GPU 를 쓰지 않는 순수 채점이라
#        추론 진행을 막지 않고, 학습 중인 다른 GPU 작업과도 안 겹친다.
#        -> 즉 전체가 끝나기 전에도 OUT_DIR/eval/table.txt 를 열어보면
#           "지금까지의" sample_mIoU 를 바로 확인할 수 있다(중간 확인용).
#     5) 모든 청크가 끝나면(또는 이미 다 끝나 있던 재실행이어도) 마지막에 한 번 더
#        전체 채점을 돌려 table.txt 를 최종 확정한다.
#
#   ── 중요한 전제 (반드시 학습이 100% 끝난 뒤에 실행할 것) ──────────
#   LORA_CKPT 는 학습이 "완전히 끝난 뒤"의 output_dir 루트를 가리켜야 한다
#   (기본값 /workspace/checkpoints/sft/ChronusOmni 자체 — checkpoint-500 같은
#   중간 서브폴더가 아님!). train.py 는 학습 도중 주기적 체크포인트에는
#   non_lora_trainables.bin 을 저장하지 않고, 학습 루프가 다 끝난 뒤 딱 한 번
#   output_dir 루트에만 저장하기 때문. 학습이 안 끝났는데 이 스크립트를 돌리면
#   "non_lora_trainables.bin 없음" 류의 에러가 난다.
#
#   ── 실행 전 준비 상태 (2026-08-30 확인) ────────────────────────
#   - Chronus repo 루트(/workspace/Chronus)에 checkpoints/ 심볼릭 링크를 미리
#     걸어둠 (large-v3.pt, BEATs_*.pt -> base 체크포인트). base 모델 config 에
#     이 인코더 경로가 상대경로("./checkpoints/...")로 박혀있는데, 로딩 실패를
#     조용히 삼키는 코드라(예외 안 뜨고 인코더가 None이 됨) 이 심볼릭 링크가
#     없으면 매 샘플 추론이 조용히 죽는다. (infer_chronus_sft.py 가 알아서
#     Chronus repo 루트로 cd 하므로 이 링크가 있어야 정상 동작.)
#
#   ── 결과물 ──────────────────────────────────────────────────
#   OUT_DIR/inputs/chunk_XXXX.json    : 쪼개진 입력 (재실행 대비 캐시)
#   OUT_DIR/results/chunk_XXXX.json   : 청크별 추론 결과
#   OUT_DIR/eval/table.txt            : ★ 최종 산출물 (sample_mIoU 등 집계)
#   OUT_DIR/inference.log             : 이 스크립트의 전체 stdout/stderr 로그(tee)
#
#   ── 사용법 ──────────────────────────────────────────────────
#   기본 그대로 실행 (SFT LoRA 평가):
#     bash run_chronus_sft_infer.sh
#
#   경로/청크크기 등을 환경변수로 오버라이드:
#     OUT_DIR=/workspace/outputs/sft/ChronusOmni/my_run \
#     CHUNK_SIZE=500 \
#     bash run_chronus_sft_infer.sh
#
#   LoRA 없이 베이스 모델만 평가하고 싶을 때 (SFT 전/후 비교용):
#     LORA_CKPT=base OUT_DIR=/workspace/outputs/base/ChronusOmni/unav100_chronus \
#     bash run_chronus_sft_infer.sh
#
#   infer_chronus_sft.py 에 추가 인자를 그대로 넘기고 싶을 때 (예: 채점 생략):
#     bash run_chronus_sft_infer.sh --no_score
#
#   죽었다가 이어서 재실행하고 싶을 때: 그냥 다시
#     bash run_chronus_sft_infer.sh
#   (OUT_DIR 이 같으면 이미 끝난 청크는 자동으로 스킵되고 이어서 진행됨)
# ============================================================
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LORA_CKPT="${LORA_CKPT:-/workspace/checkpoints/sft/ChronusOmni}"
MODEL_BASE="${MODEL_BASE:-/workspace/checkpoints/base/ChronusOmni}"
TEST_JSON="${TEST_JSON:-/workspace/Chronus/data/test/unav100_chronus.json}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/sft/ChronusOmni/unav100_chronus}"
CHUNK_SIZE="${CHUNK_SIZE:-500}"

mkdir -p "$OUT_DIR"

# ---- 가상환경/완디비 준비: source setup.sh && conda activate chronusomni ----
# (학습 때와 동일한 준비 순서. eval_chronus.py/eval_miou.py 는 GPU 없이도 돌지만
#  추론 자체(infer_chronus_sft.py)는 이 conda env 의 torch/transformers/peft/
#  decord/whisper 등이 필요하다.)
[ -f /workspace/setup.sh ] && source /workspace/setup.sh
if ! declare -F conda >/dev/null 2>&1; then
    for _c in "$CONDA_HOME" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -n "$_c" ] && [ -f "$_c/etc/profile.d/conda.sh" ] && source "$_c/etc/profile.d/conda.sh" && break
    done
fi
conda activate chronusomni

# ---- python3 바이너리는 PATH 대신 절대경로로 직접 지정 ----
# (실전에서 발견한 함정: 이미 chronusomni 가 활성화된 셸에서 setup.sh/conda activate
#  를 또 source 하면, conda 가 "이미 활성화됨"으로 보고 activate 를 사실상 no-op
#  처리하는 경우가 있는데, 그 직전에 setup.sh 가 export PATH="$CONDA_HOME/bin:$PATH"
#  로 base conda 의 bin 을 PATH 맨 앞에 한 번 더 얹어놓기 때문에 "python3" 가 base
#  conda 의 python(=torch 없음)으로 풀려서 "ModuleNotFoundError: No module named
#  'torch'" 가 나는 걸 실제로 겪었다. tmux 처럼 세션이 오래 살아있고 conda
#  activate 를 여러 번 겹쳐 부른 셸에서만 재현됨. 그래서 이후로는 python3 를
#  PATH 검색에 맡기지 않고 env 안의 실제 바이너리 절대경로를 직접 호출한다.
CHRONUSOMNI_PY="${CONDA_HOME:-/workspace/home/miniconda3}/envs/chronusomni/bin/python3"
if [ ! -x "$CHRONUSOMNI_PY" ]; then
    echo "[에러] chronusomni python 바이너리를 못 찾음: $CHRONUSOMNI_PY"
    exit 1
fi
echo "[python] $CHRONUSOMNI_PY ($("$CHRONUSOMNI_PY" -c 'import torch,sys;print("torch",torch.__version__,"@",sys.executable)'))"

# ---- LoRA 체크포인트 사전 점검: 학습이 실제로 끝났는지(non_lora_trainables.bin) 확인 ----
# base/no 로 LoRA 를 끈 게 아니라면, output_dir 루트에 이 파일이 없다는 건 학습이
# 아직 안 끝났거나(중간 checkpoint-XXX 만 있음) output_dir 을 잘못 짚었다는 뜻이라
# 여기서 바로 알려주고 종료한다 (모델 로딩까지 갔다가 늦게 실패하는 것보다 낫다).
_lc_lower="$(echo "$LORA_CKPT" | tr '[:upper:]' '[:lower:]')"
if [ "$_lc_lower" != "base" ] && [ "$_lc_lower" != "no" ] && [ "$_lc_lower" != "none" ]; then
    if [ ! -f "$LORA_CKPT/non_lora_trainables.bin" ]; then
        echo "[에러] $LORA_CKPT/non_lora_trainables.bin 이 없습니다."
        echo "       학습이 아직 안 끝났거나(체크포인트만 있고 최종 저장 전), LORA_CKPT 경로가"
        echo "       checkpoint-XXX 같은 중간 서브폴더를 가리키고 있는지 확인하세요."
        echo "       (이 파일은 train.py 가 학습 루프를 100% 마친 뒤 output_dir 루트에만 씁니다.)"
        exit 1
    fi
fi

echo "=================================================="
echo "  ChronusOmni SFT 추론 + 청크채점"
echo "  LORA_CKPT  : $LORA_CKPT"
echo "  MODEL_BASE : $MODEL_BASE"
echo "  TEST_JSON  : $TEST_JSON"
echo "  OUT_DIR    : $OUT_DIR"
echo "  CHUNK_SIZE : $CHUNK_SIZE"
echo "=================================================="

# 추론(infer_chronus_sft.py)이 청크마다 자동으로 채점까지 돌리므로, 이 스크립트가
# 끝나는 시점엔 OUT_DIR/eval/table.txt 가 최종 결과로 확정되어 있다.
"$CHRONUSOMNI_PY" "$SCRIPT_DIR/infer_chronus_sft.py" \
    --lora_ckpt "$LORA_CKPT" \
    --model_base "$MODEL_BASE" \
    --test_json "$TEST_JSON" \
    --out_dir "$OUT_DIR" \
    --chunk_size "$CHUNK_SIZE" \
    "$@" 2>&1 | tee -a "$OUT_DIR/inference.log"

echo ""
echo "[완료] 최종 결과: $OUT_DIR/eval/table.txt"
[ -f "$OUT_DIR/eval/table.txt" ] && cat "$OUT_DIR/eval/table.txt"

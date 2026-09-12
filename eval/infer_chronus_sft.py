#!/usr/bin/env python3
"""ChronusOmni SFT(LoRA) UnAV-100 청크 추론 + 청크채점 — resume 안전, 원샷 실행.

이 스크립트 하나로 "추론 → (매 청크마다) 채점 → 최종 table.txt" 까지 전부 돈다.
보통은 직접 실행하지 말고 옆의 run_chronus_sft_infer.sh 를 실행하면 됨
(conda env 활성화 + 인자 기본값 세팅을 그 쉘스크립트가 대신 해줌).

────────────────────────────────────────────────────────────────
전체 흐름 (한 번 실행 시 일어나는 일)
────────────────────────────────────────────────────────────────
  1) --test_json (예: unav100_chronus.json, 3455개) 을 --chunk_size(기본 500) 단위로
     쪼개서 <out_dir>/inputs/chunk_0000.json ... chunk_0006.json 으로 저장.
     (3455 -> 500*6 + 455, 총 7청크. 이미 쪼개져 있으면 재사용, 다시 안 쪼갬)

  2) 모델을 "딱 한 번" 로드:
     - chronus/model/builder.py::load_pretrained_model(lora_ckpt, model_base, is_lora=True) 호출.
     - 이 함수 내부에서 base 모델을 올리고 -> LoRA 어댑터를 PeftModel 로 얹고
       -> merge_and_unload() 로 즉석에서 풀-모델로 합침. 디스크에 별도 병합본을
       저장하지 않고 GPU 메모리 위에서만 합쳐서 쓰기 때문에 --lora_ckpt/--model_base
       만 주면 "머지" 단계가 알아서 끝난다.
     - --lora_ckpt 를 base/no/none 으로 주면 LoRA 를 건너뛰고 베이스 모델만 평가
       (SFT 전/후 비교용).

  3) 청크를 0번부터 순서대로 처리. 청크 하나가 끝나면:
       a) results/chunk_XXXX.json 을 즉시 저장 (원자적 쓰기: .tmp 로 쓰고 rename).
          -> 스크립트가 중간에 죽어도(OOM, 서버 재부팅 등) 이미 저장된 청크는
             안전하게 남아있고, 재실행하면 이 청크는 "skip (already done)" 으로
             건너뛰고 다음 청크부터 이어서 진행한다.
       b) 그 시점까지 저장된 모든 청크를 모아서 채점(=청크채점)을 돌린다:
          eval_chronus.py 를 서브프로세스로 호출 -> 지금까지의 results/chunk_*.json
          을 전부 concat -> id로 GT(gt_segments) 매칭 -> eval_miou.py --natural
          -> maketable.py 로 <out_dir>/eval/table.txt 갱신.
          즉 7청크 중 2청크만 끝난 시점에도 <out_dir>/eval/table.txt 를 열어보면
          "지금까지의" sample_mIoU 를 바로 확인할 수 있다(중간 확인용, 최종치와는
          다를 수 있음 - 표본이 아직 적어서).
     한 샘플에서 예외(디코딩 실패, OOM 등)가 나도 그 샘플만
     {"output": "", "error": "..."} 로 남기고 청크 전체는 죽지 않는다.

  4) 모든 청크가 끝나면(또는 전부 이미 끝나 있어서 skip 만 났어도) 마지막에
     한 번 더 채점을 돌려 <out_dir>/eval/table.txt 를 최종 확정한다.

────────────────────────────────────────────────────────────────
출력 파일 위치
────────────────────────────────────────────────────────────────
  <out_dir>/inputs/chunk_XXXX.json   : 쪼개진 입력 (재실행 대비 캐시)
  <out_dir>/results/chunk_XXXX.json  : 청크별 추론 결과 [{"id","question","output"}, ...]
  <out_dir>/eval/test_results_rank0.json : 전체 결과를 pred/gt_segments 형태로 변환한 것
  <out_dir>/eval/*_miou_summary.json : pairwise/union/sample 3종 mIoU summary
  <out_dir>/eval/table.txt           : ★ 최종 산출물 (sample_mIoU 등 집계 1행)
  <out_dir>/inference.log            : run_chronus_sft_infer.sh 가 tee 로 남기는 전체 로그

사용 (보통은 run_chronus_sft_infer.sh 를 통해 실행):
  python3 infer_chronus_sft.py \
    --lora_ckpt /workspace/checkpoints/sft/ChronusOmni \
    --model_base /workspace/checkpoints/base/ChronusOmni \
    --test_json /workspace/Chronus/data/test/unav100_chronus.json \
    --out_dir /workspace/outputs/sft/ChronusOmni/unav100_chronus \
    --chunk_size 500 --label ChronusOmni-SFT

주의 (지난 병합 조사에서 확인된 전제):
  --lora_ckpt 는 반드시 "학습이 100% 끝난 뒤"의 output_dir 루트를 가리켜야 한다
  (예: /workspace/checkpoints/sft/ChronusOmni 자체). train.py 는 학습 도중의
  주기적 체크포인트(checkpoint-500/ 등)에는 non_lora_trainables.bin 을 안 쓰고,
  학습이 완전히 끝난 뒤 딱 한 번 output_dir 루트에만 써준다. is_lora=True 로딩은
  이 파일이 없으면 에러난다.
"""
import os

# chronus 모듈 임포트 전에 반드시 설정되어야 함 (Chronus/inference/eval.py 와 동일)
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "448"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import json
import time
import argparse
import traceback
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_CHRONUS_PY = os.path.join(SCRIPT_DIR, "eval_chronus.py")

CHRONUS_ROOT = "/workspace/Chronus"
if CHRONUS_ROOT not in sys.path:
    sys.path.insert(0, CHRONUS_ROOT)
# speech_encoder/music_encoder 가 base checkpoint config.json 에 상대경로("./checkpoints/...")로
# 박혀 있어서, cwd 가 Chronus repo 루트여야 whisper/BEATs 가중치를 찾는다.
# (repo 루트에 checkpoints/ 심볼릭 링크를 미리 만들어둠: large-v3.pt, BEATs_*.pt -> base ckpt)
os.chdir(CHRONUS_ROOT)

from tqdm import tqdm
import torch
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import librosa
import whisper
from chronus.conversation import conv_templates, SeparatorStyle
from chronus.model.builder import load_pretrained_model
from chronus.datasets.preprocess import (
    tokenizer_image_token,
    tokenizer_speech_image_token,
    tokenizer_speech_question_image_token,
    tokenizer_speech_token,
)
from chronus.mm_utils import KeywordsStoppingCriteria, process_anyres_video, process_anyres_highres_image
from chronus.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX


def load_audio(audio_file_name):
    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    speechs, speech_wavs = [], []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i: i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = [whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0) for chunk in speechs]

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 20:
        mels = mels[:20]
        speech_wavs = speech_wavs[:20]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs


def split_chunks(data, chunk_size, chunks_dir):
    os.makedirs(chunks_dir, exist_ok=True)
    n_chunks = (len(data) + chunk_size - 1) // chunk_size if data else 0
    for i in range(n_chunks):
        p = os.path.join(chunks_dir, f"chunk_{i:04d}.json")
        if not os.path.exists(p):
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data[i * chunk_size:(i + 1) * chunk_size], f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
    return n_chunks


@torch.inference_mode()
def run_one(sample, tokenizer, model, image_processor, frame_num):
    text = sample["question"]
    video_path = sample.get("video")
    audio_path = sample.get("audio")
    image_path = sample.get("image")

    if video_path is not None:
        modality = "video"
        visual = video_path
    elif image_path is not None:
        modality = "image"
        visual = image_path
    else:
        modality = "text"

    speechs, speech_lengths, speech_wavs, speech_chunks = [], [], [], []
    if modality == "video":
        vr = VideoReader(visual, ctx=cpu(0))
        total_frame_num = len(vr)
        if total_frame_num > frame_num:
            frame_idx = np.linspace(0, total_frame_num - 1, frame_num, dtype=int).tolist()
        else:
            frame_idx = np.arange(0, total_frame_num, dtype=int).tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]
        fps = vr.get_avg_fps()
        video_duration = total_frame_num / fps
        time_interval = video_duration / (min(total_frame_num, frame_num) - 1)
        timestamp = []
        for i in range(len(frame_idx)):
            timestamp_text = 'second{' + "{:.1f}".format(time_interval * i) + '}'
            timestamp.append(torch.tensor(tokenizer(timestamp_text)['input_ids']).cuda())
    elif modality == "image":
        image = [Image.open(visual)]
        image_sizes = [image[0].size]
        timestamp, time_interval = None, 0
    else:
        images = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
        image_sizes = [(224, 224)]
        timestamp, time_interval = None, 0

    if audio_path:
        speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
        speechs.append(speech.bfloat16().to('cuda'))
        speech_lengths.append(speech_length.to('cuda'))
        speech_chunks.append(speech_chunk.to('cuda'))
        speech_wavs.append(speech_wav.bfloat16().to('cuda'))
    else:
        speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
        speech_lengths = [torch.LongTensor([3000]).to('cuda')]
        speech_wavs = [torch.zeros([1, 480000]).bfloat16().to('cuda')]
        speech_chunks = [torch.LongTensor([1]).to('cuda')]

    conv_mode = "qwen_1_5"
    qs = text if text else ''
    if audio_path and image_path:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + "User's question in speech: " + DEFAULT_SPEECH_TOKEN + '\n'
    elif audio_path and video_path:
        qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif audio_path:
        qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
    elif image_path or video_path:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if audio_path and image_path:
        input_ids = tokenizer_speech_question_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif audio_path and video_path:
        input_ids = tokenizer_speech_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    elif audio_path:
        input_ids = tokenizer_speech_token(prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
    else:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

    if modality == "video":
        video_processed = []
        for frame in video:
            image_processor.do_resize = False
            image_processor.do_center_crop = False
            frame = process_anyres_video(frame, image_processor)
            video_processed.append(frame.unsqueeze(0))
        video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
        video_processed = (video_processed, video_processed)
        video_data = (video_processed, (384, 384), "video")
    elif modality == "image":
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_tensor, image_highres_tensor = [], []
        for v in image:
            t_, th_ = process_anyres_highres_image(v, image_processor)
            image_tensor.append(t_)
            image_highres_tensor.append(th_)
        if all(x.shape == image_tensor[0].shape for x in image_tensor):
            image_tensor = torch.stack(image_tensor, dim=0)
        if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
            image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
        image_tensor = (image_tensor.bfloat16().to("cuda") if not isinstance(image_tensor, list)
                         else [x.bfloat16().to("cuda") for x in image_tensor])
        image_highres_tensor = (image_highres_tensor.bfloat16().to("cuda") if not isinstance(image_highres_tensor, list)
                                 else [x.bfloat16().to("cuda") for x in image_highres_tensor])

    pad_token_ids = 151643
    attention_masks = input_ids.ne(pad_token_ids).long().to('cuda')
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    gen_kwargs = dict(max_new_tokens=1024, temperature=0, top_p=None, num_beams=1)

    if modality == "video":
        output_ids = model.generate(
            inputs=input_ids,
            images=[video_data[0][0]],
            images_highres=[video_data[0][1]],
            modalities=video_data[2],
            speech=speechs, speech_lengths=speech_lengths,
            speech_chunks=speech_chunks, speech_wav=speech_wavs,
            attention_mask=attention_masks, use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=False, temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"], max_new_tokens=gen_kwargs["max_new_tokens"],
            timestamp=[timestamp], time_interval=[time_interval],
        )
    elif modality == "image":
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor, images_highres=image_highres_tensor, image_sizes=image_sizes,
            modalities=['image'],
            speech=speechs, speech_lengths=speech_lengths,
            speech_chunks=speech_chunks, speech_wav=speech_wavs,
            attention_mask=attention_masks, use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=False, temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"], max_new_tokens=gen_kwargs["max_new_tokens"],
            timestamp=[timestamp], time_interval=[time_interval],
        )
    else:
        output_ids = model.generate(
            input_ids,
            images=images, images_highres=images_highres, image_sizes=image_sizes,
            modalities=['text'],
            speech=speechs, speech_lengths=speech_lengths,
            speech_chunks=speech_chunks, speech_wav=speech_wavs,
            attention_mask=attention_masks, use_cache=True,
            stopping_criteria=[stopping_criteria],
            do_sample=False, temperature=gen_kwargs["temperature"], top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"], max_new_tokens=gen_kwargs["max_new_tokens"],
            timestamp=[timestamp], time_interval=[time_interval],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return {"id": sample["id"], "question": qs, "output": outputs}


def run_scoring(results_dir, test_json, eval_dir, label, testset, quiet=False):
    """eval_chronus.py 를 서브프로세스로 호출 = '청크채점'.

    지금까지 results_dir 에 쌓인 chunk_*.json 을 전부 모아서(=현재까지 끝난 청크만
    반영, 아직 안 끝난 청크는 그냥 없는 셈) id로 GT(gt_segments)를 붙이고
    eval_miou.py --natural 로 3종 summary를 만든 뒤 maketable.py 로 table.txt 를
    (다시) 만든다. GPU 를 안 쓰기 때문에 학습/추론 중인 GPU 프로세스와 안 겹친다.
    실패해도(예: 아직 결과가 하나도 없어서) 추론 자체를 막지 않도록 예외를 삼킨다.
    """
    cmd = [sys.executable, EVAL_CHRONUS_PY,
           "--results_dir", results_dir, "--test_json", test_json,
           "--eval_dir", eval_dir, "--label", label, "--testset", testset]
    try:
        if quiet:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] 채점 실패(무시하고 계속 진행): {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_ckpt", default="/workspace/checkpoints/sft/ChronusOmni",
                     help="LoRA 체크포인트 디렉토리(non_lora_trainables.bin 있는 학습 완료 후 output_dir 루트). base/no 면 LoRA 미사용")
    ap.add_argument("--model_base", default="/workspace/checkpoints/base/ChronusOmni")
    ap.add_argument("--test_json", default="/workspace/Chronus/data/test/unav100_chronus.json")
    ap.add_argument("--out_dir", default="/workspace/outputs/sft/ChronusOmni/unav100_chronus")
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--frame_num", type=int, default=64)
    ap.add_argument("--label", default=None, help="채점 summary/table.txt 라벨. 기본값은 lora 여부에 따라 자동 결정")
    ap.add_argument("--testset", default=None, help="채점 summary/table.txt 의 testset 태그. 기본값은 test_json 파일명")
    ap.add_argument("--no_score", action="store_true", help="채점 단계를 건너뛰고 추론만 한다")
    args = ap.parse_args()

    is_lora = args.lora_ckpt.lower() not in ("base", "no", "none", "")
    label = args.label or ("ChronusOmni-SFT" if is_lora else "ChronusOmni-base")
    testset = args.testset or os.path.splitext(os.path.basename(args.test_json))[0]

    inputs_dir = os.path.join(args.out_dir, "inputs")
    results_dir = os.path.join(args.out_dir, "results")
    eval_dir = os.path.join(args.out_dir, "eval")
    os.makedirs(results_dir, exist_ok=True)

    with open(args.test_json) as f:
        data = json.load(f)
    n_chunks = split_chunks(data, args.chunk_size, inputs_dir)
    print(f"[infer_chronus_sft] {len(data)} samples -> {n_chunks} chunks of {args.chunk_size} (inputs: {inputs_dir})")

    # use_flash_attn=True 필수: 기본값(False)인 eager attention 은 64프레임 비디오+오디오의
    # 긴 시퀀스에서 어텐션 행렬을 O(n^2)로 통째로 만들려다 샘플마다 CUDA OOM 남
    # (실측: 9~18GB 단발 할당 시도, 거의 매 샘플 실패). flash-attn 설치되어 있어 바로 적용 가능.
    if is_lora:
        print(f"[infer_chronus_sft] loading base={args.model_base}  LoRA={args.lora_ckpt}  (merge_and_unload in-memory, flash_attn=True)")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.lora_ckpt, args.model_base, is_lora=True, use_flash_attn=True)
    else:
        print(f"[infer_chronus_sft] loading base only: {args.model_base} (flash_attn=True)")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.model_base, None, use_flash_attn=True)
    model = model.to('cuda').eval()
    model = model.bfloat16()
    print("[infer_chronus_sft] model ready.")

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
                out = run_one(sample, tokenizer, model, image_processor, args.frame_num)
            except Exception as e:
                n_fail += 1
                traceback.print_exc()
                out = {"id": sample["id"], "question": sample.get("question", ""), "output": "", "error": str(e)}
            chunk_results.append(out)

        # 원자적 저장: .tmp 로 다 쓴 뒤 rename 이라 중간에 죽어도 result_path 는
        # "완전히 끝난 청크"만 가리킨다 (반쯤 쓰인 파일이 남는 일이 없음 -> resume 안전).
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(chunk_results, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, result_path)
        print(f"[chunk {i + 1}/{n_chunks}] done: {len(chunk_results)} samples "
              f"({n_fail} failed) in {time.time() - t0:.1f}s -> {result_path}")

        # ---- 청크채점: 이 청크까지 쌓인 결과로 table.txt 를 즉시 갱신 ----
        # 아직 전체가 안 끝났어도 지금까지의 sample_mIoU 를 바로 확인할 수 있다.
        # (표본 수가 적을수록 최종치와 오차가 클 수 있음 - 참고용)
        if not args.no_score:
            print(f"[chunk {i + 1}/{n_chunks}] 중간 채점 중... (GPU 안 씀, 지금까지 결과만 반영)")
            run_scoring(results_dir, args.test_json, eval_dir, label, testset, quiet=True)

    print(f"\n[infer_chronus_sft] 추론 ALL DONE -> {results_dir}")

    # ---- 최종 채점: 새로 계산한 게 없어도(=이미 다 끝나 있던 재실행이어도) 항상 한 번 더
    # 돌려서 table.txt 가 최신 상태임을 보장한다. ----
    if not args.no_score:
        print("[infer_chronus_sft] 최종 채점 실행...")
        run_scoring(results_dir, args.test_json, eval_dir, label, testset, quiet=False)
        table = os.path.join(eval_dir, "table.txt")
        print(f"\n[infer_chronus_sft] ★ 최종 결과 -> {table}")
    else:
        print("[infer_chronus_sft] --no_score 지정됨: 채점 생략. 나중에 수동으로:")
        print(f"  python3 {EVAL_CHRONUS_PY} --results_dir {results_dir} "
              f"--test_json {args.test_json} --eval_dir {eval_dir} --label {label}")


if __name__ == "__main__":
    main()

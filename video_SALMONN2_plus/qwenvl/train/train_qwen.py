# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import time

# transformers' _load_rng_state forces weights_only=True, which rejects numpy dtypes
# saved by older trainer versions. Our checkpoints are trusted, so allow full unpickle.
_orig_torch_load = torch.load
def _torch_load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_trusted

from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module, _TIME_MARKER_TOKEN_LEN
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist

local_rank = None

def collate_fn(batch):
    return batch[0]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen2.5-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen2_5_vl

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.visual.requires_grad_(True)
    else:
        model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.visual.merger.requires_grad_(True)
    else:
        model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.audio.requires_grad_(True)
    else:
        model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.audio.qformer.requires_grad_(True)
        model.audio.q_tokens.requires_grad_(True)
        model.audio.audio_proj.requires_grad_(True)
    else:
        model.audio.qformer.requires_grad_(False)
        model.audio.q_tokens.requires_grad_(False)
        model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)

    if model_args.tune_lm_head:
        model.lm_head.requires_grad_(True)
        model.model.embed_tokens.requires_grad_(True)


def train(attn_implementation="flash_attention_2"):
    global local_rank

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert data_args.train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {data_args.train_type} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        model_args.model_base,
    )
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    

    if not data_args.run_test:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # time.sleep(random.randint(0, 20))
        # print(f"RANK {dist.get_rank()} before barrier")
        dist.barrier(device_ids=[dist.get_rank()])
        # print(f"RANK {dist.get_rank()} after barrier")
        model = video_SALMONN2_plus.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model.config.use_cache = False
        # TTI input-side marker 모드를 모델 config에 기록 → 추론/eval 시 rope2d가 자동으로
        # 같은 길이 가정으로 동작.
        model.config.tti_time_format = data_args.tti_time_format
        # 마커 길이는 dataset.py 의 _TIME_MARKER_TOKEN_LEN 단일 진실원에서 가져온다.
        # (여기에 값을 복제하면 dataset.py 와 어긋나 rope 위치가 밀린다 — 과거 9/8 불일치 사고)
        # off 는 0 → rope 가 '마커 없음' 으로 보도록 None 으로 변환.
        model.config.time_marker_token_len = (
            _TIME_MARKER_TOKEN_LEN[data_args.tti_time_format] or None
        )

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if "3" not in training_args.deepspeed:
                if training_args.gradient_checkpointing_kwargs is None:
                    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
                else:
                    training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        if model_args.lora_ckpt != "No":
            from peft import PeftModel
            audio_layers = model.audio.layers
            del model.audio.layers
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model.model.audio.layers = audio_layers
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(training_args.output_dir, "base/"))

        set_model(model_args, model)

        if training_args.no_audio:
            del model.audio

        if model_args.use_lora:
            from peft import LoraConfig, get_peft_model
            module_to_save = []
            if model_args.tune_mm_vision:
                module_to_save.append("visual")
            if model_args.tune_mm_mlp:
                module_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                module_to_save.append("audio")
            if model_args.tune_mm_qformer:
                module_to_save.append("audio.qformer")
                module_to_save.append("audio.q_tokens")
                module_to_save.append("audio.audio_proj")
            if model_args.tune_lm_head:
                module_to_save.append("lm_head")
                module_to_save.append("model.embed_tokens")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=module_to_save,
            )
            if not training_args.no_audio:
                audio_layers = model.audio.layers
                del model.audio.layers
            model = get_peft_model(model, lora_config)
            if not training_args.no_audio:
                model.model.audio.layers = audio_layers

            for k, v in model.named_parameters():
                if "lora" in k:
                    v.requires_grad_(True)
        
        if dist.get_rank() == 0:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    print(k, v.shape)
            # print(model.model.visual.merger)

        trainer = QwenVLTrainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )
        
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)

        model.config.use_cache = True

        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        pred_rank = training_args.pred_rank
        if torch.cuda.device_count() > 1:
            pred_rank = pred_rank * torch.cuda.device_count() + torch.cuda.current_device()
            data_args.dataset_use = f"dataset/{pred_rank}.json"
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

        if model_args.lora_ckpt != "No":
            if dist.get_rank() == 0:
                model = video_SALMONN2_plus.from_pretrained(
                    model_args.model_name_or_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    device_map="cpu"
                )
                from peft import PeftModel
                if not training_args.no_audio:
                    audio_layers = model.audio.layers
                    del model.audio.layers
                model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
                if not training_args.no_audio:
                    model.model.audio.layers = audio_layers
                model = model.merge_and_unload()

                if torch.cuda.device_count() > 1:
                    model.save_pretrained(os.path.join(training_args.output_dir, "generation"))
                else:
                    model.save_pretrained(os.path.join(training_args.output_dir, f"generation_{pred_rank}"))
            dist.barrier(device_ids=[local_rank])
            

        if torch.cuda.device_count() > 1:
            ds_config = {
                "fp16": {"enabled": False},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3
                },
                "train_micro_batch_size_per_gpu": 1,
            }
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
            hfdsc = HfDeepSpeedConfig(ds_config)

        if model_args.lora_ckpt == "No":
            model = video_SALMONN2_plus.from_pretrained(
                model_args.model_name_or_path,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            if torch.cuda.device_count() > 1:
                model = video_SALMONN2_plus.from_pretrained(
                    os.path.join(training_args.output_dir, "generation"),
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                )
            else:
                model = video_SALMONN2_plus.from_pretrained(
                    os.path.join(training_args.output_dir, f"generation_{pred_rank}"),
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                )
        if training_args.no_audio:
            del model.audio

        # TTI marker 모드를 추론 시에도 명시 — 학습 시 저장된 config 값이 비어있는
        # 옛 체크포인트로 평가할 때 backward-compat를 보장.
        model.config.tti_time_format = data_args.tti_time_format
        # 마커 길이는 dataset.py 의 _TIME_MARKER_TOKEN_LEN 단일 진실원에서 가져온다.
        # (여기에 값을 복제하면 dataset.py 와 어긋나 rope 위치가 밀린다 — 과거 9/8 불일치 사고)
        # off 는 0 → rope 가 '마커 없음' 으로 보도록 None 으로 변환.
        model.config.time_marker_token_len = (
            _TIME_MARKER_TOKEN_LEN[data_args.tti_time_format] or None
        )

        if torch.cuda.device_count() > 1:
            import deepspeed
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            model = ds_engine.module
        else:
            model.cuda()

        result = []
        tt_debug = []  # [TT_DEBUG] 생성 원본 토큰 ID 분석 누적

        def _tt_classify(gen_ids):
            """생성된 원본 토큰 ID 가 real 타임토큰(151666~151676)인지 BPE subword 인지 판정.
            real 이면 decode(skip_special_tokens=True)에서 사라지고, subword 면 그대로 남는다."""
            import re as _re
            TIME_LO, TIME_HI = 151666, 151676   # <t0>..<t9>,<tdot> (added_tokens.json)
            ids = [int(x) for x in gen_ids]
            n_real = sum(1 for i in ids if TIME_LO <= i <= TIME_HI)
            toks = tokenizer.convert_ids_to_tokens(ids)
            dec_keep = tokenizer.decode(ids, skip_special_tokens=False)
            dec_strip = tokenizer.decode(ids, skip_special_tokens=True)
            n_markers = len(_re.findall(r"<t\d>|<tdot>", dec_keep))
            if n_real == 0 and n_markers > 0:
                verdict = "BPE_SUBWORD"          # 텍스트엔 <t..> 있는데 real id 0개 → 글자조각
            elif n_real > 0 and n_real >= n_markers:
                verdict = "REAL_TIME_TOKEN"
            elif n_real > 0:
                verdict = "MIXED"
            else:
                verdict = "NO_TIME_MARKER"
            region = [[i, t] for i, t in zip(ids, toks)
                      if (TIME_LO <= i <= TIME_HI) or (t in ("<", "t", ">", "tdot"))
                      or (isinstance(t, str) and t.strip("Ġ▁").isdigit())][:24]
            return {
                "verdict": verdict,
                "n_real_timetoken": n_real,
                "n_markers_in_text": n_markers,
                "gen_len": len(ids),
                "decode_skip_true": dec_strip,
                "decode_skip_false": dec_keep,
                "region_tokens": region,
                "raw_ids_head": ids[:40],
            }

        test_data = data_module["train_dataset"]
        loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collate_fn,
        )
        for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
            if inputs:
                res_i = {
                    "video": inputs.pop("video", None),
                    "image": inputs.pop("image", None),
                    "prompt": inputs.pop("prompt", None),
                    "ref": inputs.pop("ref", None),
                    "gt_label": inputs.pop("gt_label", None),
                    "audio": inputs.pop("audio", None),
                    "use_audio": inputs.pop("use_audio", False),
                    "should_use": inputs.pop("should_use", True),
                }
                # audio_lengths 는 파이썬 list 라 아래 텐서 필터에 걸려 사라진다. 그런데
                # rope2d.get_rope_index_25 는 (audio_lengths is not None) 여야 TTI 마커를
                # 인지하는 분기로 들어가므로, 빠지면 마커가 input_ids 에 있어도 위치 계산이
                # 마커를 무시해 시간 grounding 이 끊긴다(= 학습 val 대비 mIoU 반토막).
                # 학습측 _val_greedy_generate 는 이 값을 명시적으로 넘긴다 — 동일하게 맞춘다.
                _audio_lengths = inputs.get("audio_lengths", None)
                inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                if _audio_lengths is not None:
                    inputs["audio_lengths"] = _audio_lengths
                # debug_interleave_dir 가 설정되고 debug_interleave_generate 가 False 면
                # dataset hook 이 이미 덤프했으니 model.generate 를 스킵한다.
                _dbg_skip = (
                    bool(getattr(data_args, "debug_interleave_dir", ""))
                    and not getattr(data_args, "debug_interleave_generate", False)
                )
                for _ in range(data_args.num_sample):
                    if _dbg_skip:
                        output_text = "[debug_interleave: generate skipped]"
                    else:
                        with torch.no_grad():
                            # repetition_penalty 를 명시하지 않으면 generation_config.json 의
                            # 1.05 가 greedy 에서도 적용된다. 타임토큰 출력은 같은 숫자 토큰을
                            # 반복 사용하므로 페널티가 값 자체를 왜곡하고, 종료(.+EOS)도 억제해
                            # 세그먼트를 과분할한다(학습 pred/샘플 1.36 → 평가 2.10).
                            # 학습측(rollout·val)은 둘 다 1.0 을 명시한다 — 동일하게 맞춘다.
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=1024,
                                do_sample=data_args.do_sample,
                                top_p=0.9,
                                repetition_penalty=1.0)
                        output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
                        output_text = tokenizer.decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        if os.environ.get("TT_DEBUG", "0") == "1":
                            _ti = _tt_classify(output_trimmed.tolist())
                            _ti["idx"] = len(tt_debug)
                            tt_debug.append(_ti)
                            print(f"[TT-DBG] #{_ti['idx']} verdict={_ti['verdict']} "
                                  f"real={_ti['n_real_timetoken']} markers_in_text={_ti['n_markers_in_text']} "
                                  f"gen_len={_ti['gen_len']}", flush=True)
                            print(f"         skip=True : {_ti['decode_skip_true'][:140]!r}", flush=True)
                            print(f"         skip=False: {_ti['decode_skip_false'][:140]!r}", flush=True)
                            print(f"         region   : {_ti['region_tokens'][:16]}", flush=True)
                    if data_args.num_sample == 1:
                        res_i["pred"] = output_text
                    else:
                        if "pred" in res_i:
                            res_i["pred"].append(output_text)
                        else:
                            res_i["pred"] = [output_text]
                if not res_i["should_use"]:
                    continue
                result.append(res_i)
        with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if os.environ.get("TT_DEBUG", "0") == "1":
            _dbg_path = os.path.join(training_args.output_dir, training_args.run_name,
                                     f"time_token_debug_rank{pred_rank}.json")
            with open(_dbg_path, "w") as f:
                json.dump(tt_debug, f, indent=2, ensure_ascii=False)
            from collections import Counter as _Counter
            _summary = dict(_Counter(d["verdict"] for d in tt_debug))
            print(f"[TT-DBG] ===== SUMMARY (n={len(tt_debug)}) verdict 분포: {_summary} =====", flush=True)
            print(f"[TT-DBG] 상세 저장: {_dbg_path}", flush=True)

        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")

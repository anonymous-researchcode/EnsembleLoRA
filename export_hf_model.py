# %%
import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.custom.alpaca_model import AlpacaModel
from src.custom.alpaca_data_module import AlpacaDataModule
from peft import get_peft_model, LoraConfig


logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision("high")

class args:
    model_key = "meta-llama/Llama-3.2-1B" #  "../llama/llama-3/Meta-Llama-3-8B-hf" #  # "google/gemma-2b" # "EleutherAI/gpt-neo-1.3B" # "mistralai/Mistral-7B-v0.3" #
    train_lora = True
    lora_rank = 16
    lora_alpha = 128

    lr = 5e-5
    weight_decay = 1e-4
    max_length = 256
    use_wandb = False
    load_model_dir =  "meta-llama-Llama-3.2-1B_hellaswag_lora_r_16_task_grouping_4_run_0/epoch_epoch=4.ckpt" # "Alpaca_google-gemma-2b_lora_r_4_run_0/epoch_epoch=0" #
    # save_model_dir = "./exported_model/TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_strategy_qa_ft_cot_t70_64aug_lora_r_16_meta_train_epoch_4"
    # "external_lightning_logs/Instruction__TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_128_selected_0.20_run_0/epoch_epoch=0"
    # "./exported_model/Alpaca_TinyLlama-TinyLlama-1.1B-intermediate-step-1431k-3T_lora_r_4_quantized_meta_initialization_run_0_epoch_epoch=4"

    use_qlora = False
    device = 0


print("arguments".upper().center(80, "-"))
print(args)
print("-" * 80)

model_key = args.model_key

if "gpt" in args.model_key or "Llama" in model_key \
    or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
    hf_key = args.model_key.replace("_", "-")
    tokenizer = AutoTokenizer.from_pretrained(hf_key)
    if args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
            )
        model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16,
                                                     device_map={"": args.device}) 
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_key)
    model_type = "decoder"
    append_eos = True
elif "flan" in model_key:
    hf_key = "google/{}".format(model_key.replace("_", "-"))
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_key)
    tokenizer = AutoTokenizer.from_pretrained(hf_key, model_max_length=512)
    model_type = "encoder_decoder"
    append_eos = False  # t5 tokenizers already append eos
else:
    raise NotImplementedError(args.model_key)

if args.train_lora:
        if args.model_key == "gpt2": # for gpt2, we generally use full model
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["c_attn", "c_proj", "c_fc"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif args.model_key == "EleutherAI/gpt-neox-20b":
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["query_key_value"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        elif "flan" in args.model_key:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q", "k", "v"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        else:
            config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="lora_only",
                modules_to_save=[],
            )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

lm = AlpacaModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb)

# %%
from lightning_fabric.utilities.cloud_io import _load as pl_load
load_model_dir = args.load_model_dir
load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
args.save_model_dir = load_model_dir.replace(".ckpt", ".pt")

checkpoint = pl_load(load_model_dir, map_location=f"cpu")
state_dict = checkpoint["state_dict"]
state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
# model.load_state_dict(state_dict, strict=False)

# for epoch in range(0,10):
#     args.load_model_dir = args.load_model_dir[:-1] + str(epoch)
#     args.save_model_dir = args.save_model_dir[:-1] + str(epoch)
# load_model_dir = args.load_model_dir
# load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
# if load_model_dir is not None and os.path.exists(load_model_dir + ".ckpt"):
#     lm = AlpacaModel.load_from_checkpoint(load_model_dir + ".ckpt",  map_location=f"cuda:{args.device}", model=model, tokenizer=tokenizer, model_type=model_type, use_cpu_offload=False,
#                 lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length)
#     logging.info(f"Loaded model from {load_model_dir}")
# 
# def get_state_dict(model):
#     state_dict = model.state_dict()
#     returned_state_dict = {}
#     for k in state_dict.keys():
#         if "lora" in k: 
#             returned_state_dict[k] = state_dict[k].cpu().clone()
#     return returned_state_dict

print(list((state_dict.keys())))
torch.save(state_dict, f"{args.save_model_dir}")

# %%
# torch.load(f"{args.save_model_dir}")

# # %%
# import numpy as np
# def compute_norm(state_dict, use_lora = True, removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings"]):
#     norm = 0
#     for key, val in state_dict.items():
#         if use_lora:
#             if "lora" in key:
#                 norm += val.clone().square().sum().item()
#         else:
#             if any([rkey in key for rkey in removing_keys]):
#                     continue
#             norm += val.clone().square().sum().item()
#     return np.math.sqrt(norm)


# compute_norm(lm.model.state_dict(), use_lora=True)

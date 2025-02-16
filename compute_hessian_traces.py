import argparse
import logging
import os
import wandb

from src.custom.glue_multitask_data_module import GLUEMultitaskDataModule
from src.custom.glue_multitask_model import GLUEMultitaskModel

from functools import partial
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy, _or_policy
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.trainer.states import RunningStage, TrainerFn

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Embedding

from peft import get_peft_model, LoraConfig
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from collections import defaultdict
import time

from adapters import SeqBnInvConfig, PrefixTuningConfig, BnConfig, DoubleSeqBnConfig, SeqBnConfig
from adapters import AutoAdapterModel,list_adapters, BnConfig
from torch._inductor.async_compile import AsyncCompile
from src.utils.hessian import compute_hessians_trace, compute_eigenvalue

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")


def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in args.model_key or "Llama" in model_key \
        or "bloomz" in model_key or "gemma" in model_key or "Mistral" in model_key:
        hf_key = args.model_key.replace("_", "-")
        tokenizer = AutoTokenizer.from_pretrained(hf_key)
        tokenizer.padding_side = 'right'
        if args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
                )
            model = AutoModelForCausalLM.from_pretrained(hf_key, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map={"": args.devices[0]}) #
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
    
    
    if args.train_adapter:
        
        if args.use_qadapter:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' 
            )

            model = AutoAdapterModel.from_pretrained(
                hf_key, 
                quantization_config=quantization_config, 
                torch_dtype=torch.bfloat16, 
                device_map={"": args.devices[0]}
            )
        
        else: model = AutoAdapterModel.from_pretrained(hf_key)

        bottleneck_config = DoubleSeqBnConfig(
            mh_adapter=True,    
            output_adapter=True,    
            reduction_factor=args.reduction_factor,     
            non_linearity="relu"     
        )

        model.add_adapter(adapter_name="seq_bn",config=bottleneck_config)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

        model.set_active_adapters("seq_bn")
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params_count = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable_params_count} || All parameters: {all_params_count} || ratio: {trainable_params_count/all_params_count}")
        print("-"*20,"Bottleneck_Adapter","-"*20)

    
    if args.use_3bit or args.use_2bit:
        from src.lqlora_utils import lora_utils
        model = lora_utils.prepare_model_for_lora(
            model=model,
            num_ranks=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.1,
            use_gradient_checkpointing=True)

        lora_utils.transform_lora_layers(
            lpq=False,
            model=model,
            model_name="nf3" if args.use_3bit else "nf2",
            device=f"cuda:{args.devices[0]}")
        model.to(f"cuda:{args.devices[0]}")        

    elif args.train_lora:
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, hf_key, model_type, append_eos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, nargs="+", default=["cola"])
    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    # deprecated
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--downsample_ratio", type=float, default=1.0)
    parser.add_argument("--minimum_samples", type=int, default=1e6)
    parser.add_argument("--minimum_samples_validation", type=int, default=1e6)

    parser.add_argument("--train_adapter", action="store_true")
    parser.add_argument("--reduction_factor", type=int, default=128)
    parser.add_argument("--use_qadapter", action="store_true")

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_3bit", action="store_true")
    parser.add_argument("--use_2bit", action="store_true")

    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    parser.add_argument("--save_name", type=str, default=None)

    parser.add_argument("--load_model_dir", type=str, default=None)
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--generate_output", action="store_true")

    # for hessians
    parser.add_argument('--use_test', action="store_true")
    parser.add_argument('--num_samples', type=int, default=1)

    args = parser.parse_args()
    args.enable_checkpointing = not args.disable_checkpointing
    print("arguments".upper().center(80, "-"))
    print(args)
    print("-" * 80)

    model_key = args.model_key.replace("/", "-").replace("..", "")
    save_name = (f"{args.save_name}" if args.save_name else "") + \
                (f"_lora_r_{args.lora_rank}" if args.train_lora else "") 
    file_dir = os.path.join("./results/", save_name)
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    metrics = {}

    # Load the model
    model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

    batch_size = args.batch_size
    if args.inference_batch_size is None:
        inference_batch_size = batch_size
    else:
        inference_batch_size = args.inference_batch_size
    
    data_module = GLUEMultitaskDataModule(
            task_names=args.task_names,
            tokenizer=tokenizer,
            batch_size=batch_size,
            inference_batch_size=inference_batch_size,
            max_input_length=args.max_length,
            downsample_ratio=args.downsample_ratio,
            minimum_samples=args.minimum_samples,
            minimum_samples_validation=args.minimum_samples_validation)
    data_module.setup(stage="fit")

    # task_answer_choices = {}
    # for task_name in args.task_names:
    #     answer_choices = data_module.task_to_templates[task_name].answer_choices.split("|||")
    #     # process the answer choices, different models tokenize them differently
    #     if "gpt" in args.model_key: 
    #         answer_choices = [" " + choice.strip() for choice in answer_choices] 
    #         answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
    #     elif "TinyLlama" in args.model_key or ("CodeLlama" in args.model_key):
    #         answer_choices = [choice.strip() for choice in answer_choices]
    #         answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
    #     elif "Llama-3" in args.model_key:
    #         answer_choices = [" " + choice.strip() for choice in answer_choices] 
    #         answer_choices = [tokenizer([choice])["input_ids"][0][1] for choice in answer_choices]; answer_choices.sort()
    #     else:
    #         answer_choices = [" " + choice.strip() for choice in answer_choices] 
    #         answer_choices = [tokenizer([choice])["input_ids"][0][0] for choice in answer_choices]; answer_choices.sort()
    #     task_answer_choices[task_name] = answer_choices
    # lm = GLUEMultitaskModel(model, tokenizer, model_type, use_cpu_offload=False,
    #                 lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
    #                 optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names, task_answer_choices=task_answer_choices)
    
    # load checkpoint
    load_model_dir = args.load_model_dir
    if load_model_dir is not None:
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        assert ("pt" in load_model_dir) and os.path.exists(load_model_dir)
        print(model.load_state_dict(torch.load(load_model_dir), strict=False))
        print(f"Loaded model from {load_model_dir}")


    hessian_traces = []; hessian_lambdas = []
    max_layer_trace = []; max_lambda_1 = []

    data_loader = data_module.test_dataloader() if args.use_test else data_module.train_dataloader() 
    device = torch.device(f"cuda:{args.devices[0]}")
    sample_size = 0
    model.eval()
    if (not args.use_qlora) and (not args.use_qadapter):
        model.to(device)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for batch in data_loader:
            task_name = batch["task_name"]; batch = batch["data"]
            batch = {k: v.to(device) for k, v in batch.items()}
            num_samples = batch["input_ids"].shape[0]
            kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            loss = model(**kwargs)["loss"]
            print(loss)

            layer_traces = compute_hessians_trace(model, loss, device=device)
            lambda_1, _ = compute_eigenvalue(model, loss, device=device, top_n=1) 
            
            hessian_traces.append(layer_traces)
            hessian_lambdas.append(np.array(lambda_1[0]))

            if len(max_layer_trace) == 0:
                max_layer_trace = layer_traces
                max_lambda_1 = np.array(lambda_1[0])
            else:
                max_layer_trace = np.maximum(max_layer_trace, layer_traces) # layer-wise maximum
                max_lambda_1 = np.maximum(max_lambda_1, np.array(lambda_1[0])) # layer-wise maximum

            print("Maximum Layer Trace: ")
            print(max_layer_trace); print(max_layer_trace.sum())
            print("Maximum Top-1 Eigenvalues: ")
            print(max_lambda_1); print(max_lambda_1.sum())
            print("Current sum of layer trace: ")
            print(np.mean(np.array(hessian_traces), axis=0), np.mean(np.array(hessian_traces), axis=0).sum())
            print("Current sum of top-1 eigenvalues: ")
            print(np.mean(np.array(hessian_lambdas), axis=0), np.mean(np.array(hessian_lambdas), axis=0).sum())
            print("========== Batch Complete ==========")

            sample_size += num_samples
            if sample_size > args.num_samples:
                break
    print("Sum of trace: {}".format(np.mean(np.array(hessian_traces), axis=0).sum())) # average of layer-wise traces
    print("Sum of top-1 eigenvalues: {}".format(np.mean(np.array(hessian_lambdas), axis=0).sum())) # average of layer-wise top-1 eigenvalues
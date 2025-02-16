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

from src.custom.shakespeare_data_module import ShakespeareDataModule
from src.custom.shakespeare_model import ShakespeareModel

logging.basicConfig(level=logging.INFO, force=True)
torch.set_float32_matmul_precision("high")

# peft.__version__ '0.12.0'    

def add_result_to_csv(result_datapoint, file_name):
    for key, val in result_datapoint.items():
        result_datapoint[key] = [val, ]
    
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        tmp_df = pd.DataFrame(result_datapoint)
        result_df = pd.concat([result_df, tmp_df], ignore_index = True)
        result_df.to_csv(file_name)
    else:
        result_df = pd.DataFrame(result_datapoint)  
        result_df.to_csv(file_name)   

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
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

    return model, tokenizer, hf_key, model_type, append_eos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tasks", type=int, default=500)
    parser.add_argument("--task_idxes", type=int, nargs="+", default=None)

    parser.add_argument("--model_key", type=str, default="gpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=256)
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
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--load_model_dir", type=str, default="test")
    parser.add_argument("--write_results", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--generate_output", action="store_true")

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
    for run in range(args.runs):
        model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        batch_size = args.batch_size
        if args.inference_batch_size is None:
            inference_batch_size = batch_size
        else:
            inference_batch_size = args.inference_batch_size
        
        task_idxes = args.task_idxes if args.task_idxes is not None else list(range(args.num_tasks))
        data_module = ShakespeareDataModule(task_idxes,
                tokenizer=tokenizer,
                batch_size=batch_size,
                inference_batch_size=inference_batch_size,
                max_input_length=args.max_length,
                downsample_ratio=args.downsample_ratio,
                minimum_samples=args.minimum_samples,
                minimum_samples_validation=args.minimum_samples_validation)
        data_module.setup(stage="fit")

        task_names = [task_name for task_name in data_module.task_to_train_datasets.keys()]
        lm = ShakespeareModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                optimizer=args.optimizer, generate_output=args.generate_output, task_names=task_names)
        
        load_model_dir = args.load_model_dir
        load_model_dir = os.path.join("external_lightning_logs", load_model_dir)
        if load_model_dir is not None:
            if ("ckpt" in load_model_dir) and os.path.exists(load_model_dir):
                lm = ShakespeareModel.load_from_checkpoint(load_model_dir, model=model, tokenizer=tokenizer, model_type=model_type,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=args.task_names)
                print(f"Loaded model from {load_model_dir}")
            elif ("pt" in load_model_dir) and os.path.exists(load_model_dir):
                model.load_state_dict(torch.load(load_model_dir), strict=False)
                print(f"Loaded model from {load_model_dir}")

        if not os.path.exists("external_lightning_logs"):
            raise Exception("external_lightning_logs/ does not exist")
        default_root_dir = os.path.join("external_lightning_logs", 
                                        f"{model_key}_" + \
                                        "shakespeare_num_tasks_{}".format(len(task_names)) + \
                                        (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                                        (f"_{args.save_name}" if args.save_name else "") + \
                                        f"_run_{run}"
                                        )

        
        checkpoint_callback = ModelCheckpoint(
            monitor="loss",
            dirpath=default_root_dir,
            filename="epoch_{epoch}",
            save_top_k=(-1 if args.save_every_epoch else 1),
            mode="min",
        )
        
        trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                            default_root_dir=default_root_dir, min_epochs=args.epochs, max_epochs=args.epochs,
                            accumulate_grad_batches=args.accumulate, precision=args.precision,
                            enable_checkpointing=args.enable_checkpointing,
                            callbacks=[checkpoint_callback]
                            )
        # save initial weights
        if args.train_lora:
            if not os.path.exists(default_root_dir):
                os.makedirs(default_root_dir)
            model_path = default_root_dir + "/initial_weights.pt"
            state_dict = model.state_dict()
            state_dict = {k: v.clone() for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, model_path)
        # elif args.train_adapter:
        #     if not os.path.exists(default_root_dir):
        #         os.makedirs(default_root_dir)
        #     model_path = default_root_dir + "/initial_weights.pt"
        #     state_dict = model.state_dict()
        #     state_dict = {k: v for k, v in state_dict.items() if ("adapter" in k) or ("head" in k)}
        #     torch.save(state_dict, model_path)

        start_time = time.time()
        if args.epochs > 0:
            trainer.fit(lm, datamodule=data_module)
        end_time = time.time()
        print(f"Training time: {end_time - start_time}")


        # evaluate the best checkpoint
        start_time = time.time()
        if args.train_lora:
            from lightning_fabric.utilities.cloud_io import _load as pl_load
            checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
            state_dict = checkpoint["state_dict"]
            state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
            torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
        elif args.train_adapter:
            from lightning_fabric.utilities.cloud_io import _load as pl_load
            checkpoint = pl_load(checkpoint_callback.best_model_path, map_location=lm.device)
            state_dict = checkpoint["state_dict"]
            state_dict = {k[6:]: v for k, v in state_dict.items() if ("adapter" in k or "head" in k)}
            torch.save(state_dict, checkpoint_callback.best_model_path.replace(".ckpt", ".pt"))
            
            
        if args.epochs > 0:
            if args.use_qadapter or args.use_qlora or args.use_3bit or args.use_2bit:                         
                model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
                model.load_state_dict(state_dict, strict=False)
                lm = ShakespeareModel(model, tokenizer, model_type, use_cpu_offload=False,
                        lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb,
                        optimizer=args.optimizer, generate_output=args.generate_output, task_names=task_names)
                if args.use_3bit or args.use_2bit:
                    trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                    trainer.validate_loop.inference_mode = False
                summary = trainer.validate(lm, datamodule=data_module)[0]
            else:
                summary = trainer.validate(lm, datamodule=data_module, ckpt_path=checkpoint_callback.best_model_path)[0]
            logging.info(summary)
        else:
            if args.use_3bit or args.use_2bit:
                trainer.validate_loop.trainer_fn = TrainerFn.FITTING
                trainer.validate_loop.inference_mode = False
            summary = trainer.validate(lm, datamodule=data_module)[0]
            logging.info(summary)
        end_time = time.time()
        print(f"Evaluation time: {end_time - start_time}")
            
        for key in summary:
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(summary[key])

        # delete the whole model checkpoint and only keep the lora parameters
        if args.train_lora or args.train_adapter:
            os.system(f"rm {checkpoint_callback.best_model_path}")
    
    for key in metrics:
        logging.info("{}: {:.4f} +/- {:.4f}".format(key, np.mean(metrics[key]), np.std(metrics[key])))
    
    # save indexes 
    if args.write_results:
        for i, task_name in enumerate(task_names):
            result_datapoint = {
                "Task index": i,
                "Trained with": " ".join([str(i) for i in task_idxes]),
            }
            for key, val in metrics.items():
                if task_name in key:
                    tmp_key = key.replace(f"{task_name}_", "")
                    result_datapoint[tmp_key] = np.mean(val)
            file_name = os.path.join(file_dir, "results.csv")
            add_result_to_csv(result_datapoint, file_name)
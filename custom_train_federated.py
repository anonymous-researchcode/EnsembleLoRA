# %% 
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import get_peft_model, LoraConfig
from src.lqlora_utils import lora_utils

import argparse
import logging
import os
import wandb

from src.custom.shakespeare_model import ShakespeareModel
from src.lqlora_utils import lora_utils

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

from torch._inductor.async_compile import AsyncCompile

def initialize_model(args):
    model_key = args.model_key.replace("/", "-").replace("..", "")
    if "gpt" in model_key or "Llama" in model_key \
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
    
    if args.use_3bit or args.use_2bit:
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

# %%
class args:
    model_key = "meta-llama/Llama-3.1-8B"
    batch_size = 4
    inference_batch_size = 8
    devices = [1]
    accumulate = 1
    strategy = "auto"
    precision = "bf16-true"
    lr = 1e-5
    weight_decay = 0
    epochs = 100
    max_length = 256
    save_every_epoch = False
    downsample = None
    
    train_lora = True
    lora_rank = 16
    lora_alpha = 128

    optimizer = "adamw"
    use_qlora = False
    use_3bit = False
    use_2bit = False

    save_name = None
    load_model_dirs = []
    write_results = False
    use_wandb = False
    generate_output = False 

    downsample_rate = 1.0
    n_estimators = 10

    

# %%
# Initialize the model
model_key = args.model_key.replace("/", "-").replace("..", "")
save_name = (f"_{args.save_name}" if args.save_name else "") + \
            (f"_lora_r_{args.lora_rank}" if args.train_lora else "")         
metrics = {}
model, tokenizer, hf_key, model_type, append_eos = initialize_model(args)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

import numpy as np
from src.custom.shakespeare_data_module import ShakespeareDataModule
from transformers import AutoTokenizer

task_idxes = np.arange(0, 200)
# create a dataloader for all tasks
test_data_module = ShakespeareDataModule(task_idxes, tokenizer, batch_size=8, inference_batch_size=8, max_input_length=80, downsample_ratio=1.0, minimum_samples=100, minimum_samples_validation=100, downsample_seed=0)
test_data_module.setup(stage="fit")
num_samples = [len(dataset) for task_name, dataset in test_data_module.task_to_train_datasets.items()]
task_names = [task_name for task_name in test_data_module.task_to_train_datasets.keys()]

lm = ShakespeareModel(model, tokenizer, model_type, use_cpu_offload=False,
                lr=args.lr, weight_decay=args.weight_decay, max_length=args.max_length, use_wandb=args.use_wandb, 
                optimizer=args.optimizer, generate_output=args.generate_output, task_names=task_names)

if not os.path.exists("external_lightning_logs"):
    raise Exception("external_lightning_logs/ does not exist")
default_root_dir = os.path.join("external_lightning_logs", 
                            f"federated_learning_{model_key}_" + \
                            "num_tasks_{}".format(len(task_names)) + \
                            (f"_lora_r_{args.lora_rank}" if args.train_lora else "") + \
                            (f"_{args.save_name}" if args.save_name else "") )
if not os.path.exists(default_root_dir):
    os.makedirs(default_root_dir)


# %%
num_epochs = 1
batch_size_of_users = 10

from lightning_fabric.utilities.cloud_io import _load as pl_load
def save_trained_model(checkpoint_dir, save_path_dir):
    checkpoint = pl_load(checkpoint_dir, map_location=lm.device)
    state_dict = checkpoint["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if "lora" in k}
    torch.save(state_dict, save_path_dir)

def model_averaging(state_dicts, factors, scale=1.0):
    return_state_dict = {}
    factors = torch.tensor(factors)
    for key in state_dicts[0].keys():
        shape = state_dicts[0][key].shape
        return_state_dict[key] = torch.sum(
            torch.stack([state_dict[key] for state_dict in state_dicts], dim=0)*factors.view(-1, *([1]*len(shape))),
            dim=0)*scale

    return return_state_dict

# save the initial model
state_dict = model.state_dict()
state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
torch.save(state_dict, os.path.join(default_root_dir, f"epoch_0.pt"))

for epoch in range(num_epochs):
    # sampling user batches
    permuations = np.random.permutation(task_idxes)
    for idx in range(0, len(permuations), batch_size_of_users):
        batch_idxes = permuations[idx:min(idx+batch_size_of_users, len(permuations))]
        checkpoint_dirs = []
        # train single task model
        for task_idx in batch_idxes:
            # load a single task
            tmp_data_module = ShakespeareDataModule([task_idx], tokenizer, batch_size=8, inference_batch_size=8, max_input_length=80)
            tmp_data_module.setup(stage="fit")

            # load last epoch model
            print("loading the model after epoch {}".format(epoch))
            print(model.load_state_dict(torch.load(os.path.join(default_root_dir, f"epoch_{epoch}.pt")), strict=False))

            # train the model
            tmp_dir = os.path.join(default_root_dir, f"task_{task_idx}")
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            checkpoint_callback = ModelCheckpoint(
                monitor="loss",
                dirpath=tmp_dir,
                filename="step_{step}",
                save_top_k=(-1 if args.save_every_epoch else 1),
                mode="min",
                every_n_train_steps=args.epochs
            )

            trainer = pl.Trainer(accelerator="gpu", devices=args.devices, strategy=args.strategy,
                                default_root_dir=tmp_dir, min_steps=args.epochs, max_steps=args.epochs,
                                accumulate_grad_batches=args.accumulate, precision=args.precision,
                                enable_checkpointing=True,
                                callbacks=[checkpoint_callback]
                                )
            trainer.fit(lm, tmp_data_module)

            # save the model
            checkpoint_dir = os.path.join(default_root_dir, f"task_{task_idx}.pt")
            checkpoint_dirs.append(checkpoint_dir)
            save_trained_model(checkpoint_callback.best_model_path, checkpoint_dir)
            os.remove(checkpoint_callback.best_model_path)
            os.system(f"rm -r tmp_dir")

        # average the models in this batch
        state_dicts = []
        for checkpoint_dir in checkpoint_dirs:
            checkpoint = torch.load(checkpoint_dir, map_location="cpu")
            state_dicts.append(checkpoint)

        factors = np.array([num_samples[task_idx] for task_idx in batch_idxes])
        factors = factors/np.sum(factors)
        state_dict = model_averaging(state_dicts, factors)
        torch.save(model.state_dict(), os.path.join(default_root_dir, f"epoch_{epoch+1}.pt"))

        print("loading the model after epoch {}".format(epoch+1))
        print(model.load_state_dict(state_dict, strict=False))
        
    # test the model
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
                        enable_checkpointing=True,
                        callbacks=[checkpoint_callback]
                        )
    trainer.validate(lm, dataloaders=test_data_module.val_dataloader())
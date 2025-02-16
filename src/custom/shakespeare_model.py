import copy
import json
import logging
from typing import List, Dict
import wandb
from collections import defaultdict
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
# from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import PreTrainedTokenizerBase
import numpy as np
from utils.sam import SAM
import os
from utils.compute_metrics import compute_metrics
from sklearn.metrics import accuracy_score, f1_score

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(1)
    info = nvmlDeviceGetMemoryInfo(handle)
    # print(info.used, info.total)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

class ShakespeareModel(pl.LightningModule):
    validation_predictions: Dict

    def __init__(self, model, tokenizer: PreTrainedTokenizerBase, model_type: str, use_cpu_offload=False,
                lr=3e-4, truncate_early=True, max_length=1024, weight_decay=1e-4, use_wandb=False,
                optimizer="adamw", generate_output=False, task_names=[],
                use_sample_weights=False, fit_least_square = False, compute_gradients = False,
                compute_gradients_seed = 0, project_gradients_dim = 200, gradients_dir = "test", 
                compute_gradients_steps = 1e7, start_step = 0):
        """
        - completion_metadata: metaddata used to save completions. If None, completions are not saved.
          `epoch_N` is appended to the `train_key` when saving intermediate validation completions.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.lr = lr
        self.max_length = max_length
        self.truncate_early = truncate_early
        self.weight_decay = weight_decay
        self.use_wandb = use_wandb
        self.validation_step_outputs = []
        self.optimizer = optimizer
        self.generate_output = generate_output
        self.task_names = task_names
        self.use_sample_weights = use_sample_weights # AdaBoost
        self.fit_least_square = fit_least_square # Gradient Boosting

        self.compute_gradients = compute_gradients
        if compute_gradients:
            gradient_dim = 0
            self.removing_keys = ["shared", "lm_head", "wte", "wpe", "ln", "embed_tokens", "norm", "word_embeddings", "quant", "absmax"]
            for name, param in model.named_parameters():
                if any([key in name for key in self.removing_keys]):
                    continue
                if param.requires_grad:
                    gradient_dim += param.numel()
            print("Creating project matrix with dimensions: ", gradient_dim, project_gradients_dim)

            np.random.seed(compute_gradients_seed)
            self.project_matrix = (2 * np.random.randint(2, size=(gradient_dim, project_gradients_dim)) - 1).astype(float)
            self.project_matrix *= 1 / np.sqrt(project_gradients_dim)
            self.gradient_dir = f"./gradients/{gradients_dir}"
            if not os.path.exists(self.gradient_dir):
                os.makedirs(self.gradient_dir)
        self.param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        self.compute_gradients_steps = compute_gradients_steps
        self.start_step = start_step

    def get_trainable_parameters(self):
        return [param for name, param in self.model.named_parameters()\
                if (name in self.param_names) and (not any([key in name for key in self.removing_keys]))]

    def on_validation_end(self) -> None:
        if not self.automatic_optimization:
            # Save a checkpoint of the model
            ckpt_path = os.path.join(self.trainer.log_dir, 'checkpoints', 'ckpt.pt')
            self.trainer.save_checkpoint(ckpt_path)
        return super().on_validation_end()

    def training_step(self, batch, batch_idx):
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        
    
        loss = self.model(**kwargs)["loss"]
        if self.use_wandb:
            wandb.log({"train_loss": loss})
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """        
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        forward_output = self.model(**kwargs)

        if batch_idx == 1:
            print_gpu_utilization()

        output_dict = {
            "task_name": task_name,
            "loss": forward_output['loss'],
            "labels": batch["labels"],
        }
        self.validation_step_outputs.append(output_dict)
        return output_dict
    
    def predict_step(self, batch, batch_idx):
        """
        Returns outputs in dictionary format, since it's the only way that seems to work with `all_gather`
        """
        if batch_idx < self.start_step:
            return {}
        if batch_idx >= self.compute_gradients_steps:
            return {}
        if self.compute_gradients:
            torch.set_grad_enabled(True)

        # forward pass
        task_name = batch["task_name"]
        batch = batch["data"]        
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        forward_output = self.model(**kwargs)

        assert self.model_type == "decoder", "Only decoder model type is supported for prediction"
        ''' Compute the logits of the labels '''
        logits = forward_output["logits"][:, :-1].contiguous()
        labels = batch["labels"][:, 1:].contiguous()

        # obtain the outputs and gradients of the outputs
        if self.compute_gradients:
            # TODO: need to be rewritten
            gradients = []; returned_outputs = np.zeros(labels.shape)
            for i in range(len(labels)):
                tmp_mask = labels[i] != -100
                tmp_logits = logits[i][tmp_mask]
                tmp_probs = torch.softmax(tmp_logits, dim=-1)
                tmp_labels = labels[i][tmp_mask]

                tmp_outputs = tmp_probs[range(tmp_probs.size(0)), tmp_labels]
                tmp_outputs[tmp_outputs>0.9] -= 1e-2 # in case (1-tmp_outputs) is less than zero
                tmp_outputs[tmp_outputs<0.001] += 1e-2            
                tmp_outputs = torch.log(tmp_outputs/(1-tmp_outputs))
                tmp_loss = tmp_outputs.mean()

                tmp_gradients = torch.autograd.grad(tmp_loss, self.get_trainable_parameters(), retain_graph=True, create_graph=False)
                tmp_gradients = torch.cat([gradient.reshape(-1) for gradient in tmp_gradients]).cpu().type(torch.float32).numpy() # flatten gradients
                tmp_gradients = (tmp_gradients.reshape(1, -1) @ self.project_matrix).flatten()
                gradients.append(tmp_gradients)
                returned_outputs[i, :tmp_outputs.size(0)] = tmp_outputs.clone().detach().cpu().type(torch.float32).numpy()
            
            gradients = np.array(gradients)
            np.save(f"{self.gradient_dir}/train_batch_{batch_idx}_gradients.npy", gradients)
            np.save(f"{self.gradient_dir}/train_batch_{batch_idx}_outputs.npy", returned_outputs)
            forward_output['loss'].detach(); logits.detach()
            forward_output['logits'].detach()
            return {} 

        output_dict = {
            "task_name": task_name,
            "loss": forward_output['loss'],
            "labels": batch["labels"],
        }
        if "weights" in batch:
            output_dict.update({"weights": batch["weights"]})
        return output_dict
    
    def on_validation_epoch_end(self) -> None:
        """
        Gather outputs from all GPUs and save validation predictions as a CompletionDataset and
        log validation metrics.

        Note, `all_gather` *concatenates* tensors from all GPUs along the first dimension.
        """
        summary = {f"{task_name}_loss": 0 for task_name in self.task_names}
        
        # average loss
        outputs = self.validation_step_outputs
        losses = [output["loss"] for output in outputs]
        losses = torch.stack(losses)
        losses = losses[torch.isnan(losses) == False]
        summary.update({"loss": losses.mean().item()})

        task_counts = {task_name: 0 for task_name in self.task_names}
        for batch in outputs:
            task_name = batch["task_name"]
            if len(batch["labels"]) == 0:
                continue
            summary[f"{task_name}_loss"] += batch["loss"].item()*len(batch["labels"]) if torch.isnan(batch["loss"]) == False else 0
            task_counts[task_name] += len(batch["labels"])
        
        for task_name in self.task_names:
            summary[f"{task_name}_loss"] = (summary[f"{task_name}_loss"]/task_counts[task_name]) if task_counts[task_name] > 0 else 0
            
        # Log metrics
        logging.info(summary)
        if summary:
            for key, value in summary.items():
                if self.use_wandb:
                    wandb.log({f"val_{key}": value})
                self.log(key, value, prog_bar=False, logger=True)
        self.validation_step_outputs.clear()
        return summary

    def forward(self, batch, batch_idx):
        task_name = batch["task_name"]; batch = batch["data"]
        kwargs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        if self.model_type == "encoder_decoder":
            kwargs["decoder_attention_mask"] = batch["decoder_attention_mask"]
        outputs = self.model(**kwargs)
        return outputs

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            import bitsandbytes as bnb
            optimizer_dict = {
                "adamw_8bit": bnb.optim.AdamW8bit,
                "paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
                "paged_adamw_32bit": bnb.optim.PagedAdamW32bit,
            }

            optimizer = optimizer_dict[self.optimizer](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            # force embedding layers to use 32 bit for numerical stability
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1003445038
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        module, "weight", {"optim_bits": 32}
                    )
        return optimizer
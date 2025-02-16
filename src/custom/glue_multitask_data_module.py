import pytorch_lightning as pl
import torch
import os
import numpy as np
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler, IterableDataset
from transformers import DataCollatorForLanguageModeling
from transformers.data.data_collator import *
from torch.utils.data import BatchSampler

import glob
import tqdm
import random

from utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from custom.glue_task_constants import task_to_benchmark, task_to_instruction_template, task_is_generative_task
from utils.template_utils import apply_template
from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

@dataclass
class Seq2SeqInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch
        # prepare input sources
        sources = []
        for instance in converted_batch:
            source = instance["input"]
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
        model_inputs = self.tokenizer(
                sources, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of)
        
        # prepare labels
        labels = [instance["output"] for instance in converted_batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                labels,
                max_length=self.max_target_length,
                padding=self.padding,
                return_tensors=self.return_tensors,
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of
            )
        label_mask = labels["attention_mask"].bool()
        model_inputs["labels"] = labels["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)

        return model_inputs

@dataclass
class CasualLMInstructionCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_source_length: Optional[int] = None # maximum length of the output
    max_target_length: Optional[int] = None # maximum length of the input
    pad_to_multiple_of: Optional[int] = None 
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, batch, return_tensors=None):

        if return_tensors is None:
                return_tensors = self.return_tensors

        converted_batch = batch

        # prepare input sources
        sources = []; source_lengths = []
        for instance in converted_batch:
            source = instance["input"]
            source = source.replace("\n", " ")
            source = " ".join(source.split())
            tokenized_source = self.tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.max_source_length:
                sources.append(source)
            else:
                sources.append(self.tokenizer.decode(tokenized_source[:self.max_source_length], skip_special_tokens=True))
            source_lengths.append(min(len(tokenized_source), self.max_source_length))

        labels = []; label_lengths = []
        for instance in converted_batch:
            label = instance["output"]
            label = label.replace("\n", " ")
            label = " ".join(label.split())
            tokenized_label = self.tokenizer(label)["input_ids"]
            if len(tokenized_label) <= self.max_target_length:
                labels.append(label)
            else:
                labels.append(self.tokenizer.decode(tokenized_label[:self.max_target_length], skip_special_tokens=True))
            label_lengths.append(min(len(tokenized_label), self.max_target_length))

        inputs = [source + " " + label for source, label in zip(sources, labels)]

        model_inputs = self.tokenizer(
                text = inputs, 
                max_length=self.max_source_length, 
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True)
        
        # prepare labels
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        label_mask = model_inputs["attention_mask"].clone().bool()
        model_inputs["labels"] = model_inputs["labels"].masked_fill(~label_mask, self.label_pad_token_id)
        for i, length in enumerate(source_lengths):
            model_inputs["labels"][i, :length] = self.label_pad_token_id            

        if "weights" in converted_batch[0]:
            model_inputs["weights"] = torch.Tensor([instance["weights"] for instance in converted_batch])

        if "residuals" in converted_batch[0]:
            model_inputs["residuals"] = torch.Tensor([instance["residuals"] for instance in converted_batch])
        
        return model_inputs

class GLUEMultitaskDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        task_names,
        tokenizer,
        batch_size= 8,
        inference_batch_size=32,
        max_input_length=512,
        max_output_length=4, # deprecated
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0
    ):
        super().__init__()

        self.task_names = task_names
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation

        '''
        Add this to pytorch lightning "pytorch_lightning/utilities/data.py" line 282:
        if hasattr(batch_sampler, "_task_to_datasets"):
            batch_sampler = batch_sampler_cls(
                sampler,
                batch_size=batch_sampler.batch_size,
                drop_last=(False if is_predicting else batch_sampler.drop_last),
                task_to_datasets=batch_sampler._task_to_datasets, shuffle=False
            )
        else:
            batch_sampler = batch_sampler_cls(
                sampler,
                batch_size=batch_sampler.batch_size,
                drop_last=(False if is_predicting else batch_sampler.drop_last),
            )
        '''

        '''
        try:
            if hasattr(batch_sampler, "_task_to_datasets"):
                batch_sampler = batch_sampler_cls(
                    sampler,
                    batch_size=batch_sampler.batch_size,
                    drop_last=(False if is_predicting else batch_sampler.drop_last),
                    task_to_datasets=batch_sampler._task_to_datasets, shuffle=False
                )
            else:
                batch_sampler = batch_sampler_cls(
                    sampler,
                    batch_size=batch_sampler.batch_size,
                    drop_last=(False if is_predicting else batch_sampler.drop_last),
                )
        '''

    def setup(self, stage=None):
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        for task_name in self.task_names:
            do_eval = True
            do_predict = True
            benchmark_name = task_to_benchmark[task_name] # Test set does not have labels

            # load task and original template
            if benchmark_name is not None:
                if benchmark_name == "story_cloze":
                    print(f"loading dataset {benchmark_name}/{task_name}....")
                    raw_datasets = load_dataset(benchmark_name, "2016", trust_remote_code=True,
                                                data_dir="./data/story_cloze/")
                    dataset_templates = DatasetTemplates(benchmark_name, "2016")
                    print(f"{benchmark_name}/{task_name} loading completed!")
                elif benchmark_name == "anli":
                    print(f"loading dataset {benchmark_name}/{task_name}....")
                    raw_datasets = load_dataset(benchmark_name, trust_remote_code=True)
                    dataset_templates = DatasetTemplates("anli")
                    print(f"{benchmark_name}/{task_name} loading completed!")
                else:
                    print(f"loading dataset {benchmark_name}/{task_name}....")
                    raw_datasets = load_dataset(benchmark_name, task_name, trust_remote_code=True)
                    dataset_templates = DatasetTemplates(benchmark_name, task_name)
                    print(f"{benchmark_name}/{task_name} loading completed!")
            else:
                print(f"loading dataset {task_name}....")
                raw_datasets = load_dataset(task_name, trust_remote_code=True)
                dataset_templates = DatasetTemplates(task_name)
                print(f"{task_name} loading completed!")

            keys = list(dataset_templates.name_to_id_mapping.keys())
            if task_is_generative_task[task_name]:
                templates = [dataset_templates[key] for key in keys if 'ROUGE' in dataset_templates[key].metadata.metrics]
            else:
                templates = [dataset_templates[key] for key in keys]

            # load the first template with answer choices
            i = 0
            template = templates[i]
            while not template.answer_choices and (i + 1) < len(templates):
                i += 1
                template = templates[i]

            # unifying labels space
            template = Template(name="defined", jinja=task_to_instruction_template[task_name], reference="", answer_choices = template.answer_choices)
            if task_name == "copa":
                template.answer_choices = "First ||| Second"
            elif task_name == "hellaswag":
                template.answer_choices = "A ||| B ||| C ||| D"
            elif task_name == "winogrande_debiased":
                template.answer_choices = "A ||| B"
            elif task_name == "story_cloze":
                template.answer_choices = "A ||| B"
            elif "anli" in task_name:
                template.answer_choices = "Yes ||| Maybe ||| No"
            self.task_to_templates[task_name] = template


            ''' Loading splits '''
            if "anli" in task_name:            
                versioon = task_name.split("_")[-1]
                train_dataset = raw_datasets[f'train_{versioon}'] # raw_dataset.select(permutation[:int(0.8*len(train_dataset))])
                eval_dataset = raw_datasets[f'dev_{versioon}']
                predict_dataset = raw_datasets[f'test_{versioon}']
            elif task_name == "story_cloze":
                train_dataset = raw_datasets['validation']
                eval_dataset = raw_datasets['test']
                predict_dataset = raw_datasets['test']
            else:
                if "train" not in raw_datasets:
                    raise ValueError("Requires a train dataset")
                train_dataset = raw_datasets["train"]

                valid_name = "validation_matched" if task_name == "mnli" else "validation"
                test_name = "test_matched" if task_name == "mnli" else "test"
                if do_eval:
                    if valid_name not in raw_datasets:
                        print("No validation dataset, using test set")
                        if test_name not in raw_datasets:
                            raise ValueError("--do_eval requires a validation dataset")
                        eval_dataset = raw_datasets[test_name]
                    else:
                        eval_dataset = raw_datasets[valid_name] 
                else:
                    eval_dataset = []

                if do_predict:
                    predict_dataset = raw_datasets[valid_name]
                    # if test_name not in raw_datasets:
                    #     print("No test dataset, using validation set")
                    #     if valid_name not in raw_datasets:
                    #         raise ValueError("--do_predict requires a test dataset")
                    #     predict_dataset = raw_datasets[valid_name]
                    # else:
                    #     predict_dataset = raw_datasets[test_name] 
                else:
                    predict_dataset = []

            # Prepare validation and test dataset
            column_names = train_dataset.column_names
            if "input" in column_names:
                column_names.remove("input")
                train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
                train_dataset.remove_columns(["old_input"])
            else:
                train_dataset = train_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

            if self.downsample_rate < 1.0:
                rng = np.random.default_rng(self.downsample_seed)
                permutations = rng.permutation(len(train_dataset))
                min_sample = max(self.minimum_sample, int(self.downsample_rate*len(train_dataset)))
                train_dataset = train_dataset.select(permutations[:min_sample])

            if do_eval:
                column_names = eval_dataset.column_names
                if "input" in column_names:
                    column_names.remove("input")
                    eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
                    eval_dataset.remove_columns(["old_input"])
                else:
                    eval_dataset = eval_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

                if self.downsample_rate < 1.0:
                    rng = np.random.default_rng(self.downsample_seed)
                    permutations = rng.permutation(len(eval_dataset))
                    min_sample = max(self.minimum_sample_validation, int(self.downsample_rate*len(eval_dataset)))
                    eval_dataset = eval_dataset.select(permutations[:min_sample])

            if do_predict:
                column_names = predict_dataset.column_names
                if "input" in column_names:
                    column_names.remove("input")
                    predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)
                    predict_dataset.remove_columns(["old_input"])
                else:
                    predict_dataset = predict_dataset.map(apply_template(template), batched=True, remove_columns=column_names)

                if self.downsample_rate < 1.0:
                    rng = np.random.default_rng(self.downsample_seed)
                    permutations = rng.permutation(len(predict_dataset))
                    min_sample = max(self.minimum_sample_validation, int(self.downsample_rate*len(predict_dataset)))
                    predict_dataset = predict_dataset.select(permutations[:min_sample])


            print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))

            self.task_to_train_datasets[task_name] = train_dataset
            self.task_to_valid_datasets[task_name] = eval_dataset
            self.task_to_test_datasets[task_name] = predict_dataset
            self.task_to_collators[task_name] = CasualLMInstructionCollator(self.tokenizer, padding="max_length", 
                                                    max_source_length=self.max_input_length, max_target_length=self.max_output_length)

        self.multitask_train_dataset = MultitaskDataset(self.task_to_train_datasets)
        self.multitask_valid_dataset = MultitaskDataset(self.task_to_valid_datasets)
        self.multitask_test_dataset = MultitaskDataset(self.task_to_test_datasets)
        self.multitask_collator = MultitaskCollator(self.task_to_collators)
        self.multitask_train_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_train_datasets.values()])), 
                                                                batch_size=self.batch_size, drop_last=False, task_to_datasets=self.task_to_train_datasets, shuffle=True)
            # self.task_to_train_datasets, self.batch_size, shuffle=True)
        self.multitask_valid_sampler = MultitaskBatchSampler(sampler=np.arange(sum([len(dataset) for dataset in self.task_to_valid_datasets.values()])), 
                                                                batch_size=self.inference_batch_size, drop_last=False, task_to_datasets=self.task_to_valid_datasets, shuffle=False)
            # self.task_to_valid_datasets, self.inference_batch_size, shuffle=False)

        if hasattr(self, "residuals") and hasattr(self, "weights"):
            cur_len = 0
            for task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)]) # add weights to train dataset
                cur_len += len(train_dataset)

            cur_len = 0
            for task_name, train_dataset in self.task_to_train_datasets.items():
                self.task_to_train_datasets[task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
                cur_len += len(train_dataset)

            print("Weights and residuals loaded!", "Weights mean: ", self.weights.mean(), "Residuals mean: ", self.residuals.mean())

    def initialize_weights_and_residuals(self):
        # initialize weights
        self.weights = np.ones(sum([len(train_dataset) for train_dataset in self.task_to_train_datasets.values()]))
        self.sum_of_weights = self.weights.sum() # weights should sum to the number of examples
        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            self.task_to_train_datasets[task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)]) # add weights to train dataset
            cur_len += len(train_dataset)

        # initialize residuals 
        self.residuals = np.zeros(sum([len(train_dataset) for train_dataset in self.task_to_train_datasets.values()]))
        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            self.task_to_train_datasets[task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
            cur_len += len(train_dataset)
    
    def update_weights(self, alpha, masks):
        self.weights[masks] = np.exp(alpha*self.weights[masks])
        self.weights = self.weights*self.sum_of_weights/self.weights.sum()

        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            train_dataset = train_dataset.remove_columns("weights")
            self.task_to_train_datasets[task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)])
            cur_len += len(train_dataset)

    def load_weights(self, weights):
        self.weights = weights
        self.weights = self.weights*self.sum_of_weights/self.weights.sum()
        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            train_dataset = train_dataset.remove_columns("weights")
            self.task_to_train_datasets[task_name] = train_dataset.add_column("weights", self.weights[cur_len: cur_len+len(train_dataset)])
            cur_len += len(train_dataset)

    def update_residuals(self, residuals, masks):
        self.residuals = residuals

        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            train_dataset = train_dataset.remove_columns("residuals")
            self.task_to_train_datasets[task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
            cur_len += len(train_dataset)

    def load_residuals(self, residuals):
        self.residuals = residuals

        cur_len = 0
        for task_name, train_dataset in self.task_to_train_datasets.items():
            train_dataset = train_dataset.remove_columns("residuals")
            self.task_to_train_datasets[task_name] = train_dataset.add_column("residuals", self.residuals[cur_len: cur_len+len(train_dataset)])
            cur_len += len(train_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.multitask_train_dataset,
            batch_sampler=self.multitask_train_sampler,
            collate_fn=self.multitask_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.multitask_valid_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.multitask_test_dataset,
            batch_sampler=self.multitask_valid_sampler,
            collate_fn=self.multitask_collator,
        )
        
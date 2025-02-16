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

from utils.multitask_dataset import MultitaskDataset, MultitaskBatchSampler, MultitaskCollator
from custom.glue_task_constants import task_to_benchmark, task_to_instruction_template, task_is_generative_task
from utils.template_utils import apply_template
from datasets import load_dataset
from promptsource.templates import DatasetTemplates, Template

import torch 
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

class UserDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        return {
            "input_ids": data['input_ids'][0],
            "attention_mask": data['attention_mask'][0],
        } 

class StringDataCollator(DataCollatorForLanguageModeling):
    """Custom data collator for samples with string data in addition to tensors."""
    def __init__(self, tokenizer, string_columns=[], mlm=False):
        super().__init__(tokenizer, mlm)
        self.string_columns = string_columns

    def __call__(self, examples):
        tensor_examples = [{k: v for k,v in ex.items() if k not in self.string_columns} for ex in examples]
        string_examples = [{k: v for k,v in ex.items() if k in self.string_columns} for ex in examples]
        batch = super().__call__(tensor_examples)
        # not in use
        counts = [len(s) for s in string_examples]
        if sum(counts) != 0:
            for col in self.string_columns:
                if col in string_examples[0]: # check that the string_column exists
                    batch[col] = [ex[col] for ex in string_examples]
        return batch

class ShakespeareDataModule(pl.LightningDataModule):
        
    ''' 
    660: Total train size: 46310 Total validation size: 4802
    500: Total train size: 35367 Total validation size: 3671
    300: Total train size: 21933 Total validation size: 2278
    200: Total train size: 15162 Total validation size: 1580
    100: Total train size: 7927 Total validation size: 829
    '''

    def __init__(
        self,
        task_indexes, # take the first-k tasks from the dataset
        tokenizer,
        batch_size= 8,
        inference_batch_size=32,
        max_input_length=512,
        downsample_ratio=1.0, # ratio of downsampling
        minimum_samples=100,
        minimum_samples_validation=100,
        downsample_seed=0
    ):
        super().__init__()

        self.task_indexes = task_indexes
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        if inference_batch_size is None:
            self.inference_batch_size = batch_size
        else:
            self.inference_batch_size = inference_batch_size
        self.downsample_rate = downsample_ratio
        self.downsample_seed = downsample_seed
        self.minimum_sample = minimum_samples
        self.minimum_sample_validation = minimum_samples_validation

        self.train_data_dir = "./data/shakespeare/data/train/all_data_niid_0_keep_0_train_9.json"
        self.test_data_dir = "./data/shakespeare/data/test/all_data_niid_0_keep_0_test_9.json"

    def setup(self, stage=None):
        ''' load the whole data '''
        with open(self.train_data_dir) as json_file:
            train_data = json.load(json_file)
        with open(self.test_data_dir) as json_file:
            test_data = json.load(json_file)  
        # get the first k tasks      
        self.task_names = [train_data["users"][i] for i in self.task_indexes]
        print("There are {} tasks, taking {} tasks".format(len(train_data["users"]), len(self.task_names)))
        
        ''' load the whole data '''
        self.task_to_train_datasets = {}
        self.task_to_valid_datasets = {}
        self.task_to_test_datasets = {}
        self.task_to_collators = {}
        self.task_to_templates = {}
        total_train_size = 0; total_valid_size = 0
        for task_name in self.task_names:
            user_train_data = train_data["user_data"][task_name]
            user_test_data = test_data["user_data"][task_name]

            # only take the whole sentences 
            sentence_len = len(train_data["user_data"][task_name]['x'][0])
            tmp_train_data = [] 
            for idx in np.arange(0, len(user_train_data['x']), sentence_len):
                tmp_train_data.append(user_train_data['x'][idx])

            tmp_test_data = []
            for idx in np.arange(0, len(user_test_data['x']), sentence_len):
                tmp_test_data.append(user_test_data['x'][idx])

            tokenized_train_data =[self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.max_input_length, truncation=True)\
                        for text in tmp_train_data]
            tokenized_test_data =[self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.max_input_length, truncation=True)\
                        for text in tmp_test_data]
            
            train_dataset = UserDataset(tokenized_train_data)
            eval_dataset = UserDataset(tokenized_test_data)

            # if self.downsample_rate < 1.0:
            #     rng = np.random.default_rng(self.downsample_seed)
            #     permutations = rng.permutation(len(train_dataset))
            #     min_sample = max(self.minimum_sample, int(self.downsample_rate*len(train_dataset)))
            #     train_dataset = train_dataset.select(permutations[:min_sample])

            
            # if self.downsample_rate < 1.0:
            #     rng = np.random.default_rng(self.downsample_seed)
            #     permutations = rng.permutation(len(eval_dataset))
            #     min_sample = max(self.minimum_sample_validation, int(self.downsample_rate*len(eval_dataset)))
            #     eval_dataset = eval_dataset.select(permutations[:min_sample])
            
            predict_dataset = eval_dataset

            print("Task: {} train dataset size: {} validation dataset size: {} test dataset size: {}".format(task_name, len(train_dataset), len(eval_dataset), len(predict_dataset)))

            total_train_size += len(train_dataset); total_valid_size += len(eval_dataset)
            self.task_to_train_datasets[task_name] = train_dataset
            self.task_to_valid_datasets[task_name] = eval_dataset
            self.task_to_test_datasets[task_name] = predict_dataset
            self.task_to_collators[task_name] = StringDataCollator(self.tokenizer)
        print("Total train size: {} Total validation size: {}".format(total_train_size, total_valid_size))

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
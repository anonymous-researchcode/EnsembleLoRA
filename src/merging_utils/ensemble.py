from typing import Dict, List, cast

import os
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def aggregate_tensors(outputs, aggregate_fn) -> Tensor:
    # If the output is a Tensor, take the mean
    if isinstance(outputs[0], torch.Tensor):
        return aggregate_fn(outputs)

    # If the output is a dict, take the mean of each value
    elif isinstance(outputs[0], Dict):
        result = type(outputs[0])()
        for key in outputs[0]:
            result[key] = aggregate_tensors(
                [output[key] for output in outputs], aggregate_fn
            )
        return result

    # If the output is a tuple or list, take the mean of each element
    elif isinstance(outputs[0], (tuple, list)):
        return tuple(
            aggregate_tensors([output[i] for output in outputs], aggregate_fn)
            for i in range(len(outputs[0]))
        )

    # If the output is None, return None
    elif all(output is None for output in outputs):
        return None

    # If the output is none of the above, return as is
    else:
        raise ValueError("Unsupported type for outputs")

class EnsembleAdapterModule(nn.Module):
    def __init__(self, base_model, adapter_list=[], weights=[], use_bias=False):
        super().__init__()
        self.base_model = base_model
        self.dummy_adapter = nn.ModuleDict()
        self.adapter_list = nn.ModuleList([])
        self.use_bias = use_bias

        # store one set of weights for the original base model
        for name, module in self.base_model.named_modules():
            if ("adapter_up" in name or "adapter_down" in name) and isinstance(module, nn.Linear):
                self.dummy_adapter[name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=self.use_bias)    
        
        for i, adapter in enumerate(adapter_list):
            if os.path.exists(adapter):
                state_dict = torch.load(adapter, map_location=self.base_model.device)
            else:
                state_dict = None
            self.adapter_list.append(nn.ModuleDict())
            for name, module in self.base_model.named_modules():
                if ("adapter_up" in name or "adapter_down" in name) and isinstance(module, nn.Linear):
                    self.adapter_list[i][name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=self.use_bias)    
                    if state_dict is not None:
                        self.adapter_list[i][name.replace(".", "__")].weight = nn.Parameter(state_dict[name + ".weight"].detach().clone().float())   
                        if self.use_bias:
                            self.adapter_list[i][name.replace(".", "__")].bias = nn.Parameter(state_dict[name + ".bias"].detach().clone().float()) 
        self.weights = weights

    def _aggregate_tensors(self, outputs):
        weights = torch.Tensor(self.weights).to(self.base_model.device)
        weights = cast(Tensor, weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)
    
    def add_adapter(self, adapter, weight):
        idx = len(self.adapter_list)
        if os.path.exists(adapter):
            state_dict = torch.load(adapter, map_location=self.base_model.device)
        else:
            print(f"Adapter {adapter} does not exist! Creating a random initialized adapter")
            state_dict = None
        self.adapter_list.append(nn.ModuleDict())
        for name, module in self.base_model.named_modules():
            if ("adapter_up" in name or "adapter_down" in name) and isinstance(module, nn.Linear):
                self.adapter_list[idx][name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=self.use_bias)
                if state_dict is not None:    
                    self.adapter_list[idx][name.replace(".", "__")].weight = nn.Parameter(state_dict[name + ".weight"].detach().clone().float())    
                    if self.use_bias:
                            self.adapter_list[i][name.replace(".", "__")].bias = nn.Parameter(state_dict[name + ".bias"].detach().clone().float())
        self.weights.append(weight)

    def _assign_weights(self, adapter_idx):
        for name, module in self.base_model.named_modules():
            if ("adapter_up" in name or "adapter_down" in name) and isinstance(module, nn.Linear):
                module.weight = self.adapter_list[adapter_idx][name.replace(".", "__")].weight 
                if self.use_bias:
                    module.bias = self.adapter_list[adapter_idx][name.replace(".", "__")].bias

    def _revert_weights(self):
        for name, module in self.base_model.named_modules():
            if ("adapter_up" in name or "adapter_down" in name) and isinstance(module, nn.Linear):
                module.weight = self.dummy_adapter[name.replace(".", "__")].weight
                if self.use_bias:
                    module.bias = self.dummy_adapter[name.replace(".", "__")].bias

    def forward(self, *args, **kwargs):
        outputs = []
        for adapter_idx, adapter in enumerate(self.adapter_list):
            self._assign_weights(adapter_idx)
            outputs.append(self.base_model(*args, **kwargs))

        # revert the weights back to the original base model
        self._revert_weights()
        return aggregate_tensors(outputs, self._aggregate_tensors)


class EnsembleLoRAModule(nn.Module):
    def __init__(self, base_model, adapter_list=[], weights=[]):
        super().__init__()
        self.base_model = base_model
        self.dummy_adapter = nn.ModuleDict()
        self.adapter_list = nn.ModuleList([])

        # store one set of weights for the original base model
        for name, module in self.base_model.named_modules():
            if "lora" in name and isinstance(module, nn.Linear):
                self.dummy_adapter[name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=False)    
        
        for i, adapter in enumerate(adapter_list):
            if os.path.exists(adapter):
                state_dict = torch.load(adapter, map_location=self.base_model.device)
            else:
                state_dict = None
            self.adapter_list.append(nn.ModuleDict())
            for name, module in self.base_model.named_modules():
                if "lora" in name and isinstance(module, nn.Linear):
                    self.adapter_list[i][name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=False)    
                    if state_dict is not None:
                        self.adapter_list[i][name.replace(".", "__")].weight = nn.Parameter(state_dict[name + ".weight"].detach().clone().float())    
        self.weights = weights

    def _aggregate_tensors(self, outputs):
        weights = torch.Tensor(self.weights).to(self.base_model.device)
        weights = cast(Tensor, weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)
    
    def add_adapter(self, adapter, weight):
        idx = len(self.adapter_list)
        if os.path.exists(adapter):
            state_dict = torch.load(adapter, map_location=self.base_model.device)
        else:
            print(f"Adapter {adapter} does not exist! Creating a random initialized adapter")
            state_dict = None
        self.adapter_list.append(nn.ModuleDict())
        for name, module in self.base_model.named_modules():
            if "lora" in name and isinstance(module, nn.Linear):
                self.adapter_list[idx][name.replace(".", "__")] = nn.Linear(module.in_features, module.out_features, bias=False)
                if state_dict is not None:    
                    self.adapter_list[idx][name.replace(".", "__")].weight = nn.Parameter(state_dict[name + ".weight"].detach().clone().float())    
        self.weights.append(weight)

    def _assign_weights(self, adapter_idx):
        for name, module in self.base_model.named_modules():
            if "lora" in name and isinstance(module, nn.Linear):
                module.weight = self.adapter_list[adapter_idx][name.replace(".", "__")].weight 

    def _revert_weights(self):
        for name, module in self.base_model.named_modules():
            if "lora" in name and isinstance(module, nn.Linear):
                module.weight = self.dummy_adapter[name.replace(".", "__")].weight

    def forward(self, *args, **kwargs):
        outputs = []
        for adapter_idx, adapter in enumerate(self.adapter_list):
            self._assign_weights(adapter_idx)
            outputs.append(self.base_model(*args, **kwargs))

        # revert the weights back to the original base model
        self._revert_weights()
        return aggregate_tensors(outputs, self._aggregate_tensors)


class EnsembleModule(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        # TODO: distribute models to devices
        self.model_list = nn.ModuleList(models)

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        return torch.stack(outputs).mean(dim=0)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)


class MaxModelPredictor(EnsembleModule):
    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        return torch.stack(outputs).max(dim=0).values


class WeightedEnsembleModule(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        weights: List[float] | Tensor | np.ndarray,
        normalize: bool = False,
    ):
        super().__init__()
        self.model_list = nn.ModuleList(models)
        if isinstance(weights, (list, tuple)):
            weights = torch.tensor(weights)
        elif isinstance(weights, Tensor):
            weights = weights
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        else:
            raise ValueError(f"Unsupported type for weights: {type(weights)=}")

        assert len(models) == len(weights) and weights.dim() == 1, (
            "weights must be a 1D tensor of the same length as models."
            f"But got {len(models)=}, {weights.dim()=}"
        )
        if normalize:
            weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    def _aggregate_tensors(self, outputs: List[Tensor]) -> Tensor:
        weights = cast(Tensor, self.weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.model_list]
        return aggregate_tensors(outputs, self._aggregate_tensors)



class DictMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        assert num_hidden_layers <= 2
        self.input_dim = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        elif num_hidden_layers == 1:
            self.fc1 = nn.Identity()

        if num_hidden_layers >= 1:
            self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.constant_(self.fc2.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.ones(num_experts) * init_lambda, requires_grad=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight

        if self.num_hidden_layers == 2:
            hidden_states = F.relu(self.fc1(hidden_states))
        gate_weights = self.fc2(hidden_states)
        return gate_weights

class LearnableWeightedEnsembleModule(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        hidden_size: int,
    ):
        super().__init__()
        self.model_list = nn.ModuleList(models)

        num_experts = len(models)
        init_lambda = torch.ones(num_experts) / num_experts
        self.gate = DictMoEGate(
            hidden_size,
            num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=2,
        )

    def _aggregate_tensors(self, outputs: List[Tensor], weights) -> Tensor:
        weights = cast(Tensor, weights).view(-1, *([1] * outputs[0].dim()))
        return (torch.stack(outputs) * weights).sum(dim=0)

    def forward(self, *args, **kwargs):
        # get the last hidden states
        outputs = [model(*args, **kwargs) for model in self.model_list]
        weights = None
        return aggregate_tensors(outputs, 
                                 lambda outputs: self._aggregate_tensors(outputs, weights))
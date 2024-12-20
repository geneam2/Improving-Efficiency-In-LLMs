
import math
import warnings
from typing import List

import torch
import torch.nn as nn

from torch.nn import functional as F

warnings.simplefilter("ignore")
print(torch.__version__)

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRAAdapter(nn.Module, LoRALayer):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        num_heads: int = 0,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.existing_layer = existing_layer
        self.training_mode = False

        self.r = r
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(
                (num_heads, r, in_features), dtype=self.existing_layer.weight.dtype,
                device=self.existing_layer.weight.device
            ))
            self.lora_B = nn.Parameter(torch.zeros(
                (out_features, num_heads, r), dtype=self.existing_layer.weight.dtype,
                device=self.existing_layer.weight.device
            ))
            self.scaling = (self.lora_alpha / self.r)
            self.existing_layer.requires_grad_(False)
        self.is_merged = False
        self.reset_parameters()

    def reset_parameters(self):
        if self.r > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)
        if self.r > 0:
            if mode and self.is_merged:
                self.existing_layer.weight.data -= self.scaling * (self.lora_B @ self.lora_A)
                self.is_merged = False
            elif not mode and not self.is_merged:
                self.existing_layer.weight.data += self.scaling * (self.lora_B @ self.lora_A)
                self.is_merged = True
        self.training_mode = mode

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.is_merged:
            # training
            frozen_x = F.linear(x, self.existing_layer.weight, bias=self.existing_layer.bias)
            trainable_x = (self.lora_dropout(x) @ self.lora_A.transpose(0,1) @ self.lora_B.transpose(0,1)) * self.scaling
            return frozen_x + trainable_x
        else:
            return F.linear(x, self.existing_layer.weight, bias=self.existing_layer.bias)

def match_submodules(model: nn.Module, key:str) -> List[str]:
    ret = []
    for name, params in model.named_parameters():
        module = [i for i in name.split(".") if i in key]
        if "weight" in name and len(module) == 1:
            module = module[0]
            name = name.replace(".weight", "")
            ret.append(
                (name.replace("."+module, ""), module)
            )
    return ret

def get_submodule(model: nn.Module, module_name:str):
    return model.get_submodule(module_name)

def replace_submodule(model: nn.Module, module_path: str, new_module):
    parent_module, child_module = module_path
    submodule = get_submodule(model, parent_module)
    LoRA_submodule = new_module(
        getattr(submodule, child_module)
    ).to(model.device)
    setattr(submodule, child_module, LoRA_submodule)

def inject_adapter(model: nn.Module, match_on: List[str], adapter_fn):
    submodules = []
    for match_ in match_on:
        submodules.extend(match_submodules(model, match_))
    # print("submodules", submodules)
    for submodule in submodules:
        replace_submodule(model, submodule, adapter_fn)

def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, params in model.named_parameters():
        params.requires_grad = bool("lora" in name)

"""
inject_adapter(causal_model, ["q_proj_k_proj_v_proj"], lambda x: LoRAAdapter(x, r=8, lora_alpha=8, in_features=x.in_features, out_features=x.out_features))
mark_only_lora_as_trainable(causal_model)

trainable_params = total_params = 0
for _, params in causal_model.named_parameters():
    total_params += params.numel()
    if params.requires_grad:
        trainable_params += params.numel()

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")
"""
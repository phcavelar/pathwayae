from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
            self,
            input_dim:int,
            hidden_dims:list[int],
            output_dim:int,
            nonlinearity:Callable,
            dropout_rate:float=0.5,
            bias:bool=True,
            ):
        super().__init__()
        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [output_dim]
        
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out, bias=bias) for d_in, d_out in zip(in_dims, out_dims)])
        self.nonlinearity = nonlinearity
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.dropout(self.nonlinearity(layer(x)))
        return self.layers[-1](x)
    
    def layer_activations(self, x:torch.Tensor) -> list[torch.Tensor]:
        # To allow for activation normalisation
        activations = [x]
        for layer in self.layers[:-1]:
            activations.append(self.dropout(self.nonlinearity(layer(activations[-1]))))
        return activations[1:] + [self.layers[-1](activations[-1])]

class NopLayer(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs,
            ):
        super().__init__()
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x
    
    def update_temperature(self,*args,**kwargs) -> None:
        pass

    def layer_activations(self,*args,**kwargs) -> list[torch.Tensor]:
        return []
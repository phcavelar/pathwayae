import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import *
from .._types import VAEOutput

class VAE(nn.Module):
    def __init__(
            self,
            input_dim:int=None,
            hidden_dims:list[int]=[128],
            encoding_dim:int=64,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate:float=0.5,
            bias:bool=True,
            eps:float=1e-6,
            ):
        super().__init__()
        if input_dim is None:
            raise ValueError("Must specify input dimension before initialising the model")
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        mean_logvar_encoding_dim = 2*encoding_dim
        
        self.encoder = MLP(input_dim, hidden_dims, mean_logvar_encoding_dim, nonlinearity, dropout_rate, bias)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], input_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity
        self.eps = eps
    
    def encode(self,x:torch.Tensor) -> torch.Tensor:
        mean_and_logvar = self.encoder(x)
        mean, logvar = mean_and_logvar[...,:mean_and_logvar.shape[-1]//2], mean_and_logvar[...,mean_and_logvar.shape[-1]//2:]
        return mean, logvar

    def reparameterize(self, mean:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar)
        eps = torch.randn_like(mean, device=mean.device)
        z = mean + eps*logvar
        return z
    
    def decode(self,x:torch.Tensor) -> torch.Tensor:
        return self.final_nonlinearity(self.decoder(x))
    
    def forward(self, x:torch.Tensor) -> VAEOutput:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return mean, self.decode(z), (z, mean, logvar)
    
    def layer_activations(self,x:torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError()
    
    def get_feature_importance_matrix(self) -> torch.Tensor:
        raise NotImplementedError()
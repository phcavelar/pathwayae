import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import *

class PAVAE(nn.Module):
    def __init__(
            self,
            genes_dim:int,
            pathway_definitions:list[torch.LongTensor],
            hidden_dims:list[int]=[128],
            encoding_dim:int=64,
            nonlinearity=F.relu,
            dropout_rate:float=0.5,
            bias:bool=True,
            final_nonlinearity=lambda x:x,
            pathway_hidden_dims:list[int] = [],
            pathway_nonlinearities=F.relu,
            pathway_activity_nonlinearity=torch.tanh,
            pathway_dropout_rate:int=0,
            pathway_bias:bool=True,
            ):
        super().__init__()
        if genes_dim is None:
            raise ValueError("Must specify genes dimension before initialising the model")
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]

        input_dim = len(pathway_definitions)
        mean_logvar_encoding_dim = 2*encoding_dim
        
        self.pathway_definitions = pathway_definitions

        encoder_list = [
            MLP(p.shape[0], pathway_hidden_dims, 1, pathway_nonlinearities, pathway_dropout_rate, pathway_bias) for p in pathway_definitions
        ]

        self.pathway_encoders = nn.ModuleList(encoder_list)
        self.encoder = MLP(input_dim, hidden_dims, mean_logvar_encoding_dim, nonlinearity, dropout_rate, bias)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], genes_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity
        self.pathway_activity_nonlinearity = pathway_activity_nonlinearity
    
    def get_pathway_activities(self,x:torch.Tensor) -> torch.Tensor:
        return torch.concat(
            [
                self.pathway_activity_nonlinearity(enc(x[:,pway]))
                for pway, enc in
                    zip(self.pathway_definitions, self.pathway_encoders)
            ],
            dim=-1,
        )
    
    def encode_pathways(self,a:torch.Tensor) -> torch.Tensor:
        mean_and_logvar = self.encoder(a)
        mean, logvar = mean_and_logvar[...,:mean_and_logvar.shape[-1]//2], mean_and_logvar[...,mean_and_logvar.shape[-1]//2:]
        return mean, logvar

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        a = self.get_pathway_activities(x)
        mean, logvar = self.encode_pathways(a)
        return mean, logvar

    def reparameterize(self, mean:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar)
        eps = torch.randn_like(mean, device=mean.device)
        z = mean + eps*logvar
        return z
    
    def decode(self,z:torch.Tensor) -> torch.Tensor:
        return self.final_nonlinearity(self.decoder(z))
    
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return z, self.decode(z), (z, mean, logvar) # If you change the order here, remember to change it on forward as well
    
    def layer_activations(self,x:torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError()
    
    def get_feature_importance_matrix(self) -> torch.Tensor:
        raise NotImplementedError()

class PAVAE_parallel(PAVAE):
    def get_pathway_activities(self,x:torch.Tensor) -> torch.Tensor:
        futures = [
            torch.jit.fork(lambda enc, pway: self.pathway_activity_nonlinearity(enc(x[:,pway])), enc, pway)
            for pway, enc in
                zip(self.pathway_definitions, self.pathway_encoders)
        ]
        return self.pathway_activity_nonlinearity(
            torch.concat(
                [
                    torch.jit.wait(f)
                    for f in futures
                ],
                dim=-1,
            )
        )
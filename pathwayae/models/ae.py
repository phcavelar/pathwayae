import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import *

class Autoencoder(nn.Module):
    def __init__(
            self,
            input_dim:int=None,
            hidden_dims:list[int]=[128],
            encoding_dim:int=64,
            nonlinearity=F.relu,
            final_nonlinearity=lambda x:x,
            dropout_rate:float=0.5,
            bias:bool=True,
            ):
        super().__init__()
        if input_dim is None:
            raise ValueError("Must specify input dimension before initialising the model")
        try:
            len(hidden_dims)
        except TypeError:
            hidden_dims = [hidden_dims]
        
        self.encoder = MLP(input_dim, hidden_dims, encoding_dim, nonlinearity, dropout_rate, bias)
        self.decoder = MLP(encoding_dim, hidden_dims[-1::-1], input_dim, nonlinearity, dropout_rate, bias)
        self.final_nonlinearity = final_nonlinearity
    
    def encode(self,x:torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self,x:torch.Tensor) -> torch.Tensor:
        return self.final_nonlinearity(self.decoder(x))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat
    
    def layer_activations(self,x:torch.Tensor) -> list[torch.Tensor]:
        # To allow for activation normalisation
        encoder_activations = self.encoder.layer_activations(x)
        decoder_activations = self.decoder.layer_activations(encoder_activations[-1])
        return encoder_activations + decoder_activations
    
    def get_feature_importance_matrix(self) -> torch.Tensor:
        with torch.no_grad():
            feature_importance_matrix = self.encoder.layers[0].weight.T
            for layer in self.encoder.layers[1:]:
                feature_importance_matrix = torch.matmul(feature_importance_matrix, layer.weight.T)
        return feature_importance_matrix.detach()
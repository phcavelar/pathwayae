from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss as Loss, MSELoss

from ._types import AEOutput, VAEOutput
from .utils import logcurve_start_end

IMPLEMENTED_KL_LOSS_TYPES = {"kingma","torch", "mc"}

class AE_MSELoss(MSELoss):
    def forward(self, output:AEOutput, target:torch.Tensor) -> torch.Tensor:
        _, x_hat = output
        return F.mse_loss(x_hat, target, reduction=self.reduction)

class VAELoss(Loss):
    def __init__(self, original_loss:Loss, size_average=None, reduce=None, reduction:str = 'mean', eps:float=1e-6, beta:float=1, beta_schedule:Optional[Callable]=None, kl_type:str="mc", kl_raise_error:bool=True,):
        super(VAELoss, self).__init__(size_average, reduce, reduction)
        assert kl_type.lower() in IMPLEMENTED_KL_LOSS_TYPES, ValueError(f"kl_type passed {kl_type} must be one of: {IMPLEMENTED_KL_LOSS_TYPES}")
        self.kl_type = kl_type.lower()
        self.kl_raise_error = kl_raise_error
        self.loss = original_loss
        self.distribution = None
        self.eps=eps
        self.num_calls = 0
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.log=False
    
    def forward(self, outputs:VAEOutput, target:torch.Tensor, iteration:int=None, update_calls:bool=True):
        rec_l, kl_l, beta_mult = self.get_losses(outputs, target, iteration=iteration, update_calls=update_calls)
        if self.log: print(self.num_calls, beta_mult, *map(lambda x: x.detach().cpu().numpy().item(), (rec_l, kl_l)))
        return rec_l + self.beta*(kl_l * beta_mult if beta_mult>0 else 0)

    """
    def __call__(self, outputs:VAEOutput, target:torch.Tensor, iteration:int=None, update_calls:bool=True):
        rec_l, kl_l, beta_mult = self.get_losses(outputs, target, iteration=iteration, update_calls=update_calls)
        if self.log: print(self.num_calls, beta_mult, *map(lambda x: x.detach().cpu().numpy().item(), (rec_l, kl_l)))
        return rec_l + self.beta*(kl_l * beta_mult if beta_mult>0 else 0)
    """

    def get_losses(self, outputs:VAEOutput, target:torch.Tensor, iteration:int=None, update_calls:bool=True):
        _, model_output, (z, mean, logvar) = outputs
        rec_l = self.loss(model_output, target)
        kl_l = reduce_pointwise(self.get_kl(z, mean, logvar), self.reduction)
        current_iter = iteration if iteration is not None else self.num_calls
        beta = 1 if self.beta_schedule is None else (self.beta_schedule(current_iter) if isinstance(self.beta_schedule, Callable) else self.beta_schedule)
        if update_calls and iteration is None: self.num_calls+=1
        return rec_l, kl_l, beta

    def get_kl(self, z:torch.Tensor, mean:torch.Tensor, logvar:torch.Tensor):
        if self.kl_type=="kingma":
            return logvar_kl_divergence(mean, logvar)
        # Losses that use a torch.distributions.Distribution object
        try:
            q = torch.distributions.Normal(mean, torch.exp(logvar).sqrt().pow(2))
        except ValueError as e:
            if self.kl_raise_error:
                raise
            q = torch.distributions.Normal(mean, torch.exp(logvar).sqrt().pow(2)+self.eps)
        if self.kl_type=="mc":
            return mc_kl_divergence(z,q)
        if self.kl_type=="torch":
            p = torch.distributions.Normal(torch.zeros_like(q.loc, device=q.loc.device), torch.ones_like(q.scale, device=q.scale.device))
            return torch.distributions.kl_divergence(p,q)
        raise ValueError("Invalide KL loss type passed")

def reduce_pointwise(loss_pointwise:torch.Tensor, reduction:str):
    if reduction == "mean":  # default
        return loss_pointwise.mean()
    if reduction == "batchmean":  # mathematically correct
        return loss_pointwise.sum() / input.size(0)
    if reduction == "sum":
        return loss_pointwise.sum()
    # reduction == "none"
    return loss_pointwise

def mc_kl_divergence(z:torch.Tensor, q:torch.distributions.Normal):
    """
    Monte Carlo KL Divergence, as explained in: https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
    """
    p = torch.distributions.Normal(torch.zeros_like(q.loc, device=q.loc.device), torch.ones_like(q.scale, device=q.scale.device))
    #q = torch.distributions.Normal(mean, std)

    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    kl = (log_qzx - log_pz)
    return kl

def logvar_kl_divergence(mean:torch.Tensor, logvar:torch.Tensor):
    """
    The KL Divergence loss function used by Kigma et al in arXiv:1312.6114
    """
    return -(1 + logvar - mean**2 - logvar.exp()).mean()/2

def gaussian_likelihood(x_hat:torch.Tensor, logscale:torch.Tensor, x:torch.Tensor):
    # Unused, got this from somewhere but can't remember where now.
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing x under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=list(range(1,len(x_hat.shape)))).mean()


def build_beta_schedule(beta_schedule_type, beta_start, **kwargs):
    if beta_schedule_type=="smooth":
        beta_duration = kwargs["beta_duration"]
        beta_schedule=lambda i: 0 if i<beta_start else logcurve_start_end(i,beta_start,beta_start+beta_duration)
    elif beta_schedule_type=="step":
        beta_schedule=lambda i: 0 if i<beta_start else 1
    else:
        raise ValueError("Please set a decent beta schedule type")
    return beta_schedule
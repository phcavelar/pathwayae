import typing
from torch import Tensor

AE_Learned_Representation = Tensor
AE_Reconstruction = Tensor

AEOutput = tuple[AE_Learned_Representation, AE_Reconstruction]
VAEOutput = tuple[AE_Learned_Representation, AE_Reconstruction, tuple[Tensor,Tensor,Tensor]]
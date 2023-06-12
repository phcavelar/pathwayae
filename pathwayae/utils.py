import warnings
import math
import numpy as np
import pandas as pd

def from_log2pk(df:pd.DataFrame|np.ndarray, k:float=1) -> pd.DataFrame|np.ndarray:
    """
    Applies exp2 and removes k (by default set to 1).
    Similar to np.expm1, but with base 2.
    No bound checks are performed.
    """
    if isinstance(df, pd.DataFrame):
        return df.apply(np.exp2)-k
    return np.exp2(df)-k

def to_log2pk(df:pd.DataFrame|np.ndarray, k:float=1) -> pd.DataFrame|np.ndarray:
    """
    Sums k (by default 1) and applies log2.
    Similar to np.logp1, but with base 2.
    No bound checks are performed to guarantee values are above 0 before performing the log2 operation, which may lead no nonfinite elements.
    """
    if isinstance(df, pd.DataFrame):
        return (df+k).apply(np.log2)
    return np.log2(df+k)

def fpkm_to_tpm(fpkm_values:pd.DataFrame|np.ndarray, gene_axis:int=1) -> np.ndarray:
    """
    Converts an array from the (non-log2p1) fpkm format to tpm by normalizing by the sum of counts in each sample.
    Gene axis is assumed to be 1, so we have one sample per row instead of one gene per row.
    """
    sample_gene_sum = np.sum(fpkm_values.values if isinstance(fpkm_values,pd.DataFrame) else fpkm_values, gene_axis, keepdims=True)
    # Avoid division by zero
    has_zero_sum:np.ndarray = sample_gene_sum==0
    if has_zero_sum.any():
        zero_indices = np.arange(len(has_zero_sum.squeeze()))[has_zero_sum.squeeze()].tolist()
        warnings.warn(f"Some of the sample gene sums were zero, setting gene sum for indices {zero_indices} as 1 to avoid division by zero.")
        sample_gene_sum = np.where(has_zero_sum, 1, sample_gene_sum)
    tpm_values = fpkm_values*(1e6/sample_gene_sum)
    return tpm_values

def sample_wise_preprocess_fn(
        X:pd.DataFrame|np.ndarray,
        input_is_log2pk:bool=True, input_k:float=1, input_type:str="fpkm",
        output_is_log2pk:bool=True, output_k:float=1, output_type:str="tpm",
        copy:bool=True, working_precision=np.float128,
        lower_clip_eps:float|None=2e-6,
        assert_finite = True,
        ) -> np.ndarray:
    """
    Helper function to transfer between normalization types using only sample information.
    This function should be safe to use before a machine learning pipeline since it should only consider information within each sample.
    By default it works with float128 precision due to some of the rounding losses due to some of the large scales involved, if one is getting out of memory errors change the precision to a lower precision.
    """
    x = X.copy() if copy else X
    x = x.astype(working_precision)

    if input_is_log2pk and input_type!=output_type:
        x = from_log2pk(x, k=input_k)
    
    if input_type!=output_type:
        if input_type in {"fpkm", "fpks", "fpku"} and output_type in {"fpkm", "fpks", "fpku", "tpm", "tps", "tpu"}:
            x = x * 1e6 if input_type[-1] == "u" else x
            x = x * (1e6/np.sqrt(x.shape[1])) if input_type[-1] == "s" else x
            x = fpkm_to_tpm(x, gene_axis=1) if output_type[:-1] == "tp" else x
            x = x/1e6 if output_type[-1] == "u" else x
            x = x*(np.sqrt(x.shape[1])/1e6) if output_type[-1] == "s" else x
        else:
            raise NotImplementedError(f"Conversion from {input_type} to {output_type} not implemented.")
    
    if output_is_log2pk and ( 
            (input_is_log2pk and input_type!=output_type)
            or
            (not input_is_log2pk)
        ):
        x = to_log2pk(x, k=output_k)

    if lower_clip_eps is not None:
        x = np.where(x<=lower_clip_eps,0,x)
    
    assert (not assert_finite) or np.isfinite(x).all(), ValueError("Some values in the array are non-finite")

    return x

def sigmoid(x:float) -> float:
    return 1/(1+math.exp(-x))

def logcurve(x:float, L:float=1, k:float=1, x_0:float=0) -> float:
    return L/(1+math.exp(-k*(x-x_0)))

def logcurve_start_end(x:float, s:float, e:float, L:float=1,) -> float:
    """A logcurve (sigmoid) function that starts and ends approximately at s and e, respectively, with its largest value being L."""
    s, e = min(s,s), max(e,e)
    width = e-s
    k = 10/width
    start = s+width/2
    return L/(1+math.exp(-k*(x-start)))
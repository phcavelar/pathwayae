# %%
import os
import warnings
import traceback
import math
import collections
import json
import functools
import typing
import itertools
import datetime
import time

# %%
from tqdm.notebook import tqdm

# %%
import numpy as np
import pandas as pd

# %%
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.metrics
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import skorch
import skorch.scoring

# %%
from pathwayae.models import MLP, Autoencoder, VAE, PAAE, PAVAE, NopLayer
from pathwayae.skorch_utils import ScoredNeuralNetAutoencoder

from pathwayae.losses import AE_MSELoss, VAELoss, build_beta_schedule

from pathwayae.utils import from_log2pk, to_log2pk, fpkm_to_tpm, sample_wise_preprocess_fn, sigmoid, logcurve, logcurve_start_end

from pathwayae.pathway_utils import read_pathway_from_json_file

# %% [markdown]
# ## GEx

# %%
data_folder = os.path.expanduser("~/data/") 
tcga_folder = os.path.join(data_folder, "pathwayae", "tcga")
os.makedirs(tcga_folder, exist_ok=True)

# %%
cancer_type = "BRCA"
gex_dfs = []
gex = pd.read_csv(os.path.join(tcga_folder, f"TCGA-{cancer_type}.htseq_fpkm.tsv.gz"), sep="\t", index_col="Ensembl_ID").T.dropna(axis="columns")

with open(os.path.join(tcga_folder, "ensembl_to_gene_id.json")) as f:
    ensembl_to_gex_dict = json.load(f)

columns_to_drop = [k for k in ensembl_to_gex_dict if ensembl_to_gex_dict[k]==""]

gex = gex.drop(columns=columns_to_drop)

gex = gex.rename(columns=ensembl_to_gex_dict)

gex.columns.rename("GeneName", inplace=True)
gex.index.rename("SampleID", inplace=True)
gex.index

# %%
ensembl_fpkm_phenotype_tsv_fname = os.path.join(tcga_folder, f"TCGA.{cancer_type}.sampleMap_{cancer_type}_clinicalMatrix")
phenotype = pd.read_csv(ensembl_fpkm_phenotype_tsv_fname, sep="\t", index_col="sampleID")
phenotype.index = phenotype.index.rename("SampleID")
phenotype = phenotype[[c for c in phenotype.columns if "pam50" in c.lower()]]
for c in phenotype.columns:
    print(phenotype[c].count(), phenotype[c].value_counts())
phenotype

# %%
cancer_phenotype_tgt_target = {
    "BRCA": "PAM50Call_RNAseq"
}

cancer_phenotype_tgt_mapping = {
    "BRCA": {
        "LumA":0,
        "LumB":1,
        "Basal":2,
        "Normal":3,
        "Her2":4,
    }
}

cancer_phenotype_tgt_na_value = {
    "BRCA": {f"{value}":np.nan for value in [np.nan, "not reported", ""]}
}

phenotype_clf_tgt = cancer_phenotype_tgt_target[cancer_type]
phenotype_clf_map = cancer_phenotype_tgt_mapping[cancer_type]
phenotype_clf_nan = cancer_phenotype_tgt_na_value[cancer_type]

# %%
# Drop nan
PHENOTYPE_CLF_COLUMN = "subtype"
phenotype[PHENOTYPE_CLF_COLUMN] = phenotype[phenotype_clf_tgt].replace(phenotype_clf_nan)
phenotype = phenotype.dropna(subset=[PHENOTYPE_CLF_COLUMN])

# %%
_possible_mappings = {idx:[] for idx in phenotype.index}
for idx in phenotype.index:
    for v in gex[gex.index.str.startswith(idx)].index.values:
        _possible_mappings[idx].append(v)
_replacements = {k:sorted(v)[0] for k,v in _possible_mappings.items() if len(v)>0}
phenotype = phenotype.rename(index=_replacements, inplace=False)

# %%
both_index = sorted(set(phenotype.index).intersection(gex.index))
[(len(idx), idx[:5],) for idx in [gex.index, phenotype.index, both_index]]

# %%
gex = gex.loc[both_index]

phenotype = phenotype.loc[both_index,[PHENOTYPE_CLF_COLUMN,]]
phenotype[PHENOTYPE_CLF_COLUMN] = phenotype[PHENOTYPE_CLF_COLUMN].replace(phenotype_clf_map)
phenotype[PHENOTYPE_CLF_COLUMN].value_counts(), phenotype[PHENOTYPE_CLF_COLUMN].dtype, phenotype[PHENOTYPE_CLF_COLUMN].unique(), phenotype[PHENOTYPE_CLF_COLUMN].describe()

# %%
assert(all((gi==pi for gi,pi in zip(gex.index.to_list(), phenotype.index.to_list()))))

# %%
label_encoder = sklearn.preprocessing.LabelEncoder()
y = label_encoder.fit_transform(phenotype[PHENOTYPE_CLF_COLUMN])

# %%
gex.shape, y.shape

# %% [markdown]
# ## Pathways

# %%
pathway_folder = os.path.join(data_folder, "pathways")
os.makedirs(pathway_folder, exist_ok=True)

# %%
gex_genes = set(gex.columns.values)
gex_genes_indexer = {v:i for i,v in enumerate(gex.columns.values)}
get_pathways_with_indices = lambda pathways: [[gex_genes_indexer[gene] for gene in pathway] for pathway in pathways]

# %%
kegg_pathways = read_pathway_from_json_file(os.path.join(pathway_folder,"c2.cp.kegg.v7.5.1.json"), gex_genes)
kegg_pathways_with_indices = get_pathways_with_indices(kegg_pathways)
number_of_pathways = len(kegg_pathways_with_indices)
pathways_input_dimension = sum((len(pathway) for pathway in kegg_pathways_with_indices))
number_of_input_genes = len(functools.reduce(lambda acc_p, p: acc_p.union(set(p)), kegg_pathways_with_indices, set()))
number_of_pathways, pathways_input_dimension, number_of_input_genes

# %%
hallmark_pathways = read_pathway_from_json_file(os.path.join(pathway_folder,"h.all.v7.5.1.json"), gex_genes)
hallmark_pathways_with_indices = get_pathways_with_indices(hallmark_pathways)
number_of_pathways = len(hallmark_pathways_with_indices)
pathways_input_dimension = sum((len(pathway) for pathway in hallmark_pathways_with_indices))
number_of_input_genes = len(functools.reduce(lambda acc_p, p: acc_p.union(set(p)), hallmark_pathways_with_indices, set()))
number_of_pathways, pathways_input_dimension, number_of_input_genes

# %% [markdown]
# # Model

# %%
genes_dim = gex.values.shape[1]
gex.values.shape[1]

# %%
MAX_EPOCHS = 1024
DEFAULT_LR = 1e-4
BATCH_SIZE = -1
MAX_PATIENCE = 4
DEFAULT_EARLY_STOPPING_THRESHOLD = 0.001
EARLY_STOPPING_TEST_SIZE = 0.1
DEFAULT_SINK = lambda x:x
CLASSIFIERS_TO_TEST = [sklearn.svm.SVC(probability=True), sklearn.linear_model.LogisticRegression(), sklearn.ensemble.RandomForestClassifier()]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

build_early_stopping = lambda: skorch.callbacks.EarlyStopping(
    patience = 16+MAX_PATIENCE,
    threshold = DEFAULT_EARLY_STOPPING_THRESHOLD,
    sink = DEFAULT_SINK
)

kegg_p = [torch.tensor(pathway) for pathway in kegg_pathways_with_indices]
hmrk_p = [torch.tensor(pathway) for pathway in hallmark_pathways_with_indices]

pways_keys_lst = ["KEGG", "Hallmark Genes"]
pways_defs_lst = [kegg_p, hmrk_p]
pways_defs_lst_np = [kegg_pathways_with_indices, hallmark_pathways_with_indices]

paae_pipes = {
    **{
        f"PAAE-{type(Clf).__name__}-{pathway_hidden_dims}-{hidden_dims} ({pways_key})": sklearn.pipeline.Pipeline(
            [
                ("scale", sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=gex.values.shape[0],
                        output_distribution="normal",
                        )
                ),
                (
                    "net", 
                    ScoredNeuralNetAutoencoder(
                        PAAE,
                        module__genes_dim = genes_dim,
                        module__pathway_definitions = pways,
                        module__hidden_dims = hidden_dims,
                        module__pathway_hidden_dims = pathway_hidden_dims,
                        max_epochs = MAX_EPOCHS,
                        lr=DEFAULT_LR,
                        iterator_train__shuffle=True,
                        criterion = AE_MSELoss,
                        optimizer = torch.optim.Adam,
                        callbacks=[
                        ],
                        callbacks__print_log__sink = DEFAULT_SINK,
                        device=device,
                    )
                ),
                ("clf", Clf),
            ],
        )
        for pways_key, pways in zip(
            pways_keys_lst,
            pways_defs_lst,
        )
        for hidden_dims in ([],[128])
        for pathway_hidden_dims in ([],[128])
        for Clf in CLASSIFIERS_TO_TEST
    },
}

ae_pipes = {
    f"AE-{type(Clf).__name__}": sklearn.pipeline.Pipeline(
        [
            ("scale", sklearn.preprocessing.QuantileTransformer(
                    n_quantiles=gex.values.shape[0],
                    output_distribution="normal",
                    )
            ),
            (
                "net", 
                ScoredNeuralNetAutoencoder(
                    Autoencoder,
                    module__input_dim = genes_dim,
                    module__hidden_dims = [256,128,],
                    module__encoding_dim = 64,
                    max_epochs = MAX_EPOCHS,
                    lr=DEFAULT_LR,
                    iterator_train__shuffle=True,
                    criterion = AE_MSELoss,
                    optimizer = torch.optim.Adam,
                    callbacks=[
                    ],
                    callbacks__print_log__sink = DEFAULT_SINK,
                    device=device,
                )
            ),
            ("clf", Clf),
        ],
    )
    for Clf in CLASSIFIERS_TO_TEST
}

vae_pipes = {
    **{
        f"VAE-{type(Clf).__name__}-{beta_schedule_type}-β{beta}-{hidden_dims}": sklearn.pipeline.Pipeline(
            [
                ("scale", sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=gex.values.shape[0],
                        output_distribution="normal",
                        )
                ),
                (
                    "net",
                    ScoredNeuralNetAutoencoder(
                        VAE,
                        module__input_dim = genes_dim,
                        max_epochs = MAX_EPOCHS,
                        lr=DEFAULT_LR,
                        batch_size = BATCH_SIZE,
                        iterator_train__shuffle=True,
                        criterion = VAELoss,
                        criterion__original_loss = nn.MSELoss(),
                        criterion__beta_schedule = build_beta_schedule(beta_schedule_type, beta_start, beta_duration=beta_duration),
                        criterion__beta=beta,
                        criterion__kl_type="kingma",
                        optimizer = torch.optim.Adam,
                        callbacks=[
                        ],
                        callbacks__print_log__sink = DEFAULT_SINK,
                        device=device,
                    )
                ),
                ("clf", Clf),
            ],
        )
        for beta in [1,10,100]
        for beta_start, beta_duration in [(32,128)]
        for beta_schedule_type in ("smooth", "step")
        for hidden_dims in ([],[128])
        for Clf in CLASSIFIERS_TO_TEST
    },
}

pavae_pipes = {
    **{
        f"PAVAE-{type(Clf).__name__}-{beta_schedule_type}-β{beta}-{pathway_hidden_dims}-{hidden_dims} ({pways_key})": sklearn.pipeline.Pipeline(
            [
                ("scale", sklearn.preprocessing.QuantileTransformer(
                        n_quantiles=gex.values.shape[0],
                        output_distribution="normal",
                        )
                ),
                (
                    "net",
                    ScoredNeuralNetAutoencoder(
                        PAVAE,
                        module__genes_dim = genes_dim,
                        module__pathway_definitions = pways,
                        module__hidden_dims = hidden_dims,
                        module__pathway_hidden_dims = pathway_hidden_dims,
                        max_epochs = MAX_EPOCHS,
                        lr=DEFAULT_LR,
                        batch_size = BATCH_SIZE,
                        iterator_train__shuffle=True,
                        criterion = VAELoss,
                        criterion__original_loss = nn.MSELoss(),
                        criterion__beta_schedule = build_beta_schedule(beta_schedule_type, beta_start, beta_duration=beta_duration),
                        criterion__beta=beta,
                        criterion__kl_type="kingma",
                        optimizer = torch.optim.Adam,
                        callbacks=[
                        ],
                        callbacks__print_log__sink = DEFAULT_SINK,
                        device=device,
                    )
                ),
                ("clf", Clf),
            ],
        )
        for beta in [1,10,100]
        for beta_start, beta_duration in [(64,128)]
        for beta_schedule_type in ("smooth", "step")
        for pways_key, pways in zip(
            pways_keys_lst,
            pways_defs_lst,
        )
        for hidden_dims in ([],[128])
        for pathway_hidden_dims in ([],[128])
        for Clf in CLASSIFIERS_TO_TEST
    },
}

ExperimentDefinition = collections.namedtuple("ModelDefinition", ["Model", "params"])

params = {
   "net__lr": [DEFAULT_LR],
    "net__max_epochs":  [MAX_EPOCHS],
}

models_to_test = {
    **{k: ExperimentDefinition(ae_pipes[k], params) for k in ae_pipes},
    **{k: ExperimentDefinition(vae_pipes[k], params) for k in vae_pipes},
    **{k: ExperimentDefinition(paae_pipes[k], params) for k in paae_pipes},
    **{k: ExperimentDefinition(pavae_pipes[k], params) for k in pavae_pipes},
}
heavy_models = ["Autoencoder","VAE"]
very_heavy_models = []

# %%
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# %%
DISABLE_TQDM = False
REPETITIONS = 4
EXTERNAL_XVAL_SPLITS = 8
NUMBER_OF_EXTERNAL_JOBS = 1 if device != "cpu" else EXTERNAL_XVAL_SPLITS # Bigger than 1 doesn't work with GPU enabled
INTERNAL_XVAL_SPLITS = 4
NUMBER_OF_INTERNAL_JOBS = INTERNAL_XVAL_SPLITS

def make_pipeline_scorer(score_func,
        greater_is_better=True, 
        is_transformer=False,
        needs_proba=False,
        needs_threshold=False,):
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True, but not both.")
    if needs_threshold:
        raise NotImplementedError("Threshold losses not implemented")
    def __scorer(estimator, X, y_true, sample_weight=None):
        try:
            y_pred = estimator(X)
        except TypeError:
            if needs_proba:
                y_pred = estimator.predict_proba(X)
                if y_pred.shape[1]==2:
                    y_pred = y_pred[:,1]
            elif is_transformer:
                y_pred = estimator.transform(X)
            else:
                y_pred = estimator.predict(X)
        try:
            score = score_func(y_true, y_pred)
        except ValueError:
            score = score_func(y_true, y_pred[-1])
        return score * (1 if greater_is_better else -1)
    return __scorer

def pipeline_loss_scoring(greater_is_better=True, score_func=None, score_target=None, score_result_idx=0):
    def __scorer(estimator:sklearn.pipeline.Pipeline, X, y_true, sample_weight=None):
        if not isinstance(estimator, sklearn.pipeline.Pipeline):
            raise ValueError(f"Must pass estimator {estimator} as a pipeline")
        Xt = X
        iterlist = list(estimator._iter())
        for _, _, transform in iterlist[:-1]:
            if isinstance(transform, skorch.NeuralNet) or not hasattr(transform, "transform"):
                break
            Xt = transform.transform(Xt)
        _, _, transform in iterlist[-1]
        if isinstance(transform, ScoredNeuralNetAutoencoder):
            if score_func is None:
                score = skorch.scoring.loss_scoring(transform, Xt, Xt, sample_weight)
            else:
                ae_out = transform.full_transform(X)[score_result_idx]
                ae_tgt = Xt if score_target is None else y_true
                score = score_func(ae_tgt, ae_out)
        elif isinstance(transform, skorch.NeuralNet):
            if score_func is None:
                score = skorch.scoring.loss_scoring(transform, Xt, y_true, sample_weight)
            else:
                raise NotImplementedError()
        else:
            return np.nan
        return score * (1 if greater_is_better else -1)
    return __scorer

metrics = {
    "AE_Loss": pipeline_loss_scoring(greater_is_better=False,),
    "AE_MSE": pipeline_loss_scoring(greater_is_better=False, score_func=sklearn.metrics.mean_squared_error, score_result_idx=1,),
    "AE_R2": pipeline_loss_scoring(greater_is_better=False, score_func=sklearn.metrics.r2_score, score_result_idx=1),
    "Accuracy": make_pipeline_scorer(sklearn.metrics.accuracy_score,),
    "Precision": make_pipeline_scorer(functools.partial(sklearn.metrics.precision_score, average="macro" if len(set(y))>2 else "binary",),),
    "Recall": make_pipeline_scorer(functools.partial(sklearn.metrics.recall_score, average="macro" if len(set(y))>2 else "binary",),),
    "F1": make_pipeline_scorer(functools.partial(sklearn.metrics.f1_score, average="macro" if len(set(y))>2 else "binary"),),
    "AUC": make_pipeline_scorer(functools.partial(sklearn.metrics.roc_auc_score, average="macro" if len(set(y))>2 else None, multi_class="ovr",), needs_proba=True),
}

# %%
normalization_types_to_test = ["log2p1e-3_fpkm"]

# %%
paper_models_to_test = {k:v for k,v in models_to_test.items()} #.replace("-[]","-0").replace("-[128]","-1")
print(normalization_types_to_test)
print(paper_models_to_test.keys())

# %%
len(list(iter(paper_models_to_test))), list(iter(paper_models_to_test))

# %%
validation_data = {
    "Input Normalization": [],
    "Model": [],
    "Repetition": [],
    "External Split": [],
    "Internal Split": [],
    "Grid ID": [],
    "Parameters": [],
    "Validation Loss": [],
    "Mean Fit Time": [],
    **{f"Validation {m}":[] for m in metrics},
}

test_data = {
    "Input Normalization": [],
    "Model": [],
    "Repetition": [],
    "External Split": [],
    "Parameters": [],
    "Epochs": [],
    "Fit Time": [],
    **{f"Test {m}":[] for m in metrics},
}

# %%
results_time = datetime.datetime.now()
log_fname = f"errlog_{results_time:%Y%m%d%H%M}.txt"
test_results_fname = f"clf_valid_results_{results_time:%Y%m%d%H%M}.csv"
print(test_results_fname)

try:
    for x_normtype in tqdm(normalization_types_to_test, desc="X Normalization", leave = False, disable = DISABLE_TQDM,):
        X = gex.values
        y = y
        output_is_log2pk = x_normtype.split("_")[0].startswith("log2p")
        output_k = float(x_normtype.split("_")[0].split("p")[1]) if output_is_log2pk else 1
        output_type = x_normtype.split("_")[1]
        X = sample_wise_preprocess_fn(
            X, copy=False,
            input_is_log2pk=True, input_k=1, input_type="fpkm",
            output_is_log2pk=output_is_log2pk, output_k=output_k, output_type=output_type,
            )
        X = X.astype(np.float32)
        if not np.isfinite(X).all():
            warnings.warn(f"Some inputs were not finite, skipping the normalization type {x_normtype}")
            continue
        for repetition in tqdm(range(REPETITIONS), desc="Repetition", leave = False, disable = DISABLE_TQDM,):
            for model_name in tqdm(paper_models_to_test, desc="Model", leave = False, disable = DISABLE_TQDM,):
                definition = paper_models_to_test[model_name]
                n_internal_jobs = NUMBER_OF_INTERNAL_JOBS
                if any(map(lambda x: model_name.lower().startswith(x.lower()), heavy_models)):
                    n_internal_jobs = NUMBER_OF_INTERNAL_JOBS // 2
                elif any(map(lambda x: model_name.lower().startswith(x.lower()), very_heavy_models)):
                    n_internal_jobs = 1
                gsearch = definition.Model
                try:
                    test_cv_results = sklearn.model_selection.cross_validate(
                        gsearch,
                        X, y,
                        cv=sklearn.model_selection.StratifiedKFold(EXTERNAL_XVAL_SPLITS),
                        return_estimator=True,
                        n_jobs=NUMBER_OF_EXTERNAL_JOBS,
                        scoring = metrics,
                        error_score = "raise",
                    )

                    for external_split in range(EXTERNAL_XVAL_SPLITS):
                        test_data["Input Normalization"].append(x_normtype)
                        test_data["Model"].append(model_name)
                        test_data["Repetition"].append(repetition)
                        test_data["External Split"].append(external_split)
                        test_data["Parameters"].append("")
                        for m in metrics:
                            test_data[f"Test {m}"].append(test_cv_results[f"test_{m}"][external_split])
                        try:
                            num_epochs = test_cv_results["estimator"][external_split]["net"].history[-1,"epoch"]
                        except (KeyError) as e:
                            num_epochs = np.nan
                        test_data["Epochs"].append(num_epochs)
                        test_data["Fit Time"].append(test_cv_results["fit_time"][external_split])
                            
                        test_results = pd.DataFrame(test_data)
                        test_results.to_csv(os.path.join(results_folder,test_results_fname))
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    with open(log_fname, "a") as logf:
                        print(model_name, file=logf)
                        traceback.print_exception(e, file=logf)
                        print(file=logf, flush=True)

except KeyboardInterrupt:
    pass

test_results = pd.DataFrame(test_data)
test_results.to_csv(os.path.join(results_folder,test_results_fname))

# %% [markdown]
# 

# %%




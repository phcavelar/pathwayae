# %%
import os
import gc
import time
import typing
import collections
import json
import itertools
import functools
import datetime
import warnings

# %%
from tqdm.notebook import tqdm

# %%
import numpy as np
import scipy as sp
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.metrics
import sklearn.decomposition
import sklearn.manifold
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.base
import umap

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

PATHWAY_NAME_TO_INFO = {
    "kegg": ("c2.cp.kegg.v7.5.1.json","KEGG","KEGG_"),
    "oncogenic": ("c6.all.v7.5.1.json","Oncogenic", None),
    "hallmark": ("h.all.v7.5.1.json","Hallmark Genes","HALLMARK_"),
    "reactome": ("c2.cp.reactome.v2022.1.Hs.json","Reactome","REACTOME_"),
}

def get_pathway_set(pathway_set_prefix, pathway_description_path, gex_genes, get_pathways_with_indices):

    # %%
    pathways = read_pathway_from_json_file(pathway_description_path, gex_genes)
    pathways_with_indices = get_pathways_with_indices(pathways)

    with open(pathway_description_path, "r") as f:
        pathway_descriptions = json.load(f)
    pathway_genes = [(k,pathway_descriptions[k]["geneSymbols"]) for k in pathway_descriptions]
    all_pathway_genes = functools.reduce(lambda acc, v: acc.union(set(v[1])), pathway_genes, set())
    common_genes = all_pathway_genes.intersection(gex_genes)
    pathway_genes_with_allowed_genes = [(k,[gene for gene in pathway if gene in common_genes]) for k, pathway in pathway_genes]

    pway_names = [k for (k,p), pi in zip(pathway_genes_with_allowed_genes,pathways_with_indices)]
    assert(all([(len(p)==len(pi)) for (k,p), pi in zip(pathway_genes_with_allowed_genes,pathways_with_indices)]))
    if pathway_set_prefix is not None:
        pway_names = [pname.replace(f"{pathway_set_prefix}_","") for pname in pway_names]
    return pathways, pathways_with_indices, pway_names


def main(
        model_to_use:typing.Literal["ae","paae","pavae","vae"] = "paae",
        pathway_to_use = "kegg",
        num_reps = 16,
        data_folder="~/data/",
        results_folder = "results/metabric",
        images_folder = "images/metabric",
        saving_formats = ["png", "pdf", "svg"],
        disable_tqdm = False,
        ):
    model_uses_pathways = model_to_use.startswith("pa")
    if model_uses_pathways:
        pathway_to_use = pathway_to_use.lower()
        assert pathway_to_use in PATHWAY_NAME_TO_INFO
    # %%
    data_folder = os.path.expanduser(data_folder) 
    cancer_type = "BRCA"
    tcga_folder = os.path.join(data_folder, "pathwayae", "tcga")
    meta_folder = os.path.join(data_folder, "pathwayae", "metabric")
    pathway_folder = os.path.join(data_folder, "pathwayae", "pathways")

    # %%
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    for fmt in saving_formats: os.makedirs(os.path.join(images_folder,fmt), exist_ok=True)


    # %%
    metabric_ensembl_counts_gex_tsv_fname = os.path.join(meta_folder, f"data_mrna_agilent_microarray.txt.gz")
    gex_meta = pd.read_csv(metabric_ensembl_counts_gex_tsv_fname, sep="\t", index_col="Hugo_Symbol").drop(columns="Entrez_Gene_Id")
    gex_meta.index.rename("SampleID", inplace=True)
    gex_meta = gex_meta.T.dropna(axis="columns")
    gex_meta.columns = gex_meta.columns.str.upper()
    # %%
    metabric_phenotype_csv_fname = os.path.join(meta_folder, f"data_clinical_patient.txt.gz")
    metabric_phenotype = pd.read_csv(metabric_phenotype_csv_fname, sep="\t", comment="#", index_col="PATIENT_ID")
    metabric_phenotype.index.rename("SampleID", inplace=True)
    metabric_phenotype.rename(columns={"CLAUDIN_SUBTYPE": "PAM50"}, inplace=True)
    metabric_phenotype = metabric_phenotype[["PAM50"]]
    metabric_phenotype.dropna(inplace=True)
    metabric_phenotype.drop(index=metabric_phenotype.index[metabric_phenotype["PAM50"] == "claudin-low"], inplace=True)
    metabric_phenotype.drop(index=metabric_phenotype.index[metabric_phenotype["PAM50"] == "NC"], inplace=True)

    for c in metabric_phenotype.columns:
        print(c, metabric_phenotype[c].count(), metabric_phenotype[c].value_counts())

    # %%
    assert sum(metabric_phenotype.PAM50.isna())==0

    # %%
    ensembl_fpkm_gex_tsv_fname = os.path.join(tcga_folder, f"TCGA-{cancer_type}.htseq_fpkm.tsv.gz")
    gex = pd.read_csv(ensembl_fpkm_gex_tsv_fname, sep="\t", index_col="Ensembl_ID").T.dropna(axis="columns")

    zero_variance_columns = set(gex.var()[gex.var()==0].index)

    gex = gex.drop(columns=list(zero_variance_columns))

    with open(os.path.join(tcga_folder, "ensembl_to_gene_id.json")) as f:
        ensembl_to_gex_dict = json.load(f)

    columns_to_drop = [k for k in ensembl_to_gex_dict if ensembl_to_gex_dict[k]=="" and k not in zero_variance_columns]
    gex = gex.drop(columns=columns_to_drop)

    value_counts = {}
    for k,v in ensembl_to_gex_dict.items():
        if v == '':
            continue
        elif k in gex.columns:
            if v in value_counts:
                value_counts[v].append(k)
            else:
                value_counts[v] = [k]
    value_counts = {k:v for k,v in value_counts.items() if len(v)>1}
    print(len(value_counts), sum((len(v) for v in value_counts.values())))

    # %%
    duplicate_and_in_both = ["MATR3", "EMG1", "TMSB15B", "BMS1P4", "POLR2J4",]

    # %%
    for g_name, g_ensembl_ids in [(v,value_counts[v]) for v in duplicate_and_in_both]:
        n_ambiguous = len(g_ensembl_ids)
        fig, axes = plt.subplots(3+n_ambiguous,1, sharex=True, sharey=True, figsize=[plt.rcParams["figure.figsize"][0], plt.rcParams["figure.figsize"][1]*1.5])
        to_log2pk(from_log2pk(gex[g_ensembl_ids], 1).sum(axis=1), 1).hist(bins=20, ax=axes[n_ambiguous])
        axes[n_ambiguous].set_title(f"sum {g_name} (tcga, fpkm)")
        to_log2pk(from_log2pk(gex[g_ensembl_ids], 1).mean(axis=1), 1).hist(bins=20, ax=axes[n_ambiguous+1])
        axes[n_ambiguous+1].set_title(f"mean {g_name} (tcga, fpkm)")
        gex_meta[g_name].hist(bins=20, ax=axes[n_ambiguous+2])
        axes[n_ambiguous+2].set_title(f"{g_name} (metabric, counts)")
        for i, g_e_idx in enumerate(g_ensembl_ids):
            gex[g_e_idx].hist(bins=20, ax=axes[i])
            axes[i].set_title(f"{g_e_idx} (tcga, fpkm)")
        axes[n_ambiguous+2].set_xlabel("log2p1")
        fig.tight_layout()
        for fmt in saving_formats: plt.savefig(os.path.join(images_folder, fmt, f"ambiguous_{g_name}.{fmt}"))
        plt.close()

    # %%
    # Merge values that map to the same gene symbol
    for g_name, g_ensembl_ids in value_counts.items():
        n_ambiguous = len(g_ensembl_ids)
        gex[g_name] = to_log2pk(from_log2pk(gex[g_ensembl_ids], 1).mean(axis=1), 1)
        gex.drop(columns=g_ensembl_ids, inplace=True)

    # %%
    # Rename the rest
    gex = gex.rename(columns=ensembl_to_gex_dict)

    gex.columns.rename("GeneName", inplace=True)
    gex.index.rename("SampleID", inplace=True)

    # %%
    gex.shape, gex_meta.shape

    # %%
    genes_in_both = sorted(set(gex_meta.columns).intersection(set(gex.columns)))
    len(genes_in_both)

    # %%
    gex = gex.loc[:,genes_in_both]
    gex_meta = gex_meta.loc[:,genes_in_both]
    gex.shape, gex_meta.shape

    # %%
    assert all(map(lambda x: x[0]==x[1], list(zip(gex.columns, gex_meta.columns))))


    # %%
    ensembl_fpkm_phenotype_tsv_fname = os.path.join(tcga_folder, f"TCGA.{cancer_type}.sampleMap_{cancer_type}_clinicalMatrix")
    phenotype = pd.read_csv(ensembl_fpkm_phenotype_tsv_fname, sep="\t", index_col="sampleID")
    phenotype.index = phenotype.index.rename("SampleID")
    phenotype = phenotype[[c for c in phenotype.columns if "pam50" in c.lower()]]
    for c in phenotype.columns:
        print(phenotype[c].count(), phenotype[c].value_counts())

    # %%
    phenotype_clf_tgt = "PAM50Call_RNAseq"
    phenotype_clf_tgt_meta = "PAM50"
    phenotype_clf_map = {
        "LumA":0,
        "LumB":1,
        "Basal":2,
        "Normal":3,
        "Her2":4,
    }
    phenotype_clf_nan = {f"{value}":np.nan for value in [np.nan, "not reported", ""]}
    phenotype.columns

    # %%
    # Drop nan
    PHENOTYPE_CLF_COLUMN = "subtype"
    phenotype[PHENOTYPE_CLF_COLUMN] = phenotype[phenotype_clf_tgt].replace(phenotype_clf_nan)
    phenotype = phenotype.dropna(subset=[PHENOTYPE_CLF_COLUMN])
    phenotype.columns

    # %%
    _possible_mappings = {idx:[] for idx in phenotype.index}
    for idx in phenotype.index:
        for v in gex[gex.index.str.startswith(idx)].index.values:
            _possible_mappings[idx].append(v)
    _replacements = {k:sorted(v)[0] for k,v in _possible_mappings.items() if len(v)>0}
    phenotype = phenotype.rename(index=_replacements, inplace=False)
    phenotype.columns

    # %%
    both_index = sorted(set(phenotype.index).intersection(gex.index))
    [(len(idx), idx[:5],) for idx in [gex.index, phenotype.index, both_index]]

    # %%
    both_index_meta = sorted(set(metabric_phenotype.index).intersection(gex_meta.index))
    [(len(idx), idx[:5],) for idx in [gex_meta.index, metabric_phenotype.index, both_index_meta]]

    # %%
    gex = gex.loc[both_index]
    gex_meta = gex_meta.loc[both_index_meta]

    full_phenotype = phenotype
    phenotype_meta = full_phenotype_meta = metabric_phenotype

    phenotype = phenotype.loc[both_index,[PHENOTYPE_CLF_COLUMN,]]
    phenotype_meta = full_phenotype_meta.loc[both_index_meta,["PAM50"]]
    phenotype[PHENOTYPE_CLF_COLUMN] = phenotype[PHENOTYPE_CLF_COLUMN].replace(phenotype_clf_map)
    phenotype_meta[PHENOTYPE_CLF_COLUMN] = phenotype_meta["PAM50"].replace(phenotype_clf_map)
    phenotype[phenotype_clf_tgt] = full_phenotype.loc[phenotype.index,phenotype_clf_tgt]
    phenotype_meta[phenotype_clf_tgt_meta] = phenotype_meta.loc[phenotype_meta.index,phenotype_clf_tgt_meta]


    (
        phenotype[PHENOTYPE_CLF_COLUMN].value_counts(), phenotype[PHENOTYPE_CLF_COLUMN].dtype, phenotype[PHENOTYPE_CLF_COLUMN].unique(), phenotype[PHENOTYPE_CLF_COLUMN].describe(),
        "\n",
        phenotype_meta[PHENOTYPE_CLF_COLUMN].value_counts(), phenotype_meta[PHENOTYPE_CLF_COLUMN].dtype, phenotype_meta[PHENOTYPE_CLF_COLUMN].unique(), phenotype_meta[PHENOTYPE_CLF_COLUMN].describe(),
    )

    # %%
    hue_order = sorted(phenotype[phenotype_clf_tgt].unique())
    sns.displot(data=phenotype, y=phenotype_clf_tgt, hue=phenotype_clf_tgt, hue_order=hue_order)
    for fmt in saving_formats: plt.savefig(os.path.join(images_folder, fmt, f"displot_tcga.{fmt}"))
    plt.close()
    sns.displot(data=phenotype_meta, y=phenotype_clf_tgt_meta, hue=phenotype_clf_tgt_meta, hue_order=hue_order)
    for fmt in saving_formats: plt.savefig(os.path.join(images_folder, fmt, f"displot_meta.{fmt}"))
    plt.close()

    # %%
    assert(all((gi==pi for gi,pi in zip(gex.index.to_list(), phenotype.index.to_list()))))

    # %%
    assert(all((gi==pi for gi,pi in zip(gex_meta.index.to_list(), phenotype_meta.index.to_list()))))

    # %%
    def plot_2d_space(df, SpaceTransformer=sklearn.decomposition.PCA, **kwargs):
        values = SpaceTransformer().fit_transform(df)
        return sns.scatterplot(x=values[:,0], y=values[:,1], **kwargs)

    # %%
    plot_2d_space(gex, SpaceTransformer=umap.UMAP, hue=phenotype[phenotype_clf_tgt])
    for fmt in saving_formats: plt.savefig(os.path.join(images_folder, fmt, f"dist_gex_tcga.{fmt}"))
    plt.close()

    # %%
    plot_2d_space(gex_meta, SpaceTransformer=umap.UMAP, hue=phenotype_meta[phenotype_clf_tgt_meta])
    for fmt in saving_formats: plt.savefig(os.path.join(images_folder, fmt, f"dist_gex_meta.{fmt}"))
    plt.close()

    # %%
    genes_dim = gex.values.shape[1]

    # %%
    gex_genes = set(gex.columns.values)
    gex_genes_indexer = {v:i for i,v in enumerate(gex.columns.values)}
    get_pathways_with_indices = lambda pathways: [[gex_genes_indexer[gene] for gene in pathway] for pathway in pathways]

    # %%
    pathway_description_fname, pathway_set_pretty_name, pathway_set_prefix =  PATHWAY_NAME_TO_INFO[pathway_to_use]
    pathway_description_path = os.path.join(pathway_folder, pathway_description_fname)
    pathways, pathways_with_indices, pway_names = get_pathway_set(pathway_set_prefix, pathway_description_path, gex_genes, get_pathways_with_indices)
    number_of_pathways = len(pathways_with_indices)
    pathways_input_dimension = sum((len(pathway) for pathway in pathways_with_indices))
    number_of_input_genes = len(functools.reduce(lambda acc_p, p: acc_p.union(set(p)), pathways_with_indices, set()))
    print(pathway_set_pretty_name, number_of_pathways, pathways_input_dimension, number_of_input_genes)

    # %%
    MAX_EPOCHS = 1024
    BETA_START_AND_DURATION_LIST = [(32,128)]
    DEFAULT_LR = 1e-4
    BATCH_SIZE = int(gex.values.shape[0])
    MAX_PATIENCE = 8
    DEFAULT_EARLY_STOPPING_THRESHOLD = 0.001
    DEFAULT_SINK = lambda x:x
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    build_early_stopping = lambda: skorch.callbacks.EarlyStopping(
        patience = 16+MAX_PATIENCE,
        threshold = DEFAULT_EARLY_STOPPING_THRESHOLD,
        sink = DEFAULT_SINK
    )

    pset_t = [torch.tensor(pathway) for pathway in pathways_with_indices]

    pways_keys_lst = [pathway_set_pretty_name,]
    pways_defs_lst = [pset_t,]

    paae_pipes = {
        **{
            f"PAAE-{pathway_hidden_dims}-{hidden_and_enc_dims} ({pways_key})": sklearn.pipeline.Pipeline(
                [
                    ("scale", sklearn.preprocessing.QuantileTransformer(
                            n_quantiles=max(*gex.values.shape, *gex_meta.values.shape,),
                            output_distribution="normal",
                            )
                    ),
                    (
                        "net", 
                        ScoredNeuralNetAutoencoder(
                            PAAE,
                            module__genes_dim = genes_dim,
                            module__pathway_definitions = pways,
                            module__pathway_hidden_dims = pathway_hidden_dims,
                            module__hidden_dims = hidden_and_enc_dims[:-1],
                            module__encoding_dim = hidden_and_enc_dims[-1],
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
                ],
            )
            for pways_key, pways in zip(
                pways_keys_lst,
                pways_defs_lst,
            )
            for hidden_and_enc_dims in ([64],[128,64])
            for pathway_hidden_dims in ([],[32],[32,16])
        },
    }

    ae_pipes = {
        **{
            f"AE-{hidden_and_enc_dims}": sklearn.pipeline.Pipeline(
                [
                    ("scale", sklearn.preprocessing.QuantileTransformer(
                            n_quantiles=max(*gex.values.shape, *gex_meta.values.shape,),
                            output_distribution="normal",
                            )
                    ),
                    (
                        "net", 
                        ScoredNeuralNetAutoencoder(
                            Autoencoder,
                            module__input_dim = genes_dim,
                            module__hidden_dims = hidden_and_enc_dims[:-1],
                            module__encoding_dim = hidden_and_enc_dims[-1],
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
                ],
            )
            for hidden_and_enc_dims in (
                    [128,64], [256,128,64], [512,256,128,64],
                    )
        }
    }

    vae_pipes = {
        **{
            f"VAE-{beta_schedule_type}-β{beta}-{hidden_and_enc_dims}": sklearn.pipeline.Pipeline(
                [
                    ("scale", sklearn.preprocessing.QuantileTransformer(
                            n_quantiles=max(*gex.values.shape, *gex_meta.values.shape,),
                            output_distribution="normal",
                            )
                    ),
                    (
                        "net",
                        ScoredNeuralNetAutoencoder(
                            VAE,
                            module__input_dim = genes_dim,
                            module__hidden_dims = hidden_and_enc_dims[:-1],
                            module__encoding_dim = hidden_and_enc_dims[-1],
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
                ],
            )
            for beta in [1,5,10,50,100]
            for beta_start, beta_duration in BETA_START_AND_DURATION_LIST
            for beta_schedule_type in ("smooth", "step",)
            for hidden_and_enc_dims in ([128,64], [256,128,64], [512,256,128,64],)
        },
    }

    pavae_pipes = {
        **{
            f"PAVAE-{beta_schedule_type}-β{beta}-{pathway_hidden_dims}-{hidden_and_enc_dims} ({pways_key})": sklearn.pipeline.Pipeline(
                [
                    ("scale", sklearn.preprocessing.QuantileTransformer(
                            n_quantiles=max(*gex.values.shape, *gex_meta.values.shape,),
                            output_distribution="normal",
                            )
                    ),
                    (
                        "net",
                        ScoredNeuralNetAutoencoder(
                            PAVAE,
                            module__genes_dim = genes_dim,
                            module__pathway_definitions = pways,
                            module__pathway_hidden_dims = pathway_hidden_dims,
                            module__hidden_dims = hidden_and_enc_dims[:-1],
                            module__encoding_dim = hidden_and_enc_dims[-1],
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
                ],
            )
            for beta in [1,5,10,50,100]
            for beta_start, beta_duration in BETA_START_AND_DURATION_LIST
            for beta_schedule_type in ("smooth", "step")
            for pways_key, pways in zip(
                pways_keys_lst,
                pways_defs_lst,
            )
            for hidden_and_enc_dims in ([128,64], [256,128,64],)
            for pathway_hidden_dims in ([],[32],[32,16])
        },
    }

    scale_pipe = sklearn.pipeline.Pipeline(
        [
            ("scale", sklearn.preprocessing.QuantileTransformer(
                    n_quantiles=max(*gex.values.shape, *gex_meta.values.shape,),
                    output_distribution="normal",
                    )
            ),
        ],
    )

    ExperimentDefinition = collections.namedtuple("ModelDefinition", ["Model", "params"])

    params = {
        "net__lr": [DEFAULT_LR],
        "net__max_epochs":  [MAX_EPOCHS],
    }

    models_to_test_dict = {
        "ae": {**{k: ExperimentDefinition(ae_pipes[k], params) for k in ae_pipes}},
        "z-norm": {"z-norm": ExperimentDefinition(scale_pipe, {})},
        "paae": {**{k: ExperimentDefinition(paae_pipes[k], params) for k in paae_pipes}},
        "vae": {**{k: ExperimentDefinition(vae_pipes[k], params) for k in vae_pipes}},
        "pavae": {**{k: ExperimentDefinition(pavae_pipes[k], params) for k in pavae_pipes}},
    }

    models_to_test = models_to_test_dict[model_to_use]

    paper_normtype = "log2p1e-3_fpkm"
    normalization_types_to_test = [paper_normtype]

    assert "log2p1e-3_fpkm" in normalization_types_to_test, "AAAAAA"

    # %%
    paper_models_to_test = {k:v for k,v in models_to_test.items()}

    # %%
    label_encoder = sklearn.preprocessing.LabelEncoder()
    y_train = label_encoder.fit_transform(phenotype[PHENOTYPE_CLF_COLUMN])
    y_test = label_encoder.transform(phenotype_meta[PHENOTYPE_CLF_COLUMN])

    y = y_train
    metrics = {
        "Accuracy": sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score,),
        "Precision": sklearn.metrics.make_scorer(functools.partial(sklearn.metrics.precision_score, average="macro" if len(set(y))>2 else "binary",),),
        "Recall": sklearn.metrics.make_scorer(functools.partial(sklearn.metrics.recall_score, average="macro" if len(set(y))>2 else "binary",),),
        "F1": sklearn.metrics.make_scorer(functools.partial(sklearn.metrics.f1_score, average="macro" if len(set(y))>2 else "binary"),),
        "AUC": sklearn.metrics.make_scorer(functools.partial(sklearn.metrics.roc_auc_score, average="macro" if len(set(y))>2 else None, multi_class="ovr",), needs_proba=True),
    }

    CLASSIFIERS_TO_TEST = [sklearn.svm.SVC(probability=True), sklearn.linear_model.LogisticRegression(), sklearn.ensemble.RandomForestClassifier()]

    # %%
    results_time = datetime.datetime.now()
    results_dict = {
        "Repetition": [],
        "Model": [],
        "Model Size": [],
        "Fit Time": [],
        "Train MSE": [],
        "Test MSE": [],
        "Space Type": [],
        "Classifier": [],
        **{metric: list() for metric in metrics}
    }

    # %%
    results_time = datetime.datetime.now()
    results_fname = f"clf_meta_results_{results_time:%Y%m%d%H%M}.csv"

    try:
        X_train = gex.values
        X_test = gex_meta.values
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        if not np.isfinite(X_train).all():
            raise ValueError(f"Some inputs were not finite")
        for model_name in tqdm(paper_models_to_test, desc="Model", leave=True, disable = disable_tqdm,):
            for repetition in tqdm(range(1,num_reps+1), desc="Repetition", leave=False, disable = disable_tqdm,):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                gc.collect()
                Model:sklearn.pipeline.Pipeline = paper_models_to_test[model_name].Model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_fit_start = time.time()
                    model = Model.fit(X_train)
                    model_fit_time = time.time() - model_fit_start
                    x_hat = model.transform(X_train)
                
                if not np.isfinite(x_hat).all():
                    print(f"Skip {model_name}, x_hat has nans")
                    del model
                    continue
                
                if hasattr(model, "score"):
                    model_score_train = model.score(X_train)
                    model_score_test = model.score(X_test)
                else:
                    model_score_train = np.nan
                    model_score_test = np.nan
                
                z_train = X_train
                z_test = X_test
                print(model_name, model_fit_time)
                for step_name, step in model.steps:
                    if isinstance(step,ScoredNeuralNetAutoencoder):
                        module_param = next(step.module_.parameters())
                        module_param_size = np.sum([np.prod(p.detach().cpu().numpy().shape) for p in step.module_.parameters()])

                        # Get PAs first
                        if "PA" in model_name:
                            with torch.no_grad():
                                p_train = step.module_.get_pathway_activities(torch.tensor(z_train, device=module_param.device, dtype=module_param.dtype))
                                p_test = step.module_.get_pathway_activities(torch.tensor(z_test, device=module_param.device, dtype=module_param.dtype))
                            
                            if isinstance(p_train, tuple):
                                p_train = p_train[0]
                            if isinstance(p_test, tuple):
                                p_test = p_test[0]

                            p_train = p_train.detach().cpu().numpy()
                            p_test = p_test.detach().cpu().numpy()
                        
                        # Then update Zs
                        with torch.no_grad():
                            z_train = step.module_.encode(torch.tensor(z_train, device=module_param.device, dtype=module_param.dtype))
                            z_test = step.module_.encode(torch.tensor(z_test, device=module_param.device, dtype=module_param.dtype))

                        if isinstance(z_train, tuple):
                            z_train = z_train[0]
                        if isinstance(z_test, tuple):
                            z_test = z_test[0]

                        z_train = z_train.detach().cpu().numpy()
                        z_test = z_test.detach().cpu().numpy()

                        break
                    z_train = step.transform(z_train)
                    if isinstance(step,sklearn.preprocessing.QuantileTransformer):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            z_test = sklearn.preprocessing.quantile_transform(z_test, n_quantiles=step.n_quantiles, output_distribution=step.output_distribution)
                    else:
                        z_test = step.transform(z_test)

                for Classifier in CLASSIFIERS_TO_TEST:
                    clf = sklearn.base.clone(Classifier)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        clf.fit(z_train, y_train)

                    results_dict["Repetition"].append(repetition)
                    results_dict["Model"].append(model_name)
                    results_dict["Model Size"].append(module_param_size)
                    results_dict["Fit Time"].append(model_fit_time)
                    results_dict["Classifier"].append(type(Classifier).__name__)
                    results_dict["Train MSE"].append(model_score_train)
                    results_dict["Test MSE"].append(model_score_test)
                    results_dict["Space Type"].append("Latent")
                    for metric in metrics:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            score = metrics[metric](clf, z_test, y_test)
                        results_dict[metric].append(score)

                    if "PA" in model_name:
                        clf = sklearn.base.clone(Classifier)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            clf.fit(p_train, y_train)

                        results_dict["Repetition"].append(repetition)
                        results_dict["Model"].append(model_name)
                        results_dict["Model Size"].append(module_param_size)
                        results_dict["Fit Time"].append(model_fit_time)
                        results_dict["Classifier"].append(type(Classifier).__name__)
                        results_dict["Train MSE"].append(model_score_train)
                        results_dict["Test MSE"].append(model_score_test)
                        results_dict["Space Type"].append("Pathway Activity")
                        for metric in metrics:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                score = metrics[metric](clf, p_test, y_test)
                            results_dict[metric].append(score)
                    
                    try:
                        results_df = pd.DataFrame(results_dict)
                        results_df.to_csv(os.path.join(results_folder, results_fname))
                    except ValueError:
                        with open(f"{results_fname}.json", "w") as f:
                            json.dump(results_dict, f)
                
                for step_name, step in model.steps:
                    print(step_name, end=" ")
                    if isinstance(step,ScoredNeuralNetAutoencoder):
                        if hasattr(step, "module_"):
                            del step.module_
                del model
                
    except KeyboardInterrupt:
        pass

    # %%
    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    # %%

    results_df = pd.DataFrame(results_dict)
    results_df["Model Class"] = ""
    for model_name in results_df["Model"].unique():
        model_class = model_name.split("-")[0] + ("" if "(" not in model_name else (" (" + model_name.split("(")[1]))
        results_df.loc[results_df["Model"]==model_name,"Model Class"] = model_class

    results_df.to_csv(os.path.join(results_folder, results_fname))
    return


if __name__ == "__main__":
    import fire
    fire.Fire(main)

from typing import Literal
from dataclasses import dataclass
from collections import namedtuple
from os.path import join

@dataclass
class DataSetSources:
    gex_matrix_source:str
    gex_matrix_file_name:str
    gex_matrix_file_format:str
    gex_original_expression_format:str
    gex_id_to_gene_mapping_source:str
    gex_id_to_gene_mapping_fname:str

__GDC_HUB = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/"
__GDC_GENEMAPPING_FNAME = "gencode.v22.annotation.gene.probeMap"
__GDC_CANCERS = [""]

# TODO: Try with pancan normalised/percentile normalised TCGA data?
#__TCGA_HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
#__TCGA_GENEMAPPING_FNAME = "probeMap%2Fhugo_gencode_good_hg19_V24lift37_probemap"

_DATASET_SOURCES = {
    **{
        f"gdc_{cancer.lower()}_htseq_fpkm": DataSetSources(
            gex_matrix_source = join(__GDC_HUB,f"TCGA-{cancer.upper()}.htseq_{dtype}.tsv.gz"),
            gex_matrix_file_name = f"TCGA-{cancer.upper()}.htseq_{dtype}.tsv.gz",
            gex_matrix_file_format = "tsv.gz",
            gex_original_expression_format = f"log2p1_{dtype}",
            gex_id_to_gene_mapping_source = join(__GDC_HUB,__GDC_GENEMAPPING_FNAME),
            gex_id_to_gene_mapping_fname = __GDC_GENEMAPPING_FNAME,
        )
        for cancer in __GDC_CANCERS
        for dtype in ["fpkm","counts"]
    }
    ,
    "gdc_lusc_htseq_fpkm": DataSetSources(
        gex_matrix_source = join(__GDC_HUB,""),
        gex_matrix_file_name = "",
        gex_matrix_file_format = "tsv.gz",
        gex_original_expression_format = "log2p1_fpkm",
        gex_id_to_gene_mapping_source = join(__GDC_HUB,__GDC_GENEMAPPING_FNAME),
        gex_id_to_gene_mapping_fname = __GDC_GENEMAPPING_FNAME,
    ),
    "": DataSetSources(
        gex_matrix_source = join(__GDC_HUB,""),
        gex_matrix_file_name = "",
        gex_matrix_file_format = "tsv.gz",
        gex_original_expression_format = "log2p1_fpkm",
        gex_id_to_gene_mapping_source = join(__GDC_HUB,__GDC_GENEMAPPING_FNAME),
        gex_id_to_gene_mapping_fname = __GDC_GENEMAPPING_FNAME,
    ),
}


class GeneExpressionDatasetLoader():
    def __init__(self, dataset, path="~/data/pathwayae", download=True, limit_to_genes=None):
        raise NotImplementedError()
        self.path = path
        self.download = download

    def load(self):
        raise NotImplementedError()
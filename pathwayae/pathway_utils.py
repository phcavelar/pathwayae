import functools
import json

def read_pathway_from_json_file(pathway_description_path:str, allowed_genes:set=None):
    with open(pathway_description_path, "r") as f:
        pathway_descriptions = json.load(f)
    pathway_genes = [pathway_descriptions[k]["geneSymbols"] for k in pathway_descriptions]
    if allowed_genes is None:
        return pathway_genes
    else:
        all_pathway_genes = functools.reduce(lambda acc, v: acc.union(set(v)), pathway_genes, set())
        common_genes = all_pathway_genes.intersection(allowed_genes)
        pathway_genes_with_allowed_genes = [[gene for gene in pathway if gene in common_genes] for pathway in pathway_genes]
        return pathway_genes_with_allowed_genes
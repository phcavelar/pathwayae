# PathwayAE

Code for Pathway-based AEs https://arxiv.org/abs/2306.05813

Please cite our paper if you used this repository in your work.

For the results presented in the paper, including trained PAAE models please see [here](bit.ly/phcavelar_paae_2023).

To setup the conda environment, please use the `setup_conda_environment.sh` script. Then run `external-internal.sh` to run the experiments. Run `viz-brca.ipynb` after the internal validation has finished to select the best models. Then you may run `viz-brca-external.ipynb` to generate the main plots. For the interpretability experiments you may run `dists-brca-external.ipynb` and `dists-brca-external-featureimportance.ipynb`, setting the variables accordingly to select specific models to analyse.

## Data Sources

[Xenabrowser](https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA))

[Metabric](https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz)

### Pathway definitions

Pathway definitions in the data folder or [here](https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp)

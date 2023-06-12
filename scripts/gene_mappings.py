import functools
from os.path import join, expanduser, expandvars
import time
import requests
import json

import pandas as pd

from tqdm import tqdm

@functools.cache
def ensembl_id_has_display_name(ensembl_id):
    return "display_name" in ensembl_id_request(ensembl_id)

@functools.cache
def ensembl_id_to_display_name(ensembl_id):
    return ensembl_id_request(ensembl_id)["display_name"] if ensembl_id_has_display_name(ensembl_id) else ensembl_id

_ENSEMBL_REST_API_URL = "https://rest.ensembl.org/lookup/id/{ensembl_id}?content-type=application/json"
_ENSEMBL_REST_API_RETRIES = 5
_ENSEMBL_REST_API_RETRY_DELAY = 1.0

_GENENAMES_API_URL = "https://www.genenames.org/cgi-bin/download/custom?col=gd_hgnc_id&col=gd_app_sym&col=gd_app_name&col=gd_status&col=gd_prev_sym&col=gd_aliases&col=gd_pub_chrom_map&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&col=gd_pub_ensembl_id&col=gd_prev_name&col=md_ensembl_id&status=Approved&status=Entry%20Withdrawn&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"

def ensembl_id_request(ensembl_id):
    return _ensembl_id_request(ensembl_id if "." not in ensembl_id else ensembl_id.split(".")[0])

@functools.lru_cache(128)
def _ensembl_id_request(ensembl_id):
    current_delay = _ENSEMBL_REST_API_RETRY_DELAY
    for _ in range(_ENSEMBL_REST_API_RETRIES):
        try:
            reply = requests.get(_ENSEMBL_REST_API_URL.format(ensembl_id=ensembl_id))
            if reply.status_code == 200:
                break
            elif reply.status_code == 400:
                return {}
            elif reply.status_code == 429:
                current_delay = max(current_delay*2, (float(reply.headers["X-RateLimit-Reset"])+1))
            else:
                raise NotImplementedError("Ensembl API replied with unforeseen status code {status_code}! Content: {content}".format(status_code=reply.status_code, content=reply.content))
        except ConnectionError as e:
            current_delay *= 2
        time.sleep(current_delay)
    else:
        raise Exception("Ensembl API did not reply after {retries} retries!".format(retries=_ENSEMBL_REST_API_RETRIES))
    obj = json.loads(reply.text)
    reply.close()
    return obj

def main(base_path = "~/data/pathwayae/"):
    base_path = expanduser(expandvars(base_path))
    raw_path = join(base_path, "raw")
    raw_tcga_path = join(raw_path, "tcga")
    probemap_fname = "gencode.v22.annotation.gene.probeMap"
    genemap_fname = "ensembl_to_gene_id.json"

    probemap = pd.read_csv(join(raw_tcga_path,probemap_fname), sep="\t", index_col=0)
    with open(join(raw_tcga_path,genemap_fname)) as f: genemap = json.load(f)

    genemap_eids = set(genemap.keys())
    probemap_eids = set(probemap.index)
    both_eids = genemap_eids.intersection(probemap_eids)
    both_eids_list = sorted(both_eids)

    ignore = {'assembly_name', 'biotype', 'canonical_transcript', 'db_type', 'description', 'end', "id", 'logic_name', 'object_type', 'seq_region_name', 'source', 'species', 'start', 'strand', 'version'}

    print(len(genemap_eids), len(probemap_eids), len(both_eids))

    for eid in tqdm(both_eids_list):
        if genemap[eid] != probemap.loc[eid,"gene"]:
            try:
                ans = ensembl_id_request(eid)
            except KeyboardInterrupt:
                break
            except Exception as e:
                ans = {"display_name": "", "ze_error": e}
            if "display_name" not in ans or ans["display_name"] != genemap[eid]:
                tqdm.write(
                    str((eid, genemap[eid], probemap.loc[eid,"gene"], *[f"{k}={ans[k]}" for k in ans.keys() if k not in ignore]))
                )
                

if __name__=="__main__":
    main()
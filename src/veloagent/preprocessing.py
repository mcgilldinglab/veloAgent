import scvelo as scv
import scanpy as sc
import numpy as np
import anndata
from scipy.sparse import csr_matrix


def preprocess(data, num_genes=2000, min_count=20):
    data.var.index.name = "index"
    data.layers['spliced'] = csr_matrix(data.layers['spliced'].astype(np.float32))
    data.layers['unspliced'] = csr_matrix(data.layers['unspliced'].astype(np.float32))
    scv.pp.filter_and_normalize(data, min_shared_counts=min_count, n_top_genes=num_genes)
    data.X = np.nan_to_num(data.X, nan=0)
    sc.pp.pca(data)
    scv.pp.neighbors(data, n_neighbors=30)
    scv.tl.umap(data)
    scv.pp.moments(data, n_pcs=None, n_neighbors=30)
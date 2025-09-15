import scvelo as scv
import scanpy as sc
import numpy as np
import anndata
from scipy.sparse import csr_matrix


def preprocess(data, num_genes=2000, min_count=20):
    """
    Preprocess an AnnData object for RNA velocity analysis and VAE training.

    This function performs:
    1. Conversion of spliced/unspliced layers to sparse float32 matrices.
    2. Gene filtering and normalization (selecting highly variable genes).
    3. Replacement of NaN values in `data.X` with zeros.
    4. Computation of a nearest-neighbor graph.
    5. Computation of first- and second-order moments (required for velocity estimation).

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object containing spliced/unspliced count layers.
    num_genes : int, optional (default: 2000)
        Number of highly variable genes to retain.
    min_count : int, optional (default: 20)
        Minimum number of shared counts required for gene filtering.

    Returns
    -------
    None
        Updates the input AnnData object in place with:
        - Filtered and normalized expression values
        - Updated `data.X`
        - Nearest-neighbor graph (`data.uns["neighbors"]`)
        - Moments for velocity analysis (`data.layers["Ms"]`, `data.layers["Mu"]`)
    """
    # Ensure gene index is named
    data.var.index.name = "index"

    data.layers['spliced'] = csr_matrix(data.layers['spliced'].astype(np.float32))
    data.layers['unspliced'] = csr_matrix(data.layers['unspliced'].astype(np.float32))

    # Filter lowly expressed genes, normalize counts, and select HVGs
    scv.pp.filter_and_normalize(data, min_shared_counts=min_count, n_top_genes=num_genes)

    # Replace any NaN values in main expression matrix
    data.X = np.nan_to_num(data.X, nan=0)

    # Compute k-nearest neighbors and velocity moments
    scv.pp.neighbors(data, n_neighbors=30)
    scv.pp.moments(data, n_pcs=None, n_neighbors=30)
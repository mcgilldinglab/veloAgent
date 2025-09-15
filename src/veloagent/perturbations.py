import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import statsmodels.stats.multitest
from veloproj import *


# scoring metric code
def perturbation_score(
        data, 
        cluster_name, 
        cluster_edges, 
        vel_key: str = 'velocity_u', 
        metric_option: int = 1, 
        pert_param: str = "alpha", 
        dt: float = 0.5
    ):
    """
    Compute per-gene perturbation scores for RNA velocity analysis by
    zeroing out a kinetic parameter (alpha or beta) and assessing the
    impact on velocity dynamics.

    Parameters
    ----------
    data : AnnData
        AnnData object with velocity-related layers:
        'alpha', 'beta', 'gamma', 'Mu', 'Ms', and `vel_key`.
    cluster_name : str
        Column name in `adata.obs` containing cluster assignments.
    cluster_edges : list
        List/tuple of clusters of interest (used in evaluation).
    vel_key : str, default='velocity_u'
        Key in `.layers` corresponding to velocity matrix to perturb.
    metric_option : int, default=1
        Evaluation mode:
        - 1: Evaluation via cross-boundary correctness.
        - 2: Evaluation via cosine similarity.
    pert_param : str, default="alpha"
        Kinetic parameter to perturb (either "alpha" or "beta").
    dt : float, default=0.5
        Time-step scaling factor for velocity computation.

    Returns
    -------
    score_df : pd.DataFrame
        DataFrame with index = gene names (`adata.var_names`) and one column:
        - "score": perturbation score for each gene.

    Notes
    -----
    - `metric_option=1` is computationally expensive since it recomputes
    the velocity graph and embedding for each gene.
    - `metric_option=2` is faster and computes cosine similarity between
    perturbed and original velocity fields.
    """
    def cos_sim(A, B):
        """Row-wise cosine similarity between matrices A and B."""
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        return np.sum(A_norm * B_norm, axis=1)

    # For option 2, restrict data to specified clusters
    if metric_option == 2:
        data = data[data.obs[cluster_name].isin(cluster_edges)].copy()

    # Initialize results dataframe
    score_df = pd.DataFrame(0.0, index=data.var_names, columns=["score"])

    for i in range(data.n_vars):
        if i % 100 == 0:
            print(f"Processing gene {i}/{data.n_vars}")

        if metric_option == 1:
            idata = data.copy()

            # Perturb kinetic parameter for gene i
            if pert_param == "alpha":
                idata.layers['alpha'][:, i] = 0
                a, b, u = idata.layers['alpha'], idata.layers['beta'], idata.layers['Mu']
                idata.layers[vel_key] = (a - b*u) * dt
            elif pert_param == "beta":
                idata.layers['beta'][:, i] = 0
                b, g, s, u = idata.layers['beta'], idata.layers['gamma'], idata.layers['Ms'], idata.layers['Mu']
                idata.layers[vel_key] = (b*u - g*s) * dt
            
            # Recompute velocity graph & embedding
            scv.tl.velocity_graph(idata, vkey=vel_key)
            scv.tl.velocity_confidence(idata, vkey=vel_key)
            scv.tl.velocity_embedding(idata, vkey=vel_key, basis='umap')

            # Evaluate cluster-boundary correctness
            eval_res = evaluate(
                idata, cluster_edges, cluster_name,
                k_velocity=vel_key, x_emb='X_umap', verbose=False
            )
            scores = np.concatenate(list(eval_res["Cross-Boundary Direction Correctness (A->B)"].values()))
            mean_score = np.mean(scores)

        elif metric_option == 2:
            pert_vel = data.layers[vel_key].copy()

            # Perturb velocity for gene i
            if pert_param == "alpha":
                a = data.layers['alpha']
                b = data.layers['beta']
                u = data.layers['Mu']
                pert_vel[:, i] = (a[:, i] - b[:, i] * u[:, i]) * dt
            elif pert_param == "beta":
                b = data.layers['beta']
                g = data.layers['gamma']
                s = data.layers['Ms']
                u = data.layers['Mu']
                pert_vel[:, i] = (b[:, i] * u[:, i] - g[:, i] * s[:, i]) * dt

            # Cosine similarity with original velocities
            scores = cos_sim(data.layers[vel_key], pert_vel)
            mean_score = np.mean(scores)

        score_df.iloc[i, 0] = mean_score

    return score_df


def perturb(adata, gene_list, pert_param, dt: float = 0.5):
    """
    Perturb kinetic parameters (alpha or beta) for a given set of genes 
    in an AnnData object and recompute RNA velocity.

    This function sets the chosen kinetic parameter (`alpha` or `beta`) 
    to zero for the specified genes, then recalculates the velocity layer 
    using the modified parameters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with velocity-related layers:
        'alpha', 'beta', 'gamma', 'Mu', 'Ms'.
    gene_list : list of str
        List of gene names (matching `adata.var_names`) to perturb.
    pert_param : str
        Kinetic parameter to perturb. Must be one of:
        - "alpha": set alpha = 0 for specified genes.
        - "beta": set beta = 0 for specified genes.
    dt : float, default=0.5
        Time-step scaling factor used in velocity calculation.

    Returns
    -------
    bdata : AnnData
        A copy of the input AnnData with the perturbed kinetic 
        parameters and recomputed velocity in `.layers['velocity']`.
    """
    bdata = adata.copy()

    for gene in gene_list:
        gene_index = np.where(adata.var_names == gene)[0]
        if len(gene_index) == 0:
            print(f"Warning: {gene} not found, skipping.")
            continue

        # Zero out the chosen parameter for gene
        if (pert_param == "alpha"):
            bdata.layers['alpha'][:, gene_index] = 0
        
        elif (pert_param == "beta"):
            bdata.layers['beta'][:, gene_index] = 0

    a, b, g = bdata.layers['alpha'], bdata.layers['beta'], bdata.layers['gamma']
    u, s = bdata.layers['Mu'], bdata.layers['Ms']

    # Recompute velocity
    if pert_param == "alpha":
        bdata.layers['velocity'] = (a - b*u) * dt
    elif pert_param == "beta":
        bdata.layers['velocity'] = (b*u - g*s) * dt

    return bdata


# PERTURBATION SCORE PLOT
def perturb_score_plt(scores, gene_list):
    """
    Plot the top perturbation scores as a bar chart.

    Parameters
    ----------
    scores : pd.DataFrame
        DataFrame containing a column 'score' with perturbation scores 
        (e.g., output from `perturbation_score`).
    gene_list : list of str
        List of gene names corresponding to the rows in `scores`. 
        Must be the same length/order as `scores`.
    """
    scores = scores.copy()

    top_25 = scores.sort_values(by='score', ascending=False).head(25)
    top_25['gene'] = gene_list

    plt.figure(figsize=(12, 6))
    plt.bar(top_25['gene'], top_25['score'], color='skyblue')

    # Add labels and title
    plt.xlabel('Gene', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Breast Cancer Perturbation', fontsize=14)

    # Rotate gene labels and adjust size
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    plt.yscale('log')

    # Show the plot
    plt.show()
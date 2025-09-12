
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scvelo as scv
import statsmodels.stats.multitest
from veloproj import *


# scoring metric code
def perturbation_score(data, cluster_name, cluster_edges, vel_key='velocity', metric_option=2, pert_param="alpha", dt=0.5):
    def cos_sim(A, B):
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)
        return np.sum(A_norm * B_norm, axis=1)

    # filter clusters for option 2
    if metric_option == 2:
        data = data[data.obs[cluster_name].isin(cluster_edges)].copy()

    score_df = pd.DataFrame(0.0, index=data.var_names, columns=["score"])

    for i in range(data.n_vars):
        if i % 100 == 0:  # simple progress indicator
            print(f"Processing gene {i}/{data.n_vars}")

        if metric_option == 1:
            idata = data.copy()

            if pert_param == "alpha":
                idata.layers['alpha'][:, i] = 0
                a, b, u = idata.layers['alpha'], idata.layers['beta'], idata.layers['Mu']
                idata.layers[vel_key] = (a - b*u) * dt
            elif pert_param == "beta":
                idata.layers['beta'][:, i] = 0
                b, g, s, u = idata.layers['beta'], idata.layers['gamma'], idata.layers['Ms'], idata.layers['Mu']
                idata.layers[vel_key] = (b*u - g*s) * dt
            
            scv.tl.velocity_graph(idata, vkey=vel_key)
            scv.tl.velocity_confidence(idata, vkey=vel_key)
            scv.tl.velocity_embedding(idata, vkey=vel_key, basis='umap')

            eval_res = evaluate(
                idata, cluster_edges, cluster_name,
                k_velocity=vel_key, x_emb='X_umap', verbose=False
            )
            scores = np.concatenate(list(eval_res["Cross-Boundary Direction Correctness (A->B)"].values()))
            mean_score = np.mean(scores)

        elif metric_option == 2:
            pert_vel = data.layers[vel_key].copy()

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

            scores = cos_sim(data.layers[vel_key], pert_vel)
            mean_score = np.mean(scores)

        score_df.iloc[i, 0] = mean_score

    return score_df


# plot histogram of scores
def plt_hist(scores):
    scores.hist(bins=100, figsize=(10, 6))
    plt.xlabel('Score')
    plt.ylabel('Num Genes')
    plt.tight_layout()
    plt.show()


def perturb(adata, gene_list, pert_param, dt=0.5):
    bdata = adata.copy()

    for gene in gene_list:
        # Find the index of the gene
        gene_index = np.where(adata.var_names == gene)[0]
        if len(gene_index) == 0:
            print(f"Warning: {gene} not found, skipping.")
            continue

        if (pert_param == "alpha"):
            bdata.layers['alpha'][:, gene_index] = 0
        
        elif (pert_param == "beta"):
            bdata.layers['beta'][:, gene_index] = 0

    a, b, g = bdata.layers['alpha'], bdata.layers['beta'], bdata.layers['gamma']
    u, s = bdata.layers['Mu'], bdata.layers['Ms']

    if pert_param == "alpha":
        bdata.layers['velocity'] = (a - b*u) * dt

    elif pert_param == "beta":
        bdata.layers['velocity'] = (b*u - g*s) * dt

    return bdata

å
# PERTURBATION SCORE PLOT
def perturb_score_plt(scores):
    # Sort DataFrame by 'score' in descending order and select the top 25 rows
    top_25 = scores.sort_values(by='score', ascending=False).head(25).reset_index()

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(top_25['index'], top_25['score'], color='skyblue')

    # Add labels and title
    plt.xlabel('gene', fontsize=12)
    plt.ylabel('score', fontsize=12)
    plt.title('Breast Cancer Perturbation', fontsize=14)
    # Reduce the fontsize for tick labels
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()

    plt.yscale('log')

    # Show the plot
    plt.show()
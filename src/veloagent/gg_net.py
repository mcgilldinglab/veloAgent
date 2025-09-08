import torch
import logging
import time
import torch.nn as nn
import torch.nn.init as init
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import leidenalg
import anndata
import scipy
import scvelo as scv
import math
import torch.nn.functional as F
from scipy.sparse import csr_matrix, issparse, spmatrix
from typing import Union
from numpy import ndarray
from torch_geometric.nn import Sequential, GCNConv
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from velovi import VELOVI
from memory_profiler import memory_usage


logging.basicConfig(level=logging.INFO)

def load_protein_paths(species, base):
    if species == "mouse":
        return [f"{base}/mouse/10090.protein.info.v12.0.txt",
            f"{base}/mouse/10090.protein.aliases.v12.0.txt",
            f"{base}/mouse/10090.protein.links.v12.0.txt"]
        
    elif species == "chicken":
        return [f"{base}/chicken/9031.protein.info.v12.0.txt",
            f"{base}/chicken/9031.protein.aliases.v12.0.txt",
            f"{base}/chicken/9031.protein.links.v12.0.txt"]
        
    elif species == "human":
        return [f"{base}/human/9606.protein.info.v12.0.txt",
            f"{base}/human/9606.protein.aliases.v12.0.txt",
            f"{base}/human/9606.protein.links.v12.0.txt"]


def create_con_mat(data, num_genes, prot_names, prot_alias, gene_conn, varname):
    
    # create protein alias
    def get_alias(data, prot_names, prot_alias):
        names = pd.read_csv(prot_names, sep='\t', usecols=[0,1])
        names = names[names['alias'].isin(data.var_names)]
        alias = pd.read_csv(prot_alias, sep='\t', usecols=[0,1])
        alias = alias[alias['alias'].isin(data.var_names)]
        alias = alias[~alias['alias'].duplicated(keep='first')]
    
        # find gene synonymes protein id
        merged = pd.merge(alias, names, on='alias', how='left', indicator=True)
        missing = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
        missing = missing.drop(columns=['protein_id_y'])
        missing = missing.rename(columns={'protein_id_x': 'protein_id'})
    
        merged_alias = pd.merge(names, missing, how='outer')
        merged_alias = merged_alias.reset_index(drop=True)
        
        return merged_alias

    alias = get_alias(data, prot_names, prot_alias)
    
    genes = pd.DataFrame(data.var.index)
    genes.rename(columns={varname: 'gene'}, inplace=True)
    
    # Merge the gene list with the alias DataFrame, keeping all genes and their corresponding proteins
    proteins = pd.merge(genes, alias, left_on='gene', right_on='alias', how='left')
    
    # Keep only the gene and protein columns
    proteins = proteins[['gene', 'protein_id']]
    
    gpair = pd.read_csv(gene_conn, sep=' ')
        
    # prune relevant connections
    protein_set = set(alias['protein_id'])
    gpair = gpair[(gpair['protein1'].isin(protein_set)) & (gpair['protein2'].isin(protein_set))].reset_index(drop=True)
    
    # convert proteins to index
    protein_to_idx = {protein: idx for idx, protein in enumerate(proteins['protein_id'])}
    gpair['protein1_idx'] = gpair['protein1'].map(protein_to_idx)
    gpair['protein2_idx'] = gpair['protein2'].map(protein_to_idx)
    
    rows = gpair['protein1_idx']
    cols = gpair['protein2_idx']
    mat = np.ones(len(gpair), dtype=int)
    
    connection_mat = csr_matrix((mat, (rows, cols)), shape=(num_genes, num_genes))
    dense_mat = connection_mat.toarray()

    # Step 1: Identify empty rows and columns
    empty_rows = np.all(dense_mat == 0, axis=1)
    
    # Step 2: Connect empty rows and columns to non-empty ones
    # Here, I'm connecting empty rows and columns to non-empty ones by setting their values to 1
    for i in range(dense_mat.shape[0]):
        if empty_rows[i]:  # If the row is empty
            # Connect it to a non-empty row (you can choose a method for this)
            dense_mat[i,:] = 1  # Set the row's empty columns to 1
            dense_mat[:,i] = 1  # Set the column's empty columns to 1

    np.fill_diagonal(dense_mat, 1)
    
    return dense_mat


# Define custom autograd function for masked connection.
class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.

        Arguments
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class GeneNet(nn.Module):
    def __init__(self, in_dim, gene_dim, conn=None):
        self.num_genes = gene_dim
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.lyr1 = nn.Linear(in_dim, gene_dim//2)
        self.bn1 = nn.BatchNorm1d(gene_dim//2)
        self.lyr2 = nn.Linear(gene_dim//2, gene_dim)
        self.bn2 = nn.BatchNorm1d(gene_dim)
        self.lyr3 = CustomizedLinear(torch.tensor(conn.T), bias=None)
        self.bn3 = nn.BatchNorm1d(gene_dim)
        self.lyr4 = nn.Linear(gene_dim, 2*gene_dim)
        self.bn4 = nn.BatchNorm1d(2*gene_dim)
        self.lyr5 = nn.Linear(2*gene_dim, 3*gene_dim)
        
    def forward(self, x):
        layer1 = self.dropout(F.leaky_relu(self.bn1(self.lyr1(x))))
        layer2 = self.dropout(F.leaky_relu(self.bn2(self.lyr2(layer1))))
        layer3 = self.dropout(F.leaky_relu(self.bn3(self.lyr3(layer2))))
        layer4 = self.dropout(F.leaky_relu(self.bn4(self.lyr4(layer3))))
        layer5 = self.lyr5(layer4)
        output = torch.sigmoid(layer5)
        return output


# ABLATION
class GeneNetAblation(nn.Module):
    def __init__(self, in_dim, gene_dim, conn=None):
        self.num_genes = gene_dim
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.lyr1 = nn.Linear(in_dim, gene_dim//2)
        self.bn1 = nn.BatchNorm1d(gene_dim//2)
        self.lyr2 = nn.Linear(gene_dim//2, gene_dim)
        self.bn2 = nn.BatchNorm1d(gene_dim)
        self.lyr3 = nn.Linear(gene_dim, gene_dim)
        self.bn3 = nn.BatchNorm1d(gene_dim)
        self.lyr4 = nn.Linear(gene_dim, 2*gene_dim)
        self.bn4 = nn.BatchNorm1d(2*gene_dim)
        self.lyr5 = nn.Linear(2*gene_dim, 3*gene_dim)
        
    def forward(self, x):
        layer1 = self.dropout(F.leaky_relu(self.bn1(self.lyr1(x))))
        layer2 = self.dropout(F.leaky_relu(self.bn2(self.lyr2(layer1))))
        layer3 = self.dropout(F.leaky_relu(self.bn3(self.lyr3(layer2))))
        layer4 = self.dropout(F.leaky_relu(self.bn4(self.lyr4(layer3))))
        layer5 = self.lyr5(layer4)
        output = torch.sigmoid(layer5)
        return output


def velo_pred(spliced, unspliced, rates, umax, smax, dt):
    
    num_genes = len(spliced[0])

    alpha = rates[:,0:num_genes]
    beta = rates[:,num_genes:2*num_genes]
    gamma = rates[:,2*num_genes:3*num_genes]

    alpha = alpha * umax
    beta = beta * smax
    gamma = gamma / smax

    vel_u = (alpha - beta*unspliced)*dt
    vel = (beta*unspliced - gamma*spliced)*dt
    
    pred_unspliced = unspliced + vel_u
    pred_spliced = spliced + vel
    
    return pred_spliced, pred_unspliced, alpha, beta, gamma, vel, vel_u


def cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices):
    """
    Return:
    list of cosine distance and a list of the index of the next cell
    """
    uv, sv = pred_unspliced-unspliced, pred_spliced-spliced # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
    unv, snv = unspliced[indices.T[1:]] - unspliced, spliced[indices.T[1:]] - spliced # Velocity from (unsplice, splice) to its neighbors

    eps = 1e-12
    den = torch.sqrt(unv**2 + snv**2 + eps) * torch.sqrt(uv**2 + sv**2 + eps)
    den[den==0] = -1
    cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: column -> individual cell (cellI); row -> nearby cells of cell id ; value -> cosine between col and row cells
    cosine_max, cosine_max_idx = torch.max(cosine, dim=0)
    
    return 1 - cosine_max


def nbr_cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices, vel, vel_u):
    """
    Return:
    list of cosine distance and a list of the index of the next cell
    """
    uv, sv = pred_unspliced-unspliced, pred_spliced-spliced # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
    unv, snv = vel_u[indices.T[1:]], vel[indices.T[1:]] # Velocity from (unsplice, splice) to its neighbors

    eps = 1e-12
    den = torch.sqrt(unv**2 + snv**2 + eps) * torch.sqrt(uv**2 + sv**2 + eps)
    den[den==0] = -1
    cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: column -> individual cell (cellI); row -> nearby cells of cell id ; value -> cosine between col and row cells
    cosine_sum = cosine.sum()
    return cosine_sum


def adj_velocity(data, velocity, indices):
    
    adj_vel = velocity

    for j in range(data.n_obs):
        vel_n = velocity[indices[j][:]]
        vel_n = vel_n.mean(dim=0)
        adj_vel[j] = vel_n

    return adj_vel


def train_gg(num_epochs, data, embed_basis, genenet, device, optimizer, patience=10, num_nbrs=30, dt=0.3, batch=0.25):
    umax = torch.max(torch.tensor(data.layers['Mu']), dim=0)[0]
    smax = torch.max(torch.tensor(data.layers['Ms']), dim=0)[0]
    umax[umax == 0] = 1
    smax[smax == 0] = 1

    best_net = None
    best_loss = -1
    imp_counter = 0
    
    with trange(num_epochs) as pbar:
        for epoch in pbar:
            
            subsample_idx = np.random.choice(data.n_obs, size=int(data.n_obs * batch), replace=False)
            subsample = data[subsample_idx, :]

            latent = torch.tensor(subsample.obsm[embed_basis], dtype=torch.float32).to(device)
            spliced = torch.tensor(subsample.layers['Ms'], dtype=torch.float32).to(device)
            unspliced = torch.tensor(subsample.layers['Mu'], dtype=torch.float32).to(device)

            nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(latent)

            distances, indices = nbrs.kneighbors(latent)

            indices = torch.tensor(indices)

            outputs = genenet(latent)

            pred_spliced, pred_unspliced, alpha, beta, gamma, velocity, velocity_u = velo_pred(spliced, unspliced, outputs, umax, smax, dt)

            cosim = cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices)
            
            gene_loss = torch.sum(cosim)

            loss = gene_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN or Inf detected in loss, exiting")
                break

            if loss < best_loss or best_loss == -1:
                best_net = genenet.state_dict()
                best_loss = loss
                imp_counter = 0
            else:
                imp_counter += 1

            if imp_counter > patience:
                logging.info("Early stopping triggered")
                break
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    genenet.load_state_dict(best_net)

    spliced = torch.tensor(data.layers['Ms'], dtype=torch.float32).to(device)
    unspliced = torch.tensor(data.layers['Mu'], dtype=torch.float32).to(device)
    latent = torch.tensor(data.obsm[embed_basis], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        rate_vals = genenet(latent)
        pred_spliced, pred_unspliced, alpha, beta, gamma, velocity, velocity_u = velo_pred(spliced, unspliced, rate_vals, umax, smax, dt)
        data.layers["velocity_u"] = velocity_u.detach().numpy()
        data.layers["velocity"] = velocity.detach().numpy()
        data.layers["alpha"] = alpha.detach().numpy()
        data.layers["beta"] = beta.detach().numpy()
        data.layers["gamma"] = gamma.detach().numpy()
    
    logging.info("Training complete")


def train_nbr(num_epochs, data, embed_basis, genenet, device, optimizer, num_nbrs=30, dt=0.3, batch=0.25):
    umax = torch.max(torch.tensor(data.layers['Mu']), dim=0)[0]
    smax = torch.max(torch.tensor(data.layers['Ms']), dim=0)[0]
    umax[umax == 0] = 1
    smax[smax == 0] = 1
    
    for i in range(num_epochs):
        
        subsample_idx = np.random.choice(data.n_obs, size=int(data.n_obs * batch), replace=False)
        subsample = data[subsample_idx, :]

        latent = torch.tensor(subsample.obsm[embed_basis], dtype=torch.float32).to(device)
        spliced = torch.tensor(subsample.layers['Ms'], dtype=torch.float32).to(device)
        unspliced = torch.tensor(subsample.layers['Mu'], dtype=torch.float32).to(device)
        velocity = torch.tensor(subsample.layers['velocity'])

        nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(latent)

        distances, indices = nbrs.kneighbors(latent)

        indices = torch.tensor(indices)

        outputs = genenet(latent)
        
        pred_spliced, pred_unspliced, alpha, beta, gamma, velocity, velocity_u = velo_pred(spliced, unspliced, outputs, umax, smax, dt)

        cosim = nbr_cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices, velocity, velocity_u)
        
        gene_loss = -cosim/len(subsample)

        loss = gene_loss

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning("NaN or Inf detected in loss, exiting")
            break
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    spliced = torch.tensor(data.layers['Ms'], dtype=torch.float32).to(device)
    unspliced = torch.tensor(data.layers['Mu'], dtype=torch.float32).to(device)
    latent = torch.tensor(data.obsm[embed_basis], dtype=torch.float32).to(device)

    nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(latent)
    distances, indices = nbrs.kneighbors(latent)
    indices = torch.tensor(indices)
    
    with torch.no_grad():
        rate_vals = genenet(latent)
        pred_spliced, pred_unspliced, alpha, beta, gamma, velocity, velocity_u = velo_pred(spliced, unspliced, rate_vals, umax, smax, dt)
        velocity = adj_velocity(data, velocity, indices)
        data.layers["velocity_u"] = velocity_u.detach().numpy()
        data.layers["velocity"] = velocity.detach().numpy()
        data.layers["alpha"] = alpha.detach().numpy()
        data.layers["beta"] = beta.detach().numpy()
        data.layers["gamma"] = gamma.detach().numpy()
        
    logging.info("Training complete")
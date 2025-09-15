import torch
import torch.nn as nn
import sys
import os
import numpy as np
import scanpy as sc
import math
from torch_geometric.nn import Sequential
from tqdm import trange
import scvelo as scv


class Encoder(nn.Module):
    """
    Variational autoencoder (VAE) encoder.

    This encoder processes two distinct input modalities (unspliced `u` 
    and spliced `s` counts), projects them into latent embeddings, and 
    outputs parameters for a Gaussian latent distribution.

    Architecture
    ------------
    - Two parallel subnetworks (`net_u`, `net_s`) process unspliced 
      and spliced inputs independently into latent representations.
    - Each subnetwork uses linear layers with LeakyReLU activations, 
      dropout for regularization, and BatchNorm for stability.
    - The two embeddings are concatenated and transformed into 
      Gaussian parameters:
        - `fc_mu_z`: latent mean
        - `fc_std_z`: latent variance (Softplus applied to ensure positivity).

    Parameters
    ----------
    in_dim : int
        Input dimensionality (number of genes).
    z_dim : int
        Latent dimensionality (size of encoded representation).

    Forward Inputs
    --------------
    u : torch.Tensor
        Unspliced count input of shape (batch_size, in_dim).
    s : torch.Tensor
        Spliced count input of shape (batch_size, in_dim).
    edge_index : torch.Tensor
        Graph connectivity (currently unused in this implementation, 
        included for compatibility with GNN-based extensions).

    Forward Outputs
    ---------------
    mu : torch.Tensor
        Latent mean vector of shape (batch_size, z_dim).
    log_var : torch.Tensor
        Latent variance vector (positive) of shape (batch_size, z_dim).

    Notes
    -----
    - Xavier uniform initialization is applied to all Linear layers.
    - BatchNorm weights are initialized to 1, biases to 0.
    - Dropout (p=0.2) applied after activations to reduce overfitting.
    """
    
    def __init__(self, in_dim, z_dim):
        super(Encoder, self).__init__()

        # Hidden dimensions scale between input and latent sizes
        dim_diff = in_dim - z_dim
        hd1 = z_dim + 2*dim_diff//3
        hd2 = z_dim + dim_diff//3

        # Subnetwork for unspliced counts
        self.net_u = nn.Sequential(nn.Linear(in_dim, hd1), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd2, z_dim), nn.BatchNorm1d(z_dim), nn.LeakyReLU(), nn.Dropout(p=0.2))
        
        # Subnetwork for spliced counts
        self.net_s = nn.Sequential(nn.Linear(in_dim, hd1), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd2, z_dim), nn.BatchNorm1d(z_dim), nn.LeakyReLU(), nn.Dropout(p=0.2))
        
        # Fully connected layers to parameterize Gaussian latent distribution
        self.fc_mu_z = nn.Linear(z_dim*2, z_dim)
        self.fc_std_z = nn.Linear(z_dim*2, z_dim)
        self.spt = nn.Softplus() # ensures variance > 0
        
        self.act = nn.LeakyReLU()
        
        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for linear and batchnorm layers."""
        for m in self.net_u.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.net_s.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in [self.fc_mu_z, self.fc_std_z]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, u, s, edge_index):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        u : torch.Tensor
            Unspliced count matrix, shape (batch_size, in_dim).
        s : torch.Tensor
            Spliced count matrix, shape (batch_size, in_dim).
        edge_index : torch.Tensor
            Graph connectivity (unused, placeholder for GNN extensions).

        Returns
        -------
        mu : torch.Tensor
            Latent mean vector of shape (batch_size, z_dim).
        log_var : torch.Tensor
            Latent variance vector of shape (batch_size, z_dim).
        """
        # Encode unspliced and spliced separately
        h_u = self.net_u(u)
        h_s = self.net_s(s)

        # Concatenate embeddings
        h = torch.cat((h_s.squeeze(0), h_u.squeeze(0)), dim=1)

        # Compute latent distribution parameters
        mu = self.fc_mu_z(h)
        log_var = self.spt(self.fc_std_z(h))

        return mu, log_var

class Decoder(nn.Module):
    """
    Variational autoencoder (VAE) decoder for RNA velocity modeling.

    The decoder reconstructs unspliced (`u`) and spliced (`s`) gene 
    expression profiles from a latent representation `z`.

    Architecture
    ------------
    - Two parallel subnetworks (`net_u`, `net_s`) map the latent space 
      back to the input dimension.
    - Each subnetwork consists of:
        - Linear → LeakyReLU
        - Linear → LeakyReLU
        - Linear → LeakyReLU
    - Hidden layer sizes scale between `z_dim` and `in_dim` for smooth 
      dimensionality transition.

    Parameters
    ----------
    in_dim : int
        Input dimensionality (number of genes).
    z_dim : int
        Latent dimensionality (size of encoded representation).

    Forward Inputs
    --------------
    z_data : torch.Tensor
        Latent representation of shape (batch_size, z_dim).

    Forward Outputs
    ---------------
    reconstructed_u : torch.Tensor
        Reconstructed unspliced counts, shape (batch_size, in_dim).
    reconstructed_s : torch.Tensor
        Reconstructed spliced counts, shape (batch_size, in_dim).
    """
    
    def __init__(self, in_dim, z_dim):
        super(Decoder, self).__init__()

        # Hidden dimensions scale between z_dim and in_dim
        dim_diff = in_dim - z_dim
        hd1 = z_dim + dim_diff//3
        hd2 = z_dim + 2*dim_diff//3

        # Decoder network for unspliced counts
        self.net_u = nn.Sequential(nn.Linear(z_dim, hd1), nn.LeakyReLU(),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(),
                                   nn.Linear(hd2, in_dim), nn.LeakyReLU())
        
        # Decoder network for spliced counts
        self.net_s = nn.Sequential(nn.Linear(z_dim, hd1), nn.LeakyReLU(),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(),
                                   nn.Linear(hd2, in_dim), nn.LeakyReLU())
    
    def forward(self, z_data):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        z_data : torch.Tensor
            Latent embedding, shape (batch_size, z_dim).

        Returns
        -------
        reconstructed_u : torch.Tensor
            Reconstructed unspliced counts.
        reconstructed_s : torch.Tensor
            Reconstructed spliced counts.
        """
        reconstructed_u = self.net_u(z_data)
        reconstructed_s = self.net_s(z_data)
        return reconstructed_u, reconstructed_s

class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) for RNA velocity representation learning.

    The VAE consists of:
    - Encoder: Projects unspliced (`u`) and spliced (`s`) gene expression 
      into a latent distribution (mean `mu` and variance `var`).
    - Reparameterization: Samples latent embeddings `z` using the 
      reparameterization trick.
    - Decoder: Reconstructs unspliced and spliced counts from latent embeddings.

    Parameters
    ----------
    in_dim : int
        Input dimensionality (number of genes).
    z_dim : int
        Latent dimensionality (size of the encoded space).

    Forward Inputs
    --------------
    u : torch.Tensor
        Unspliced counts, shape (batch_size, in_dim).
    s : torch.Tensor
        Spliced counts, shape (batch_size, in_dim).
    edge_index : torch.Tensor
        Graph connectivity (not used in this version but passed for 
        potential graph-based extensions).

    Forward Outputs
    ---------------
    mu : torch.Tensor
        Mean of the latent distribution, shape (batch_size, z_dim).
    log_var : torch.Tensor
        Log-variance of the latent distribution, shape (batch_size, z_dim).
    reconstructed_u : torch.Tensor
        Reconstructed unspliced counts, shape (batch_size, in_dim).
    reconstructed_s : torch.Tensor
        Reconstructed spliced counts, shape (batch_size, in_dim).
    z_reparam : torch.Tensor
        Latent embedding sampled via reparameterization, shape (batch_size, z_dim).
    """
    
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(in_dim, z_dim)
        self.decoder = Decoder(in_dim, z_dim)
        
    def reparameterization(self, mu, var):
        """
        Reparameterization trick to sample latent embeddings.

        z = mu + sigma * epsilon
        where epsilon ~ N(0, I).

        Parameters
        ----------
        mu : torch.Tensor
            Mean of latent distribution, shape (batch_size, z_dim).
        var : torch.Tensor
            Standard deviation of latent distribution, shape (batch_size, z_dim).

        Returns
        -------
        z_reparam : torch.Tensor
            Sampled latent embeddings, shape (batch_size, z_dim).
        """
        epsilon = torch.randn_like(var)
        z_reparam = mu + var*epsilon
        return z_reparam
        
    def forward(self, u, s, edge_index):
        """
        Forward pass through the VAE.

        Parameters
        ----------
        u : torch.Tensor
            Unspliced counts.
        s : torch.Tensor
            Spliced counts.
        edge_index : torch.Tensor
            Graph connectivity (unused in this implementation).

        Returns
        -------
        mu, log_var, reconstructed_u, reconstructed_s, z_reparam
        """
        mu, log_var = self.encoder(u, s, edge_index)
        z_reparam = self.reparameterization(mu, torch.exp(0.5 * log_var))
        reconstructed_u, reconstructed_s = self.decoder(z_reparam)
        return mu, log_var, reconstructed_u, reconstructed_s, z_reparam

def get_vae(adata, z_dim, lr=1e-2):
    """
    Utility function to initialize the VAE, optimizer, and loss function.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    z_dim : int
        Latent dimensionality.
    lr : float, optional (default: 1e-2)
        Learning rate for the Adam optimizer.

    Returns
    -------
    vae : VariationalAutoEncoder
        Instantiated VAE model.
    optimizer : torch.optim.Adam
        Adam optimizer for training the VAE.
    loss_fn : torch.nn.MSELoss
        Mean Squared Error loss (sum reduction).
    """
    loss_fn = nn.MSELoss(reduction="sum")
    vae = VariationalAutoEncoder(adata.n_vars, z_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr)
    
    return vae, optimizer, loss_fn

def train_vae(adata, vae, optimizer, loss_fn, global_nb_indices, patience, num_epochs=1000, batch=0.25, device="cpu", verbose=True):
    """
    Train a Variational Autoencoder (VAE) on spliced/unspliced single-cell data
    with neighborhood-aware graph edges.

    The function uses mini-batching, KL divergence regularization, and early stopping.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spliced and unspliced layers.
    vae : VariationalAutoEncoder
        VAE model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    loss_fn : callable
        Loss function (typically MSELoss).
    global_nb_indices : np.ndarray
        Precomputed neighbor indices for all cells (shape: [n_cells, n_neighbors]).
    patience : int
        Number of epochs with no improvement before early stopping is triggered.
    num_epochs : int, optional (default: 1000)
        Maximum number of training epochs.
    batch : float, optional (default: 0.25)
        Proportion of cells per batch (between 0 and 1).
    device : str, optional (default: "cpu")
        Device to run training on ("cpu" or "cuda").
    verbose : bool, optional (default: True)
        If False, suppresses training output.

    Returns
    -------
    best_vae : dict
        State dictionary of the best-performing VAE model (based on validation loss).

    Notes
    -----
    - Loss is composed of three terms:
        * Reconstruction loss for spliced counts
        * Weighted reconstruction loss for unspliced counts (0.3x)
        * KL divergence regularization
    - Uses early stopping to prevent overfitting.
    """
    org_stdout = sys.stdout # Save original stdout
    f = None
    
    if not verbose:
        f = open(os.devnull, "w")
        sys.stdout = f

    try:
        best_vae = None
        best_loss = -1
        imp_counter = 0
        
        batch_size = int(math.ceil(adata.n_obs*batch))
        
        with trange(num_epochs) as pbar:
            for epoch in pbar:
                tot_loss = 0
                iters = 0
                shuffled_indices = np.random.permutation(adata.n_obs)
    
                loss = 0
                
                for batch_idx in range(0, adata.n_obs, batch_size):
                    iters += 1
    
                    end_idx = min(batch_idx + batch_size, adata.n_obs)
    
                    # Get global indices for current batch
                    batch_indices = shuffled_indices[batch_idx:end_idx]
                    batch_data = adata[batch_indices, :]
    
                    s = torch.tensor(batch_data.layers['spliced'].A, dtype=torch.float32).to(device)
                    u = torch.tensor(batch_data.layers['unspliced'].A, dtype=torch.float32).to(device)
                    
                    # Create mapping from global index → local index
                    global_to_local = {g: l for l, g in enumerate(batch_indices)}
                    
                    # Subset and remap neighbors
                    nb_indices_global = global_nb_indices[batch_indices]
                    remapped = np.vectorize(lambda g: global_to_local[g] if g in global_to_local else -1)(nb_indices_global)
                    mask = remapped != -1
                    nb_indices_local = np.where(mask, remapped, 0)  # temporarily fill -1s with dummy index
                    
                    # Drop self-loops (first column is self)
                    xs = np.repeat(range(nb_indices_local.shape[0]), nb_indices_local.shape[1] - 1)
                    ys = nb_indices_local[:, 1:].flatten()
                    
                    edge_i = torch.LongTensor(np.vstack([xs, ys])).to(device)
    
                    mu, log_var, reconstructed_u, reconstructed_s, z = vae(u, s, edge_i)
    
                    # compute loss
                    recon_loss_u = loss_fn(reconstructed_u, u)
                    recon_loss_s = loss_fn(reconstructed_s, s)
                    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                    
                    loss += recon_loss_s + 0.3*recon_loss_u + kl_div
    
                # early stop test
                if loss < best_loss or best_loss == -1:
                    best_vae = vae.state_dict()
                    best_loss = loss
                    imp_counter = 0
                else:
                    imp_counter += 1
    
                if imp_counter > patience:
                    if verbose:
                        print("Early stopping triggered")
                    break
    
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tot_loss += loss
                avg_loss = tot_loss/iters
                pbar.set_postfix(loss=loss.item())
                
    finally:
        sys.stdout = org_stdout
        if f is not None:
            f.close()

    return best_vae

def get_embedding(adata, vae_model, device):
    """
    Compute latent embeddings for cells using a trained VAE model.

    This function:
    1. Extracts unspliced and spliced counts from the AnnData object.
    2. Constructs a neighbor graph (`edge_index`) based on precomputed neighbors.
    3. Passes the data through the VAE to obtain latent embeddings.
    4. Stores the embeddings in `adata.obsm["cell_embed"]`.
    5. Recomputes the neighbor graph using the new latent embeddings.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing spliced/unspliced layers and neighbor graph info.
    vae_model : VariationalAutoEncoder
        Trained VAE model used to generate latent embeddings.
    device : str
        Device for computation ("cpu" or "cuda").

    Returns
    -------
    None
        Updates `adata` in place with:
        - `adata.obsm["cell_embed"]`: latent embeddings (shape: [n_cells, z_dim])
        - Recomputed neighbors in `adata.uns["neighbors"]` using the latent space
    """
    s = torch.tensor(adata.layers['spliced'].A, dtype=torch.float32).to(device)
    u = torch.tensor(adata.layers['unspliced'].A, dtype=torch.float32).to(device)

    scv.pp.neighbors(adata, n_neighbors=30)
    nb_indices = adata.uns['neighbors']['indices']

    # Build edge index (drop self-loops: first neighbor is self)
    xs, ys = np.repeat(range(adata.n_obs), nb_indices.shape[1]-1), nb_indices[:, 1:].flatten()
    edge_i = torch.LongTensor(np.vstack([xs.reshape(1,-1), ys.reshape(1,-1)])).to(device)
    
    # Forward pass through VAE to get latent embeddings
    _,_,_,_,z = vae_model(u, s, edge_i)

    # Store embeddings in AnnData
    adata.obsm["cell_embed"] =  z.detach().numpy()

    # Recompute neighbors in the new embedding space
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="cell_embed")
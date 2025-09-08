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
import anndata


class EncoderGCN(nn.Module):
    
    def __init__(self, in_dim, z_dim):
        super(EncoderGCN, self).__init__()
        dim_diff = in_dim - z_dim
        hd1 = z_dim + 2*dim_diff//3
        hd2 = z_dim + dim_diff//3

        self.net_u = nn.Sequential(nn.Linear(in_dim, hd1), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd2, z_dim), nn.BatchNorm1d(z_dim), nn.LeakyReLU(), nn.Dropout(p=0.2))
        
        self.net_s = nn.Sequential(nn.Linear(in_dim, hd1), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(), nn.Dropout(p=0.2),
                                   nn.Linear(hd2, z_dim), nn.BatchNorm1d(z_dim), nn.LeakyReLU(), nn.Dropout(p=0.2))
        
        self.fc_mu_z = nn.Linear(z_dim*2, z_dim)
        self.fc_std_z = nn.Linear(z_dim*2, z_dim)
        self.spt = nn.Softplus()
        
        self.act = nn.LeakyReLU()
        
        self._init_weights()

    def _init_weights(self):
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
        h_u = self.net_u(u)
        h_s = self.net_s(s)
        h = torch.cat((h_s.squeeze(0), h_u.squeeze(0)), dim=1)
        mu, log_var = self.fc_mu_z(h), self.spt(self.fc_std_z(h))
        return mu, log_var

class Decoder(nn.Module):
    
    def __init__(self, in_dim, z_dim):
        super(Decoder, self).__init__()
        dim_diff = in_dim - z_dim
        hd1 = z_dim + dim_diff//3
        hd2 = z_dim + 2*dim_diff//3

        self.net_u = nn.Sequential(nn.Linear(z_dim, hd1), nn.LeakyReLU(),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(),
                                   nn.Linear(hd2, in_dim), nn.LeakyReLU())
        
        self.net_s = nn.Sequential(nn.Linear(z_dim, hd1), nn.LeakyReLU(),
                                   nn.Linear(hd1, hd2), nn.LeakyReLU(),
                                   nn.Linear(hd2, in_dim), nn.LeakyReLU())
    
    def forward(self, z_data):
        reconstructed_u = self.net_u(z_data)
        reconstructed_s = self.net_s(z_data)
        return reconstructed_u, reconstructed_s

class VariationalAutoEncoderGCN(nn.Module):
    
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.encoder = EncoderGCN(in_dim, z_dim)
        self.decoder = Decoder(in_dim, z_dim)
        
    def reparameterization(self, mu, var):
        epsilon = torch.randn_like(var)
        z_reparam = mu + var*epsilon
        return z_reparam
        
    def forward(self, u, s, edge_index):
        mu, log_var = self.encoder(u, s, edge_index)
        z_reparam = self.reparameterization(mu, torch.exp(0.5 * log_var))
        reconstructed_u, reconstructed_s = self.decoder(z_reparam)
        return mu, log_var, reconstructed_u, reconstructed_s, z_reparam

def get_vae(adata, z_dim):

    loss_fn = nn.MSELoss(reduction="sum")
    vae = VariationalAutoEncoderGCN(adata.n_vars, z_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-2)
    
    return vae, optimizer, loss_fn

def train_vae(adata, vae, optimizer, loss_fn, global_nb_indices, patience, num_epochs=1000, batch=0.25, device="cpu", verbose=True):

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

def get_embedding(adata, vae_model):
    s = torch.tensor(adata.layers['spliced'].A, dtype=torch.float32).to(device)
    u = torch.tensor(adata.layers['unspliced'].A, dtype=torch.float32).to(device)

    scv.pp.neighbors(adata, n_neighbors=30)
    conn = adata.obsp['connectivities']
    nb_indices = adata.uns['neighbors']['indices']
    xs, ys = np.repeat(range(adata.n_obs), nb_indices.shape[1]-1), nb_indices[:, 1:].flatten()
    edge_i = torch.LongTensor(np.vstack([xs.reshape(1,-1), ys.reshape(1,-1)])).to(device)
    _,_,_,_,z = vae_model(u, s, edge_i)

    adata.obsm["cell_embed"] =  z.detach().numpy()
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="cell_embed")
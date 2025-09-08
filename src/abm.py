import mesa
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scanpy as sc
import scvelo as scv
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import sys
sys.path.insert(1,'../SIRV/')
from main import SIRV
from mesa.visualization.UserParam import *
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
import anndata
import math
import mplscience


# Agent Based Model
class CellAgent(mesa.Agent):
    def __init__(self, unique_id, model, cell_x, cell_y, exp, velo, velo_u):
        super().__init__(unique_id, model)
        self.pos_x = cell_x
        self.pos_y = cell_y
        self.expression = exp
        self.neighbors = None
        self.nbs_dists = None
        self.density = None
        self.similarity_w = None
        self.density_w = None
        self.spatial_w = None
        self.sig = None
        self.lam = None
        self.velocity = velo.copy()
        self.velocity_u = velo_u.copy()
        # temp storage for synchronous updates
        self.next_velocity = self.velocity.copy()
        self.next_velocity_u = self.velocity_u.copy()

    def compute_next_velo(self):
        self_inf = self.density_w * self.velocity
        self_inf_u = self.density_w * self.velocity_u
    
        if not self.neighbors:
            nb_inf = np.zeros_like(self.velocity)
            nb_inf_u = np.zeros_like(self.velocity_u)
        else:
            nb_velos = np.vstack([nb.velocity for nb in self.neighbors])
            nb_velos_u = np.vstack([nb.velocity_u for nb in self.neighbors])
    
            # ensure similarity_w & spatial_w exist
            sim_w = np.array(self.similarity_w) if self.similarity_w is not None else np.zeros(len(self.neighbors))
            spatial_w = np.array(self.spatial_w) if self.spatial_w is not None else np.zeros(len(self.neighbors))
    
            # If lengths mismatch, pad the shorter with zeros
            n = len(self.neighbors)
            if sim_w.shape[0] != n:
                sim_w = np.resize(sim_w, n)
            if spatial_w.shape[0] != n:
                spatial_w = np.resize(spatial_w, n)
    
            total_w = sim_w + spatial_w
            s = total_w.sum()
            if s > 0:
                total_w = total_w / s
                nb_inf = np.dot(total_w, nb_velos)
                nb_inf_u = np.dot(total_w, nb_velos_u)
            else:
                # fallback: uniform neighbor averaging
                nb_inf = nb_velos.mean(axis=0)
                nb_inf_u = nb_velos_u.mean(axis=0)
    
        gamma = 0.2
        new_velo = (1 - gamma) * self.velocity + gamma * (self_inf + nb_inf)
        new_velo_u = (1 - gamma) * self.velocity_u + gamma * (self_inf_u + nb_inf_u)
    
        max_norm = self.model.max_gene_norm
        cur_norm = np.linalg.norm(new_velo)
        if cur_norm < 1e-8:
            new_velo = np.zeros_like(new_velo)
        elif cur_norm > max_norm:
            new_velo = new_velo / cur_norm * max_norm
    
        self.next_velocity = new_velo
        self.next_velocity_u = new_velo_u

    def step(self):
        # compute only; assignment is done after all agents compute
        self.compute_next_velo()


class CellModel(mesa.Model):
    
    def __init__(self, adata, steps, freedom=2, nbr_radius=40, sig_ratio=0.7, max_gene_norm=10.0):
        self.num_agents = adata.n_obs
        self.adata = adata
        self.num_genes = adata.n_vars
        self.deg_fred = freedom
        self.nbr_radius = nbr_radius
        self.sig_ratio = sig_ratio
        self.max_gene_norm = max_gene_norm
        self.min_density = None
        self.max_density = None
        self.tau = 0.2
        self.curr_step = 0
        self.num_steps = steps

        # scheduler + grid
        self.schedule = mesa.time.BaseScheduler(self)

        # create grid sized to integer max coordinate + 1
        max_x = int(np.max(adata.obs['x_loc'])) + 1
        max_y = int(np.max(adata.obs['y_loc'])) + 1
        self.grid = mesa.space.MultiGrid(width=max_x, height=max_y, torus=False)

        # Create agents first
        for i in range(self.num_agents):
            ad_row = self.adata[self.adata.obs_names[i]]
            exp = np.squeeze(ad_row.layers['Ms'])
            x_loc = int(ad_row.obs['x_loc'][0])
            y_loc = int(ad_row.obs['y_loc'][0])
            velo = np.squeeze(ad_row.layers['velocity'][0])
            velo_u = np.squeeze(ad_row.layers['velocity_u'][0])

            ag = CellAgent(i, self, x_loc, y_loc, exp, velo, velo_u)
            self.schedule.add(ag)
            self.grid.place_agent(ag, (x_loc, y_loc))

        # Build a positions array for all agents (for nearest-neighbors lookups)
        coords = np.array([[ag.pos_x, ag.pos_y] for ag in self.schedule.agents])

        # Use sklearn NearestNeighbors with radius (euclidean)
        nbrs = NearestNeighbors(radius=self.nbr_radius, metric='euclidean', n_jobs=-1)
        nbrs.fit(coords)
        indices_list, dists_list = nbrs.radius_neighbors(coords, return_distance=True)

        # attach neighbors + inverse-distance weights
        for i, ag in enumerate(self.schedule.agents):
            idxs = indices_list[i]
            # remove self index
            mask = idxs != i
            neighbor_idxs = idxs[mask]
            ag.neighbors = [self.schedule.agents[j] for j in neighbor_idxs]
            dists = dists_list[i][mask]
            # avoid divisions by zero; cap very small distances
            dists = np.maximum(dists, 1e-6)
            ag.nbs_dists = (1.0 / (dists ** self.deg_fred)).tolist()

        self.get_neighbors()
        self.get_min_max_density()
        self.get_nbs_dist()
        self.get_nbs_sim()
        self.w_density()
        self.calc_sig_lam()
        self.w_similarity()
        self.w_spatial()
        
    def distance(self, p1, p2):
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_neighbors(self):
        """
        Use neighbors already attached to agents (from NearestNeighbors in __init__).
        Set agent.density accordingly and print average.
        """
        num_neighbors = 0
        for agent in self.schedule.agents:
            if agent.neighbors is None:
                agent.neighbors = []
            agent.density = len(agent.neighbors)
            num_neighbors += len(agent.neighbors)
    
        avg_neighbors = num_neighbors / max(1, len(self.schedule.agents))
        print('Avg neighbors: {:.3f}'.format(avg_neighbors))
            
    def get_min_max_density(self):
        min_density = None
        max_density = None
        
        for agent in self.schedule.agents:
            den = agent.density
            
            if min_density is None:
                min_density = den
            elif den < min_density:
                min_density = den
            
            if max_density is None:
                max_density = den
            elif den > max_density:
                max_density = den
        
        self.min_density = min_density
        self.max_density = max_density
            
    def get_nbs_dist(self):
        for agent in self.schedule.agents:
            nb_inv_dist = []
            x_ag = agent.pos_x
            y_ag = agent.pos_y
            ag_pos = (x_ag, y_ag)
            for nb in agent.neighbors:
                x_nb = nb.pos_x
                y_nb = nb.pos_y
                nb_pos = (x_nb, y_nb)
                
                dist = self.distance(ag_pos, nb_pos)
                
                inv_dist = 1/(dist**self.deg_fred)
                
                nb_inv_dist.append(inv_dist)
            
            agent.nbs_dists = nb_inv_dist
    
    def get_nbs_sim(self):
        eps = 1e-12
        for agent in self.schedule.agents:
            nb_sim = []
            exp1 = np.squeeze(agent.expression)
            norm_a = norm(exp1)
            if norm_a < eps:
                norm_a = eps
            for nb in agent.neighbors:
                exp2 = np.squeeze(nb.expression)
                norm_b = norm(exp2)
                if norm_b < eps:
                    norm_b = eps
                dot_product = np.dot(exp1, exp2)
                similarity = dot_product / (norm_a * norm_b)
                nb_sim.append(similarity)
            agent.nbs_sim = nb_sim
            
    def w_similarity(self):
        for agent in self.schedule.agents:
            sum_sim = sum(agent.nbs_sim) if hasattr(agent, 'nbs_sim') else 0.0
            if sum_sim <= 0:
                # fallback: uniform weights if any neighbors exist
                n = len(agent.neighbors)
                if n > 0:
                    nb_sim_w = [agent.sig / n] * n
                else:
                    nb_sim_w = []
            else:
                nb_sim_w = [agent.sig * (s / sum_sim) for s in agent.nbs_sim]
            agent.similarity_w = nb_sim_w
    
    def w_density(self):
        denom = (self.max_density - self.min_density)
        if denom == 0:
            # All agents same density -> uniform density_w
            for agent in self.schedule.agents:
                agent.density_w = 1.0  # or some default like 1.0
            return
    
        for agent in self.schedule.agents:
            agent.density_w = 1 - ((agent.density - self.min_density) / denom) * self.tau
    
    def w_spatial(self):
        for agent in self.schedule.agents:
            sum_dist = sum(agent.nbs_dists) if hasattr(agent, 'nbs_dists') else 0.0
            nb_dist_w = []
            if sum_dist <= 0:
                nb_dist_w = [0.0] * len(agent.neighbors)
            else:
                for i in range(len(agent.neighbors)):
                    dist_w = agent.lam * (agent.nbs_dists[i] / sum_dist)
                    nb_dist_w.append(dist_w)
            agent.spatial_w = nb_dist_w
            
    def calc_sig_lam(self):
        for agent in self.schedule.agents:
            
            if self.sig_ratio > 1.0:
                raise ValueError("The sigma ratio must be smaller than 1.0")
                
            lam_ratio = 1 - self.sig_ratio
            agent.sig = self.sig_ratio * (1 - agent.density_w)
            agent.lam = lam_ratio * (1 - agent.density_w)
            
    def step(self):
        for _ in range(self.num_steps):
            # First pass: compute next velocities for all agents
            for agent in self.schedule.agents:
                agent.compute_next_velo()
            # Second pass: commit updates (synchronous)
            for agent in self.schedule.agents:
                agent.velocity = agent.next_velocity
                agent.velocity_u = agent.next_velocity_u
            self.curr_step += 1

        velos = []
        velos_u = []
        for agent in self.schedule.agents:
            velos.append(agent.velocity)
            velos_u.append(agent.velocity_u)
        self.adata.layers['velocity'] = np.vstack(velos)
        self.adata.layers['velocity_u'] = np.vstack(velos_u)
import mesa
import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from mesa.visualization.UserParam import *
import math


# Agent Based Model
class CellAgent(mesa.Agent):
    """
    Represents a single cell agent in a spatial RNA velocity simulation.

    Each cell stores its position, gene expression, velocities (spliced/unspliced),
    neighbors, and interaction weights for computing neighbor-influenced velocity updates.

    Attributes
    ----------
    pos_x, pos_y : float
        Spatial coordinates of the cell.
    expression : np.ndarray
        Gene expression vector for this cell.
    neighbors : list of CellAgent
        Neighboring cell agents influencing this cell.
    nbs_dists : np.ndarray
        Distances to neighboring cells.
    density : float
        Local cell density.
    similarity_w : np.ndarray
        Similarity weights for neighbors based on gene expression.
    spatial_w : np.ndarray
        Spatial weights for neighbors.
    density_w : float
        Self-weight based on local density.
    velocity : np.ndarray
        Current predicted spliced RNA velocity.
    velocity_u : np.ndarray
        Current predicted unspliced RNA velocity.
    next_velocity : np.ndarray
        Temporary storage for updated spliced velocity (synchronous update).
    next_velocity_u : np.ndarray
        Temporary storage for updated unspliced velocity (synchronous update).

    Parameters
    ----------
    unique_id : int
        Unique identifier for the agent.
    model : mesa.Model
        Mesa model this agent belongs to.
    cell_x, cell_y : float
        Spatial coordinates.
    exp : np.ndarray
        Gene expression vector.
    velo : np.ndarray
        Initial spliced RNA velocity vector.
    velo_u : np.ndarray
        Initial unspliced RNA velocity vector.
    """

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
        """
        Compute the next velocity for this cell by combining self-influence
        (weighted by density) and neighbor influence (weighted by similarity
        and spatial proximity).

        Steps:
        1. Compute self-influence using density weight.
        2. Aggregate neighbor velocities weighted by similarity + spatial weights.
        3. Combine self and neighbor influence with a gamma blending factor.
        4. Normalize resulting velocity to respect maximum gene norm.
        5. Store updated velocities in `next_velocity` and `next_velocity_u`.
        """
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
    
        gamma = 0.2 # blending factor for self vs neighbor influence
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
        """
        Single agent step in the Mesa model scheduler.

        This function computes the next velocity but does not yet assign it
        to `velocity` or `velocity_u`. Synchronous updates are applied after
        all agents have computed their next velocities.
        """
        self.compute_next_velo()


class CellModel(mesa.Model):
    """
    An agent-based spatial RNA velocity model.

    Each agent represents a cell with a position, gene expression,
    spliced/unspliced RNA velocities, and neighbor relationships.
    The model computes neighbor-influenced velocity updates across steps.

    Attributes
    ----------
    num_agents : int
        Number of cells (agents) in the dataset.
    adata : AnnData
        Single-cell RNA-seq annotated dataset with spliced/unspliced layers.
    num_genes : int
        Number of genes in the dataset.
    deg_fred : int
        Exponent used to compute inverse-distance weights for neighbors.
    nbr_radius : float
        Radius for considering neighbors in spatial grid.
    sig_ratio : float
        Ratio controlling similarity weight contribution.
    max_gene_norm : float
        Maximum allowed L2 norm for velocities.
    min_density : float
        Minimum observed cell density among all agents.
    max_density : float
        Maximum observed cell density among all agents.
    tau : float
        Scaling factor for density-based weighting.
    curr_step : int
        Current simulation step.
    num_steps : int
        Total number of steps per `step()` call.
    schedule : mesa.time.BaseScheduler
        Scheduler managing agent updates.
    grid : mesa.space.MultiGrid
        Spatial grid to place agents and compute neighborhoods.

    Methods
    -------
    distance(p1, p2)
        Euclidean distance between two 2D points.
    get_neighbors()
        Assign neighbors and density to agents based on spatial proximity.
    get_min_max_density()
        Compute min/max density across all agents.
    get_nbs_dist()
        Compute inverse-distance weights for neighbors.
    get_nbs_sim()
        Compute cosine similarity between agent and neighbor expressions.
    w_density()
        Compute density weight for each agent.
    calc_sig_lam()
        Compute similarity (sig) and spatial (lam) weights based on density.
    w_similarity()
        Compute similarity-based neighbor weights.
    w_spatial()
        Compute spatial distance-based neighbor weights.
    step()
        Perform synchronous velocity updates for all agents and store results in `adata`.
    """
    
    def __init__(self, adata, steps, freedom=2, nbr_radius=40, sig_ratio=0.7, max_gene_norm=10.0):
        """
        Initialize CellModel.

        Parameters
        ----------
        adata : AnnData
            Single-cell dataset with spatial coordinates and velocity layers.
        steps : int
            Number of steps to simulate per `step()` call.
        freedom : int, default=2
            Exponent for inverse-distance neighbor weighting.
        nbr_radius : float, default=40
            Radius to define neighbors in spatial space.
        sig_ratio : float, default=0.7
            Ratio controlling similarity vs spatial weighting.
        max_gene_norm : float, default=10.0
            Maximum allowed L2 norm for velocities to prevent explosion.
        """
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
            neighbor_idxs = neighbor_idxs.astype(int)
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
        """
        Compute the Euclidean distance between two 2D points.

        Parameters
        ----------
        p1 : tuple of float
            Coordinates of the first point (x1, y1).
        p2 : tuple of float
            Coordinates of the second point (x2, y2).

        Returns
        -------
        float
            Euclidean distance between p1 and p2.
        """
        (x1, y1), (x2, y2) = p1, p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def get_neighbors(self):
        """
        Assign neighbors and compute local density for each agent.

        For each agent in the model, this method:
        - Ensures that `agent.neighbors` is a list (empty if no neighbors).
        - Sets `agent.density` to the number of neighbors.
        - Computes and prints the average number of neighbors across all agents.

        Neighbors are expected to be precomputed and attached to each agent.
        """
        num_neighbors = 0
        for agent in self.schedule.agents:
            if agent.neighbors is None:
                agent.neighbors = []
            agent.density = len(agent.neighbors) # local density
            num_neighbors += len(agent.neighbors)
    
        avg_neighbors = num_neighbors / max(1, len(self.schedule.agents))
        print('Avg neighbors: {:.3f}'.format(avg_neighbors))
            
    def get_min_max_density(self):
        """
        Compute the minimum and maximum local densities across all agents.

        This method iterates through all agents in the model and:
        - Reads each agent's `density` attribute (number of neighbors).
        - Tracks the smallest (`min_density`) and largest (`max_density`) values.
        - Stores the results in `self.min_density` and `self.max_density`.

        These values can be used later for normalizing density-based weights
        in agent interactions.
        """
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
        """
        Compute inverse-distance weights for each agent's neighbors.

        For each agent:
        - Calculates the Euclidean distance to all neighbors.
        - Adds a small epsilon (1e-6) to prevent division by zero.
        - Computes inverse-distance weights raised to the power of `deg_fred`.
        - Stores the list of inverse-distance weights in `agent.nbs_dists`.

        These weights are later used for spatial influence in velocity updates.
        """
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
                epsilon = 1e-6
                inv_dist = 1 / ((dist + epsilon) ** self.deg_fred)
                
                nb_inv_dist.append(inv_dist)
            
            agent.nbs_dists = nb_inv_dist
    
    def get_nbs_sim(self):
        """
        Compute cosine similarity between each agent and its neighbors in a gene expression perspective.

        For each agent:
        - Flatten the agent's expression vector.
        - Normalize the vector (avoid division by very small values using eps=1e-12).
        - For each neighbor:
            - Flatten and normalize the neighbor's expression vector.
            - Compute the cosine similarity: dot(exp_agent, exp_neighbor) / (||exp_agent|| * ||exp_neighbor||)
        - Store the list of similarities in `agent.nbs_sim`.

        These similarities are later used as weights for velocity updates and neighbor influence.
        """
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
        """
        Compute similarity-based weights for each agent's neighbors.

        For each agent:
        - Sum the cosine similarities stored in `agent.nbs_sim`.
        - If the sum is positive:
            - Normalize each neighbor's similarity by the total sum and multiply by `agent.sig`.
        - If the sum is zero or no neighbors exist:
            - Assign uniform weights (fallback) to neighbors based on `agent.sig`.
        - Store the resulting list of weights in `agent.similarity_w`.

        These weights determine how much influence each neighbor's velocity has during updates.
        """
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
        """
        Compute density-based weight for each agent.

        For each agent:
        - Calculate `density_w` as a scaled inverse of its local density relative to the
            minimum and maximum agent densities.
        - Formula: density_w = 1 - ((agent.density - min_density) / (max_density - min_density)) * tau
        - If all agents have the same density (denominator is zero), assign a uniform weight of 1.0.

        These weights reduce the influence of agents in dense regions during velocity updates.
        """
        denom = (self.max_density - self.min_density)
        if denom == 0:
            # All agents same density -> uniform density_w
            for agent in self.schedule.agents:
                agent.density_w = 1.0  # or some default like 1.0
            return
    
        for agent in self.schedule.agents:
            agent.density_w = 1 - ((agent.density - self.min_density) / denom) * self.tau
    
    def w_spatial(self):
        """
        Compute spatial-based weight for each agent's neighbors.

        For each agent:
        - Calculate `spatial_w` for neighbors based on inverse-distance weights stored in `agent.nbs_dists`.
        - Normalize the inverse-distance weights by the sum of all neighbor distances.
        - Scale by the agent's lambda factor (`agent.lam`).
        - If there are no neighbors or the sum of distances is zero, assign 0.0 to all neighbor weights.

        These weights capture spatial influence when updating velocities, giving closer neighbors more impact.
        """
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
        """
        Compute scaling factors for similarity and spatial weights for each agent.

        For each agent:
        - Ensure `sig_ratio` is ≤ 1.0.
        - Compute `agent.sig` as the similarity scaling factor: sig_ratio * (1 - agent.density_w)
        - Compute `agent.lam` as the spatial scaling factor: (1 - sig_ratio) * (1 - agent.density_w)

        These factors are later used to weight neighbor contributions in velocity updates, modulated by local density.
        """
        for agent in self.schedule.agents:
            
            if self.sig_ratio > 1.0:
                raise ValueError("The sigma ratio must be smaller than 1.0")
                
            lam_ratio = 1 - self.sig_ratio
            agent.sig = self.sig_ratio * (1 - agent.density_w)
            agent.lam = lam_ratio * (1 - agent.density_w)
            
    def step(self):
        """
        Advance the model by one simulation step (or `num_steps` sub-steps).

        Procedure:
        1. For each sub-step:
            a. Compute the next velocities (`velocity` and `velocity_u`) for all agents using `compute_next_velo`.
            b. Synchronously update each agent’s current velocity to the computed next velocity.
        2. After all sub-steps, collect the final velocities of all agents.
        3. Store the velocity matrices in `adata.layers['velocity']` and `adata.layers['velocity_u']`.

        This ensures that all agents update synchronously and that the model state is saved in the AnnData object.
        """
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
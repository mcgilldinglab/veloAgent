import torch
import logging
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm, trange


logging.basicConfig(level=logging.INFO)

def load_protein_paths(species, base):
    """
    Return file paths for STRING protein data for a given species.

    Parameters
    ----------
    species : str
        Species name ("mouse", "chicken", or "human").
    base : str
        Base directory where protein data files are stored.

    Returns
    -------
    list of str
        Paths to the protein info, aliases, and links files for the given species.

    Raises
    ------
    ValueError
        If the species is not supported.
    """
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
    """
    Create a gene–gene connectivity matrix based on protein–protein interactions (PPIs).

    This function maps genes from an AnnData object to STRING protein IDs using both
    protein name and alias files, prunes the global PPI network to relevant proteins,
    and constructs a connectivity matrix that can be used as a prior for models.

    Parameters
    ----------
    data : AnnData
        Annotated data matrix containing gene expression and metadata.
    num_genes : int
        Number of genes to include in the connectivity matrix (matrix will be square).
    prot_names : str
        Path to STRING protein names file (e.g., `*.protein.info.v12.0.txt`).
    prot_alias : str
        Path to STRING protein aliases file (e.g., `*.protein.aliases.v12.0.txt`).
    gene_conn : str
        Path to STRING protein–protein interaction links file (e.g., `*.protein.links.v12.0.txt`).
    varname : str
        Column name in `data.var` used as gene identifiers (typically `"index"` or `"gene_name"`).

    Returns
    -------
    numpy.ndarray
        Dense binary (0/1) connectivity matrix of shape `(num_genes, num_genes)`,
        where entry (i, j) = 1 if gene i and gene j are connected in the PPI network.

    Notes
    -----
    - Genes with no known PPI connections are artificially connected to all others to
      avoid isolated nodes (ensuring the matrix has no empty rows/columns).
    - Internally, a helper function `get_alias` is used to merge gene aliases with
      protein IDs for mapping.
    """

    # create protein alias
    def get_alias(data, prot_names, prot_alias):
        """
        Map genes in the AnnData object to STRING protein IDs using protein names and aliases.

        This function loads the STRING protein names and alias files, filters them to genes 
        present in the AnnData object, removes duplicate aliases, and merges synonyms so that 
        each gene is associated with a valid protein ID.

        Parameters
        ----------
        data : AnnData
            Annotated data matrix containing gene information (`data.var_names`).
        prot_names : str
            Path to STRING protein names file (e.g., `*.protein.info.v12.0.txt`).
        prot_alias : str
            Path to STRING protein aliases file (e.g., `*.protein.aliases.v12.0.txt`).

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns:
            - `alias` : gene names present in the dataset
            - `protein_id` : corresponding STRING protein IDs (including synonyms)

        Notes
        -----
        - Duplicated aliases are dropped.
        - Missing aliases are preserved from the alias file 
        and merged into the final mapping.
        """
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
    A custom autograd function that performs a masked linear transformation.

    This function behaves like a standard linear layer (`y = xW^T + b`), but 
    allows applying a binary mask to the weight matrix during both the forward 
    and backward passes. Entries in the weight matrix corresponding to `mask == 0` 
    are forced to zero and remain inactive in gradient updates.

    Forward pass:
        output = input.mm((weight * mask).T) + bias

    Backward pass:
        - Gradients w.r.t. weights are also masked, i.e. positions with `mask == 0` 
          receive zero gradient.
        - Bias gradient is the sum of gradients along the batch dimension.
        - Gradients are only computed for inputs that require them 
          (`ctx.needs_input_grad` is checked for efficiency).

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, in_features).
    weight : torch.Tensor
        Weight tensor of shape (out_features, in_features).
    bias : torch.Tensor, optional
        Bias tensor of shape (out_features,).
    mask : torch.Tensor, optional
        Binary mask tensor of the same shape as `weight`. Elements with value 0 
        deactivate the corresponding weight and prevent gradient flow.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, out_features).
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
    """
    GeneNet: A neural network model for gene-wise parameter prediction with 
    connectivity priors.

    This network is designed to learn gene-level embeddings or parameters 
    (e.g., transcriptional rates α, β, γ) from input features while enforcing 
    biological priors through a masked linear transformation based on 
    protein–protein interaction (PPI) or gene connectivity matrices.

    Architecture
    ------------
    1. Linear(in_dim → gene_dim//2) + BatchNorm + LeakyReLU + Dropout
    2. Linear(gene_dim//2 → gene_dim) + BatchNorm + LeakyReLU + Dropout
    3. CustomizedLinear(gene_dim → gene_dim, mask=conn) + BatchNorm + LeakyReLU + Dropout
       - Applies connectivity constraints so that only biologically relevant 
         gene–gene interactions contribute.
    4. Linear(gene_dim → 2*gene_dim) + BatchNorm + LeakyReLU + Dropout
    5. Linear(2*gene_dim → 3*gene_dim) + Sigmoid
       - Outputs constrained to [0, 1], suitable for modeling bounded parameters.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input features (e.g., latent cell embeddings).
    gene_dim : int
        Number of genes (output dimensionality per gene).
    conn : numpy.ndarray or torch.Tensor, optional
        Gene connectivity matrix (gene_dim × gene_dim). Used to mask weights 
        in the `CustomizedLinear` layer, enforcing structured sparsity 
        corresponding to known biological interactions.

    Forward Pass
    ------------
    x : torch.Tensor
        Input tensor of shape (batch_size, in_dim).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, 3*gene_dim), with values in [0, 1].
        Typically interpreted as gene-specific parameter estimates.

    Notes
    -----
    - Dropout rate is fixed at 0.2 across hidden layers.
    - Sigmoid activation ensures biologically meaningful bounded outputs.
    - The `CustomizedLinear` layer is key for incorporating biological priors.
    """

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
    """
    GeneNetAblation: Ablation version of GeneNet without connectivity priors.

    Architecture
    ------------
    1. Linear(in_dim → gene_dim//2) + BatchNorm + LeakyReLU + Dropout
    2. Linear(gene_dim//2 → gene_dim) + BatchNorm + LeakyReLU + Dropout
    3. Linear(gene_dim → gene_dim) + BatchNorm + LeakyReLU + Dropout
       - No masking; all gene connections are learned freely.
    4. Linear(gene_dim → 2*gene_dim) + BatchNorm + LeakyReLU + Dropout
    5. Linear(2*gene_dim → 3*gene_dim) + Sigmoid
       - Outputs constrained to [0, 1].
    """

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
    """
    Compute RNA velocity predictions for spliced and unspliced transcripts.

    This function uses estimated transcriptional rates (alpha, beta, gamma) to
    predict the change in spliced and unspliced RNA over a time step `dt`.

    Parameters
    ----------
    spliced : np.ndarray or torch.Tensor
        Current spliced RNA counts, shape (num_cells, num_genes).
    unspliced : np.ndarray or torch.Tensor
        Current unspliced RNA counts, shape (num_cells, num_genes).
    rates : np.ndarray or torch.Tensor
        Estimated RNA kinetic rates of shape (num_cells, 3*num_genes):
        - rates[:, 0:num_genes] → alpha (transcription rate)
        - rates[:, num_genes:2*num_genes] → beta (splicing rate)
        - rates[:, 2*num_genes:3*num_genes] → gamma (degradation rate)
    umax : float or array-like
        Scaling factor for unspliced RNA rates (alpha).
    smax : float or array-like
        Scaling factor for spliced RNA rates (beta and gamma).
    dt : float
        Time step for velocity integration.

    Returns
    -------
    pred_spliced : same type as input
        Predicted spliced RNA after time step dt.
    pred_unspliced : same type as input
        Predicted unspliced RNA after time step dt.
    alpha : array
        Scaled transcription rate per gene.
    beta : array
        Scaled splicing rate per gene.
    gamma : array
        Scaled degradation rate per gene.
    vel : array
        Spliced RNA velocity (change over dt).
    vel_u : array
        Unspliced RNA velocity (change over dt).
    """

    num_genes = len(spliced[0])

    alpha = rates[:,0:num_genes]
    beta = rates[:,num_genes:2*num_genes]
    gamma = rates[:,2*num_genes:3*num_genes]

    # Scale rates according to maximum observed counts
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
    Compute cosine similarity between predicted RNA velocity vectors and neighbors.

    This function compares the predicted velocity (change from current to predicted
    spliced/unspliced RNA) of each cell with the vectors connecting the cell to 
    its neighbors, returning a measure of alignment.

    Parameters
    ----------
    unspliced : torch.Tensor
        Current unspliced RNA counts, shape (num_cells, num_genes).
    spliced : torch.Tensor
        Current spliced RNA counts, shape (num_cells, num_genes).
    pred_unspliced : torch.Tensor
        Predicted unspliced RNA counts, same shape as `unspliced`.
    pred_spliced : torch.Tensor
        Predicted spliced RNA counts, same shape as `spliced`.
    indices : torch.Tensor
        Neighbor indices array, typically from a k-nearest neighbors graph.
        Shape: (num_cells, k_neighbors). `indices[i,j]` is the j-th neighbor of cell i.

    Returns
    -------
    1 - cosine_max : torch.Tensor
        Tensor of shape (num_cells,) containing 1 minus the maximum cosine similarity
        between the predicted velocity vector of each cell and the vectors to its neighbors.
        This can be interpreted as a distance or misalignment measure: smaller values 
        indicate better alignment of predicted velocity with neighbor directions.
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
    Compute the sum of cosine similarities between predicted RNA velocity vectors 
    and the velocities of neighboring cells.

    This function compares each cell's predicted velocity (from current to predicted
    spliced/unspliced RNA) with the velocity vectors of its neighbors, returning the 
    total alignment as a sum of cosine similarities.

    Parameters
    ----------
    unspliced : torch.Tensor
        Current unspliced RNA counts, shape (num_cells, num_genes).
    spliced : torch.Tensor
        Current spliced RNA counts, shape (num_cells, num_genes).
    pred_unspliced : torch.Tensor
        Predicted unspliced RNA counts, same shape as `unspliced`.
    pred_spliced : torch.Tensor
        Predicted spliced RNA counts, same shape as `spliced`.
    indices : torch.Tensor
        Neighbor indices array (typically from k-nearest neighbors graph).
        Shape: (num_cells, k_neighbors). `indices[i,j]` is the j-th neighbor of cell i.
    vel : torch.Tensor
        Predicted spliced RNA velocities, shape (num_cells, num_genes).
    vel_u : torch.Tensor
        Predicted unspliced RNA velocities, shape (num_cells, num_genes).

    Returns
    -------
    cosine_sum : torch.Tensor
        Scalar tensor representing the sum of cosine similarities between each 
        cell's predicted velocity and the velocity vectors of its neighbors.
    """

    #uv, sv = pred_unspliced-unspliced, pred_spliced-spliced # Velocity from (unsplice, splice) to (unsplice_predict, splice_predict)
    uv  = F.normalize(pred_unspliced-unspliced, dim=-1)
    sv  = F.normalize(pred_spliced-spliced, dim=-1)
    #unv, snv = vel_u[indices.T[1:]], vel[indices.T[1:]] # Velocity from (unsplice, splice) to its neighbors
    unv = F.normalize(vel_u[indices.T[1:]], dim=-1)
    snv = F.normalize(vel[indices.T[1:]], dim=-1)

    eps = 1e-12
    den = torch.sqrt(unv**2 + snv**2 + eps) * torch.sqrt(uv**2 + sv**2 + eps)
    den[den==0] = -1
    cosine = torch.where(den!=-1, (unv*uv + snv*sv) / den, torch.tensor(1.)) # cosine: column -> individual cell (cellI); row -> nearby cells of cell id ; value -> cosine between col and row cells
    cosine_sum = cosine.sum()
    return cosine_sum


def adj_velocity(data, velocity, indices):
    """
    Compute adjacency-adjusted velocities by averaging over neighbors.

    This function smooths RNA velocity estimates by replacing each cell's 
    velocity with the mean velocity of its neighbors. It helps reduce noise 
    and improves velocity consistency across similar cells.

    Parameters
    ----------
    data : AnnData or object
        Single-cell data object. Only `n_obs` (number of cells) is used.
    velocity : torch.Tensor
        Original velocity tensor of shape (num_cells, num_genes), e.g., spliced or unspliced velocities.
    indices : array-like or torch.Tensor
        Neighbor indices array (typically from a k-nearest neighbors graph). 
        `indices[j]` contains the indices of neighbors for cell j.

    Returns
    -------
    adj_vel : torch.Tensor
        Adjacency-adjusted velocity tensor of the same shape as `velocity`,
        where each cell's velocity is replaced by the mean of its neighbors' velocities.
    """
    
    adj_vel = velocity

    for j in range(data.n_obs):
        vel_n = velocity[indices[j][:]]
        vel_n = 0.7 * velocity[j] + 0.3 * vel_n.mean(dim=0)
        adj_vel[j] = vel_n

    return adj_vel


def train_gg(num_epochs, data, embed_basis, genenet, device, optimizer, patience=10, num_nbrs=30, dt=0.3, batch=0.25):
    """
    Train a GeneNet model to predict RNA kinetic rates.

    This function performs mini-batch training of GeneNet to predict transcriptional kinetics (alpha, beta, gamma).
    The loss is based on cosine similarity between predicted RNA velocities and neighborhood structure.

    Parameters
    ----------
    num_epochs : int
        Maximum number of training epochs.
    data : AnnData
        Single-cell RNA dataset containing layers 'Ms' (spliced) and 'Mu' (unspliced),
        and embeddings in `data.obsm[embed_basis]`.
    embed_basis : str
        Key in `data.obsm` containing cell embeddings for training.
    genenet : nn.Module
        PyTorch GeneNet model that predicts kinetic rates.
    device : str or torch.device
        Device for training ('cpu' or 'cuda').
    optimizer : torch.optim.Optimizer
        Optimizer used for training the GeneNet model.
    patience : int, optional (default=10)
        Number of epochs to wait for improvement before early stopping.
    num_nbrs : int, optional (default=30)
        Number of nearest neighbors for computing cosine similarity loss.
    dt : float, optional (default=0.3)
        Time step used in velocity prediction.
    batch : float, optional (default=0.25)
        Fraction of cells to sample per mini-batch.

    Returns
    -------
    None
        The function updates `data.layers` with the predicted velocities and kinetic rates:
        - "velocity_u": unspliced RNA velocity
        - "velocity": spliced RNA velocity
        - "alpha", "beta", "gamma": predicted kinetic rates

    Notes
    -----
    - A random subset of cells is sampled each epoch according to `batch`.
    - Nearest neighbors are computed in the embedding space (`embed_basis`) to define
      the cosine similarity loss.
    - Early stopping is implemented based on `patience`.
    - After training, predictions are computed for all cells in `data`.
    - The function ensures `umax` and `smax` are at least 1 to prevent division by zero.
    """

    umax = torch.max(torch.tensor(data.layers['Mu']), dim=0)[0]
    smax = torch.max(torch.tensor(data.layers['Ms']), dim=0)[0]
    umax[umax == 0] = 1
    smax[smax == 0] = 1

    best_net = None
    best_loss = -1
    imp_counter = 0
    
    with trange(num_epochs) as pbar:
        for epoch in pbar:
            
            # Subsample cells for mini-batch training
            subsample_idx = np.random.choice(data.n_obs, size=int(data.n_obs * batch), replace=False)
            subsample = data[subsample_idx, :]

            latent = torch.tensor(subsample.obsm[embed_basis], dtype=torch.float32).to(device)
            spliced = torch.tensor(subsample.layers['Ms'], dtype=torch.float32).to(device)
            unspliced = torch.tensor(subsample.layers['Mu'], dtype=torch.float32).to(device)

            nbrs = NearestNeighbors(n_neighbors=num_nbrs, algorithm='ball_tree').fit(latent)

            distances, indices = nbrs.kneighbors(latent)

            indices = torch.tensor(indices)

            outputs = genenet(latent)

            # Predict RNA velocities
            pred_spliced, pred_unspliced, alpha, beta, gamma, velocity, velocity_u = velo_pred(spliced, unspliced, outputs, umax, smax, dt)

            cosim = cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices)
            
            gene_loss = torch.sum(cosim)

            loss = gene_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("NaN or Inf detected in loss, exiting")
                break

            # Early stopping
            if loss < best_loss or best_loss == -1:
                best_net = genenet.state_dict()
                best_loss = loss
                imp_counter = 0
            else:
                imp_counter += 1

            if imp_counter > patience:
                logging.info("Early stopping triggered")
                break

            # Backpropagation 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # Load best model
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


def train_nbr(num_epochs, data, embed_basis, genenet, device, optimizer, num_nbrs=30, dt=0.3, batch=0.25, lambda_nbr = 0.2):
    """
    Fine-tune training GeneNet model using neighbor-based cosine similarity loss.

    This function fine-tunes the GeneNet model to predict RNA kinetic rates by aligning
    predicted velocities with the velocities of neighboring cells. The loss is
    defined as the negative sum of cosine similarities between a cell's predicted
    velocity and its neighbors' velocities.

    Parameters
    ----------
    num_epochs : int
        Maximum number of training epochs.
    data : AnnData
        Single-cell RNA dataset containing layers 'Ms' (spliced) and 'Mu' (unspliced),
        and embeddings in `data.obsm[embed_basis]`.
    embed_basis : str
        Key in `data.obsm` containing cell embeddings for training.
    genenet : nn.Module
        PyTorch GeneNet model that predicts kinetic rates.
    device : str or torch.device
        Device for training ('cpu' or 'cuda').
    optimizer : torch.optim.Optimizer
        Optimizer used for training the GeneNet model.
    num_nbrs : int, optional (default=30)
        Number of nearest neighbors to compute neighbor-based loss.
    dt : float, optional (default=0.3)
        Time step used in velocity prediction.
    batch : float, optional (default=0.25)
        Fraction of cells to sample per mini-batch.

    Returns
    -------
    None
        The function updates `data.layers` with the predicted velocities and kinetic rates:
        - "velocity_u": unspliced RNA velocity
        - "velocity": spliced RNA velocity (neighbor-averaged)
        - "alpha", "beta", "gamma": predicted kinetic rates
    """

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

        cosim_self = cosine_similarity(unspliced, spliced, pred_unspliced, pred_spliced, indices)

        gene_loss = torch.sum(cosim_self) - lambda_nbr * cosim / indices.shape[1]

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
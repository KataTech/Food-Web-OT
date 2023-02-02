import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from sklearn.manifold import MDS, TSNE
from typing import List

def ot_fgw(cost_s: torch.Tensor,
           cost_t: torch.Tensor,
           p_s: torch.Tensor,
           p_t: torch.Tensor,
           ot_method: str,
           gamma: float,
           num_layer: int,
           emb_s: torch.Tensor = None,
           emb_t: torch.Tensor = None):
    tran = p_s @ torch.t(p_t)
    # tran = torch.rand(p_s.size(dim=0), p_t.size(dim=0))
    if ot_method == 'ppa':
        dual = torch.ones(p_s.size()) / p_s.size(0)
        for i in range(num_layer):
            # print(f"Iteration {i}...")
            # assert not torch.isnan(cost_s).any(), "cost_s in OT_FGW"
            # assert not torch.isnan(cost_t).any(), "cost_t in OT_FGW"
            # assert not torch.isnan(p_s).any(), "p_s in OT_FGW"
            # assert not torch.isnan(p_t).any(), "p_t in OT_FGW"
            # assert not torch.isnan(tran).any(), "tran in OT_FGW"
            # assert not torch.isnan(emb_s).any(), "emb_s in OT_FGW"
            # assert not torch.isnan(emb_t).any(), "emb_t in OT_FGW"
            cost = cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t)
            # cost /= torch.max(cost)
            kernel = torch.exp(-cost / gamma) * tran
            b = p_t / (torch.t(kernel) @ dual)
            for i in range(5):
                dual = p_s / (kernel @ b)
                b = p_t / (torch.t(kernel) @ dual)
            tran = (dual @ torch.t(b)) * kernel
            # assert not torch.isnan(tran).any(), "tran in OT_FGW"
    elif ot_method == 'b-admm':
        all1_s = torch.ones(p_s.size())
        all1_t = torch.ones(p_t.size())
        dual = torch.zeros(p_s.size(0), p_t.size(0))
        for _ in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a

            dual = dual + gamma * (tran - aux)

            cost = cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t)
            # cost /= torch.max(cost)
            kernel_t = torch.exp(-(cost + dual) / gamma) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) * tran).sum()

    assert not torch.isnan(d_gw).any(), "d_gw in OT_FGW"
    assert not torch.isnan(tran).any(), "tran in OT_FGW"

    return d_gw, tran

def cost_mat(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             p_s: torch.Tensor,
             p_t: torch.Tensor,
             tran: torch.Tensor,
             emb_s: torch.Tensor = None,
             emb_t: torch.Tensor = None) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)
    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:
    f1(a) = a^2,
    f2(b) = b^2,
    h1(a) = a,
    h2(b) = 2b
    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have
    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
        emb_s: (ns, d) matrix
        emb_t: (nt, d) matrix
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    f1_st = ((cost_s ** 2) @ p_s).repeat(1, tran.size(1))
    f2_st = (torch.t(p_t) @ torch.t((cost_t ** 2))).repeat(tran.size(0), 1)
    cost_st = f1_st + f2_st
    cost = cost_st - 2 * cost_s @ tran @ torch.t(cost_t)

    # if emb_s is not None and emb_t is not None:
    #     tmp1 = emb_s @ torch.t(emb_t)
    #     tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
    #     tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
    #     assert not torch.isnan(cost).any(), f"Found NaN values in cost_mat \n sum of cost_t: {cost_t.sum()} \n tmp2tmp3: {(tmp2 @ torch.t(tmp3)).sum()}"
    #     cost += 0.5 * (1 - tmp1 / (tmp2 @ torch.t(tmp3)))
    
    # assert not torch.isnan(cost).any(), f"Found NaN values in cost_mat \n sum of cost_t: {cost_t.sum()} \n tmp2tmp3: {(tmp2 @ torch.t(tmp3)).sum()}"

    return cost

def fgwd(graph1, embedding1, prob1,
         graph2, embedding2, prob2, tran):
    """
    Computes the fused gromov-wasserstein distance between two graphs
    given their embedding, distributions, and transport
    """
    cost = cost_mat(graph1, graph2, prob1, prob2, tran, embedding1, embedding2)
    return (cost * tran).sum()

class StructuralDataSampler(Dataset):
    """Sampling point sets via minbatch"""

    def __init__(self, data: List, labels: List):
        """
        Parameters
        ----------
        data : list 
            A list of GraphOT objects 
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        curr_graph = self.data[idx]
        cost = curr_graph.get_cost()
        features = np.ones((curr_graph.get_size(), 1)) 
        features /= np.sum(features)
        dist = curr_graph.get_node_dist()
        dist = np.reshape(features, (features.shape[0], 1))

        features = torch.from_numpy(features).type(torch.FloatTensor)
        dist = torch.from_numpy(dist).type(torch.FloatTensor)
        cost = torch.from_numpy(cost).type(torch.FloatTensor)
        label = torch.LongTensor([self.labels[idx]])
        
        return [cost, dist, features, label]

class FGWF(nn.Module): 
    """
    A PyTorch implementation of the Gromov-Wasserstein factorization model. 
    Code largely adapted from Hongteng Xu's Relational-Factorization-Model repository. 

    Source: https://github.com/HongtengXu/Relational-Factorization-Model
    """
    def __init__(self, num_samples: int, size_atoms: int, ot_method: str="ppa", 
                 dim_embedding: int = 1, gamma: float=1e-1, gwb_layers: int=5, ot_layers: int=5):
        """
        Initialize an instance of a Gromov-Wasserstein Factorization (GWF) object. 

        Parameters: 
        -----------
        num_samples : int 
            The number of samples to draw for each training iteration 
        """
        super(FGWF, self).__init__()
        self.num_samples    = num_samples
        self.size_atoms     = size_atoms
        self.num_atoms      = len(size_atoms)
        self.dim_embedding = dim_embedding
        self.ot_method      = ot_method
        self.gwb_layers     = gwb_layers
        self.ot_layers      = ot_layers
        self.gamma          = gamma 

        # random initialization of atom weights 
        self.weights = nn.Parameter(torch.randn(self.num_atoms, self.num_samples))
        self.softmax = nn.Softmax(dim=0)

        # initialize the atoms and their embedding randomly with uniform distributions
        self.ps = []
        self.atoms = nn.ParameterList()
        self.embeddings = nn.ParameterList()
        for k in range(self.num_atoms): 
            # initialize atom to be a matrix (graph) of specified size
            atom = nn.Parameter(torch.randn(self.size_atoms[k], self.size_atoms[k]))
            # initialize a random embedding per each atom
            embedding = nn.Parameter(torch.randn(self.size_atoms[k], self.dim_embedding) / self.dim_embedding)
            # assume that atoms have uniform distribution
            dist = torch.ones(self.size_atoms[k], 1) / self.size_atoms[k]
            # append the computed values to the corresponding storages
            self.ps.append(dist)
            self.atoms.append(atom)
            self.embeddings.append(embedding)
        
        # include a sigmoidal layer
        self.sigmoid = nn.Sigmoid()

    def output_weights(self, idx: int = None):
        if idx is not None:
            return self.softmax(self.weights[:, idx])
        else:
            return self.softmax(self.weights)

    def output_atoms(self, idx: int = None):
        if idx is not None:
            return self.sigmoid(self.atoms[idx])
        else:
            return [self.sigmoid(self.atoms[idx]) for idx in range(len(self.atoms))]

    def fgwb(self,
             pb: torch.Tensor,
             trans: List,
             weights: torch.Tensor):
        """
        Solve GW Barycenter problem via the proximal point-based alternating optimization
        barycenter = argmin_{B} sum_k w[k] * d_gw(atom[k], B) via proximal point-based alternating optimization:
        step 1: Given current barycenter, for k = 1:K, we calculate trans[k] by the OT-PPA layer.
        step 2: Given new trans, we update barycenter by
            barycenter = sum_k trans[k] * atom[k] * trans[k]^T / (pb * pb^T)
        Args:
            pb: (nb, 1) vector (torch tensor), the empirical distribution of the nodes/samples of the barycenter
            trans: a dictionary {key: index of atoms, value: the (ns, nb) initial optimal transport}
            weights: (K,) vector (torch tensor), representing the weights of the atoms
        Returns:
            barycenter: (nb, nb) matrix (torch tensor) representing the updated GW barycenter
        """
        tmp1 = pb @ torch.t(pb)
        tmp2 = pb @ torch.ones(1, self.dim_embedding)
        graph = torch.zeros(pb.size(0), pb.size(0))
        embedding = torch.zeros(pb.size(0), self.dim_embedding)
        for k in range(self.num_atoms):
            graph_k = self.output_atoms(k)
            graph += weights[k] * (torch.t(trans[k]) @ graph_k @ trans[k])
            embedding += weights[k] * (torch.t(trans[k]) @ self.embeddings[k])
        graph = graph / tmp1
        embedding = embedding / tmp2
        return graph, embedding

    def forward(self, graph: torch.Tensor, prob: torch.Tensor, embedding: torch.Tensor,
                index: int, trans: List, tran: torch.Tensor):
        """
        For "n" unknown samples, given their disimilarity/adjacency matrix "cost" and distribution "p", we calculate
        "d_gw(barycenter(atoms, weights), cost)" approximately.
        Args:
            graph: (n, n) matrix (torch.Tensor), representing disimilarity/adjacency matrix
            prob: (n, 1) vector (torch.Tensor), the empirical distribution of the nodes/samples in "graph"
            embedding: (n, d) matrix (torch.Tensor)
            index: the index of the "cost" in the dataset
            trans: a list of (ns, nb) OT matrices
            tran: a (n, nb) OT matrix
        Returns:
            d_gw: the value of loss function
            barycenter: the proposed GW barycenter
            tran0: the optimal transport between barycenter and cost
            trans: the optimal transports between barycenter and atoms
            weights: the weights of atoms
        """
        # variables
        weights = self.softmax(self.weights[:, index])
        graph_b, embedding_b = self.fgwb(prob, trans, weights)
        d_fgw = fgwd(graph, embedding, prob, graph_b, embedding_b, prob, tran)
        return d_fgw, self.weights[:, index], graph_b, embedding_b

def train_usl(model,
              database,
              labels,
              size_batch: int = 16,
              epochs: int = 10,
              lr: float = 1e-1,
              weight_decay: float = 0,
              shuffle_data: bool = True,
              zeta: float = None,
              mode: str = 'fit',
              visualize_prefix: str = None, 
              verbose = False,
              save = False, 
              label_keys = None):
    """
    training a FGWF model
    Args:
        model: a FGWF model
        database: a list of data, each element is a list representing [cost, distribution, feature, label]
        size_batch: the size of batch, deciding the frequency of backpropagation
        epochs: the number epochs
        lr: learning rate
        weight_decay: the weight of the l2-norm regularization of parameters
        shuffle_data: whether shuffle data in each epoch
        zeta: the weight of the regularizer enhancing the diversity of atoms
        mode: fit or transform
        visualize_prefix: display learning result after each epoch or not
        verbose: display messages or not
        save: whether or not to save the image figures
        label_keys: a mapping from integer class to string representation
    """
    if mode == 'fit':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        n = 0
        for param in model.parameters():
            if n > 0:
                param.requires_grad = False
            n += 1
        # only update partial model's parameters
        optimizer = optim.Adam([list(model.parameters())[0]], lr=lr, weight_decay=weight_decay)
    model.train()

    data_sampler = StructuralDataSampler(database, labels)
    num_samples = data_sampler.__len__()
    index_samples = list(range(num_samples))
    index_atoms = list(range(model.num_atoms))

    best_loss = float("Inf")
    best_model = None
    for epoch in range(epochs):
        counts = 0
        t_start = time.time()
        loss_epoch = 0
        loss_total = 0
        d_fgw_total = 0
        reg_total = 0
        optimizer.zero_grad()

        if shuffle_data:
            random.shuffle(index_samples)

        for idx in index_samples:
            data = data_sampler.__getitem__(idx)
            graph = data[0]
            prob = data[1]
            emb = data[2]

            # Envelop Theorem
            # feed-forward computation of barycenter B({Ck}, w) and its transports {Trans_k}
            trans = []
            for k in range(model.num_atoms):
                graph_k = model.output_atoms(k).data
                emb_k = model.embeddings[k].data
                _, tran_k = ot_fgw(graph_k, graph, model.ps[k], prob,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb_k, emb)
                trans.append(tran_k)
            tran = torch.diag(prob[:, 0])

            d_fgw, _, _, _ = model(graph, prob, emb, idx, trans, tran)
            d_fgw_total += d_fgw
            loss_total += d_fgw

            if zeta is not None and mode == 'fit':
                random.shuffle(index_atoms)
                graph1 = model.output_atoms(index_atoms[0])
                
                emb1 = model.embeddings[index_atoms[0]]
                
                p1 = model.ps[index_atoms[0]]

                graph2 = model.output_atoms(index_atoms[1])
                emb2 = model.embeddings[index_atoms[1]]
                p2 = model.ps[index_atoms[1]]

                _, tran12 = ot_fgw(graph1.data, graph2.data, p1, p2,
                                   model.ot_method, model.gamma, model.ot_layers,
                                   emb1.data, emb2.data)

                reg = fgwd(graph1, emb1, p1, graph2, emb2, p2, tran12)

                reg_total += zeta * reg
                loss_total -= zeta * reg

            counts += 1
            if counts % size_batch == 0 or counts == num_samples:
                if counts % size_batch == 0:
                    num = size_batch
                else:
                    num = counts % size_batch
                loss_epoch += loss_total
                loss_total.backward()
                optimizer.step()

                print('-- {}/{} [{:.1f}%], loss={:.4f}, dgw={:.4f}, reg={:.4f}, time={:.2f}s.'.format(
                    counts, num_samples, counts / num_samples * 100.0,
                    loss_total / num, d_fgw_total / num, reg_total / num, time.time() - t_start))

                t_start = time.time()
                loss_total = 0
                d_fgw_total = 0
                reg_total = 0
                optimizer.zero_grad()

        if best_loss > loss_epoch.data / num_samples:
            best_model = copy.deepcopy(model)
            best_loss = loss_epoch.data / num_samples

        if verbose: 
            print('{}: Epoch {}/{}, loss = {:.4f}, best loss = {:.4f}'.format(
                mode, epoch + 1, epochs, loss_epoch / num_samples, best_loss))

        if visualize_prefix is not None and epoch + 1 == epochs:
            embeddings = tsne_weights(model)
            index = list(range(num_samples))
            labels = []
            for idx in index:
                labels.append(data_sampler[idx][-1].numpy()[0])
            labels = np.array(labels)
            num_classes = int(np.max(labels) + 1)

            plt.figure(figsize=(6, 6))
            for i in range(num_classes):
                plt.scatter(embeddings[labels == i, 0],
                            embeddings[labels == i, 1],
                            s=4,
                            label=label_keys[i])
            plt.legend()
            if verbose: 
                print('{}_tsne_{}.pdf'.format(visualize_prefix, epoch+1))
            if save: 
                plt.savefig('{}_tsne_{}.pdf'.format(visualize_prefix, epoch+1))
            plt.close()
    return best_model

def tsne_weights(model) -> np.ndarray:
    """
    Learn the 2D embeddings of the weights associated with atoms via t-SNE
    Returns:
        embeddings: (num_samples, 2) matrix representing the embeddings of weights
    """
    model.eval()
    features = model.weights.cpu().data.numpy()
    features = features.T
    if features.shape[1] == 2:
        embeddings = features
    else:
        embeddings = TSNE(n_components=2).fit_transform(features)
    return embeddings

def clustering(model) -> np.ndarray:
    """
    Taking the atoms as clustering centers, we cluster data based on their weights associated with the atoms
    """
    model.eval()
    feature = model.output_weights().data.numpy()
    return np.argmax(feature, axis=0)

def save_model(model, full_path):
    """
    Save trained model
    Args:
        model: the target model
        full_path: the path of directory
    """
    torch.save(model.state_dict(), full_path)

def load_model(model, full_path):
    """
    Load pre-trained model
    Args:
        model: the target model
        full_path: the path of directory
    """
    model.load_state_dict(torch.load(full_path))
    return model

def visualize_atoms(model, idx: int, threshold: float = 0.5, filename: str = None, save = False):
    """
    Learning the 2D embeddings of the atoms via multi-dimensional scaling (MDS)
    Args:
        model: a FGWF model
        idx: an index of the atoms
        threshold: the threshold of edge
        filename: the prefix of image name
        save: whether to save the image files
    Returns:
        embeddings: (size_atom, 2) matrix representing the embeddings of nodes/samples corresponding to the atom.
    """
    graph = model.output_atoms(idx).cpu().data.numpy()
    emb = model.embeddings[idx].cpu().data.numpy()

    if emb.shape[1] == 1:
        cost = graph + graph.T
        emb = MDS(n_components=2, dissimilarity='precomputed').fit_transform(cost)
    elif emb.shape[1] > 2:
        emb = TSNE(n_components=2).fit_transform(emb)

    graph[graph >= threshold] = 1
    graph[graph < threshold] = 0

    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=80)
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if graph[i, j] > 0 and i != j:
                pair = np.asarray([i, j])
                x = emb[pair, 0]
                y = emb[pair, 1]
                plt.plot(x, y, 'r-')

    if save: 
        if filename is None:
            plt.savefig('atom_{}.pdf'.format(idx))
        else:
            plt.savefig('{}_{}.pdf'.format(filename, idx))

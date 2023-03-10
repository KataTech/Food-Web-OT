"""
This is an experimental script for conducting Gromov-Wasserstein Graph Factorization
and Graph Clustering.
"""

import model.fused_gwf as FGWF
import numpy as np
import pickle
import networkx as nx
import os
from model.fused_gwf import StructuralDataSampler
from model.graph import GraphOT_Factory
from sklearn.cluster import KMeans

np.random.seed(25)

# TODO: SET EXPERIMENT RESULT DIR
RESULT_DIR = "scratch/results"
MODEL_DIR = "scratch/models"
NAME = "cycle_star"

# TODO: SET DATA PATH
data_path = "data/processed"

# TODO: SET MODEL PARAMETERS
num_atoms = 20
size_atoms = np.random.randint(20, 100, num_atoms)
print(f"Atom Sizes: {size_atoms}")
ot_method = 'ppa'       # either `ppa` or `b-admm`
gamma = 1e-1            
gwb_layers = 10
ot_layers = 50

# TODO: SET ALGORITHM PARAMETERS
size_batch = 64         # the batch sizee 
epochs = 10             # the number of epochs to train for
lr = 0.25               # learning rate of the optimizer
weight_decay = 0
shuffle_data = True     
zeta = None             # the weight of diversity regularizer
mode = 'fit'            

# generate 25 random cycles and 25 random path graphs 
graph_dict = {}
labels = []
labels_real = []
for i in range(250): 
    graph_dict[i] = nx.cycle_graph(np.random.randint(low=10, high=50))
    labels.append(0)
for j in range(250): 
    graph_dict[j + 250] = nx.star_graph(np.random.randint(low=10, high=50))
    labels.append(1)
graph_factory = GraphOT_Factory(graph_dict, cost_method="adjacency")
graph_factory.save("scratch/exp_graph_fact.pkl")
graph_data = graph_factory.to_list()
label_keys = {0: "Cycle", 1: "Path"}

# since our test script factory does not anticipate 
# node features, we shall set dimensional embedding to 1
dim_embedding = 1
# initialize a data sampler and keep track of the labels for each data sampler
data_sampler = StructuralDataSampler(graph_data, labels)


model = FGWF.FGWF(num_samples=len(graph_data),
                size_atoms=size_atoms,
                dim_embedding=dim_embedding,
                ot_method=ot_method,
                gamma=gamma,
                gwb_layers=gwb_layers,
                ot_layers=ot_layers)
model = FGWF.train_usl(model, graph_data, labels,
                        size_batch=size_batch,
                        epochs=epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        shuffle_data=shuffle_data,
                        zeta=zeta,
                        mode=mode,
                        visualize_prefix=os.path.join(RESULT_DIR, NAME),
                        save=True,
                        verbose=True, 
                        label_keys=label_keys)
model.eval()
features = model.weights.cpu().data.numpy()
embeddings = features.T
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
pred = kmeans.fit_predict(embeddings)
best_acc = max([1 - np.sum(np.abs(pred - labels)) / len(graph_data),
                1 - np.sum(np.abs((1 - pred) - labels)) / len(graph_data)])
print(f"Best Accuracy: {best_acc}")

# save the resulting atoms 
atom_dict = {}
for i in range(num_atoms):
    atom_dict[i] = (size_atoms[i], model.output_atoms(i).cpu().data.numpy())
with open(NAME+"_atom.pkl", "wb") as f: 
    pickle.dump(atom_dict, f)

# save the model itself
FGWF.save_model(model, os.path.join(MODEL_DIR, '{}_fgwf.pkl'.format(NAME)))

    
"""
This is an experimental script for conducting Gromov-Wasserstein Graph Factorization
and Graph Clustering.
"""

import model.fused_gwf as FGWF
import numpy as np
import seaborn as sns
import os, pickle
from model.fused_gwf import StructuralDataSampler
from model.graph import GraphOT_Factory
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

np.random.seed(25)

# TODO: SET EXPERIMENT RESULT DIR
RESULT_DIR = "scratch/results"
MODEL_DIR = "scratch/models"
data_path = "data/processed"
NAME = "food_web"

# TODO: SET MODEL PARAMETERS
num_atoms = 50
size_atoms = np.random.randint(20, 70, num_atoms)
print(f"Atom Sizes: {size_atoms}")
ot_method = 'ppa'       # either `ppa` or `b-admm`
gamma = 1e-1            
gwb_layers = 5
ot_layers = 50

# TODO: SET ALGORITHM PARAMETERS
size_batch = 32         # the batch sizee 
epochs = 20             # the number of epochs to train for
lr = 0.25               # learning rate of the optimizer
weight_decay = 0 
shuffle_data = True     
zeta = None             # the weight of diversity regularizer
mode = 'fit'            

# extract the graph data and the biome information
with open("data/processed/foodweb_loc2nxgraph.pkl", "rb") as f: 
    loc2graph = pickle.load(f)
with open("data/processed/foodweb_loc2biome.pkl", "rb") as f: 
    loc2biome = pickle.load(f)

# repopulate a graph factory based on adjacency matrix
graph_factory = GraphOT_Factory(loc2graph, cost_method="adjacency")
graph_factory.save("scratch/foodweb_adjacency.pkl")
graph_data = graph_factory.to_list()

# set labels based on the biome information
label_keys = {}
biome2ct = {}
biomes = set()
labels = []
label_ct = -1
for location in graph_factory.names: 
    biome = loc2biome[location]
    # encounter a new biome, make sure to update it
    if biome not in biome2ct: 
        label_ct += 1
        biome2ct[biome] = label_ct
        label_keys[label_ct] = biome
    # extract the label of this particular graph
    label = biome2ct[biome]
    labels.append(label)

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

# measure the predictive accuracy and save the confusion matrix plot
conf_mat = confusion_matrix(pred, labels)
acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
print('Overall accuracy: {} %'.format(acc*100))
conf_plot = sns.heatmap(conf_mat/np.sum(conf_mat), annot=True, 
            fmt='.2%')
conf_plot.get_figure().savefig(os.path.join(RESULT_DIR, '{}_conf.png'.format(NAME)))

# save the resulting atoms 
atom_dict = {}
for i in range(num_atoms):
    atom_dict[i] = (size_atoms[i], model.output_atoms(i).cpu().data.numpy())
with open(NAME+"_atom.pkl", "wb") as f: 
    pickle.dump(atom_dict, f)

# save the model itself
FGWF.save_model(model, os.path.join(MODEL_DIR, '{}_fgwf.pkl'.format(NAME)))

    
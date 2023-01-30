import sys
import networkx as nx
sys.path.insert(0, '../src/')
from util import gwd_growth_experiment

# Set parameter for gw_dist growth experiments
base_sizes = [5]
# base_sizes = [5, 10, 25, 100, 500]
numComparisons = 5
step = 1
colors = ["blue", "orange", "green", "red", "purple"]

# Test Run
title = "Test"
log = gwd_growth_experiment(nx.cycle_graph, base_sizes, step, numComparisons, title, colors, 
                            verbose=1, save_transport=True, save_fig=True, save_log=True, iter=5)

# # Run experiment for cycle graphs 
# title = "Cycle Graphs"
# log = gwd_growth_experiment(nx.cycle_graph, base_sizes, step, numComparisons, title, colors, 
#                             verbose=1, save_transport=True, save_fig=True, save_log=True, iter=5)

# # Run experiment for path graphs 
# title = "Path Graphs"
# log = gwd_growth_experiment(nx.path_graph, base_sizes, step, numComparisons, title, colors, 
#                             verbose=1, save_transport=True, save_fig=True, save_log=True, iter=5)

# # Run experiment for star graphs 
# title = "Star Graphs"
# log = gwd_growth_experiment(nx.star_graph, base_sizes, step, numComparisons, title, colors, 
#                             verbose=1, save_transport=True, save_fig=True, save_log=True, iter=5)

# # Run experiment for complete graphs 
# title = "Complete Graphs"
# log = gwd_growth_experiment(nx.complete, base_sizes, step, numComparisons, title, colors, 
#                             verbose=1, save_transport=True, save_fig=True, save_log=True, iter=5)
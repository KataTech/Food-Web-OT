"""
This script contains a utility functions that are helpful for various experiments
related to this project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle, time
from model.graph import GraphOT
from model.ot_gromov import entropic_gw

SNS_CMAP = "BuPu"

def vis_transport(T, title=None, percent=True, 
                    cmap=SNS_CMAP, save_fig=False, save_path=None): 
    """
    Visualize the supplied optimal transport matrix. 

    Parameters
    ----------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between two spaces denoted S and T respectively
    title : string
        Title of the heatmap
    percent : boolean
        If True, scale the heatmap entries by the percentage of mass transported 
        from point s in S to point t in T relative to the total mass 
        at point s. Otherwise, just output the transport values as provided. 
    cmap : string
        Color map for heatmap
    save_fig : boolean
        If True, save the output heatmap to `save_path` location. Otherwise, 
        display the plot.
    save_path : string
        Location to store the output figure. If None, store current directory.

    Returns
    -------
    None
    """
    # compute optimal ratio of the heatmap based on T dimensions
    n_rows, n_cols = T.shape
    ratio = (n_rows / (n_rows + n_cols), n_cols / (n_rows + n_cols))
    ratio = (round(10 * ratio[1]), round(10 * ratio[0]))
    # if percent is True, transform T to display percentage mass transfers
    if percent: 
        T = (T.T / np.sum(T, axis=1)).T
    # round all transport values to 2 decimal places 
    T = np.round(T, decimals=2)
    # compute plot based on output mode implied by `save_fig`
    plt.figure(figsize=ratio)
    vis = sns.heatmap(T, cmap=cmap, annot=True)
    if title is not None: 
        vis.set_title(title)
    # save the figure if appropriate
    if save_fig:
        vis_title = title if title is not None else "scratch"
        if save_path is None: 
            vis_title = vis_title + ".png"
        else: 
            vis_title = os.path.join(save_path, vis_title + ".png")
        plt.savefig(vis_title)

def gwd_growth_experiment(generator, base_sizes, step, numComparison, title, colors, iter=10,
                      verbose=False, save_transport=False, save_fig=False, save_log=False, save_path=None):
    """
    Run an experiment for gromov-wasserstein distance growth between graphs of varying sizes. 

    Parameters
    ----------
    generator : func: int -> nx.Graph
        The generator function of a graph. It should take one input, which is 
        an integer representing the number of nodes, and return a networkx graph
    base_sizes : list[int]
        A list representing the sizes of the nodes for the base graphs 
    step : int 
        The size difference between the nodes of subsequent graphs 
    numComparison : int 
        The number of comparison graphs to generate per base graph
    title : str
        The title of the plot 
    colors : lst[str]
        A list of colors to use for the plot
    iter : int
        The number of times to compute gromov-wasserstein using random
        initialization transport matrices
    verbose : int
        The level of messages to output. 
        0: No messages 
        1: Elapsed Time
        2: Elapsed Time + Progression Bar
        3: Elapsed Time + Progression Bar + Convergence Warnings
    save_transport : bool 
        Whether or not to save the transport matrices 
    save_fig : bool 
        Whether or not to save the output visualization
    save_log : bool
        Whether or not to save the log dictionary
    save_path : str
        The path for saving objects. If none, store in current directory.
    

    Returns
    ----------
    log : dictionary
        A record of experimental results. 
    """

    assert len(colors) > len(base_sizes), "ERROR: Not enough colors"

    # Initialize plot settings 
    plt.figure(figsize=(10, 10))
    plt.xlabel('Difference in Size')
    plt.ylabel('GW Distance')
    plt.title(title)

    # Initialize iterator for difference in base and alternative graphs
    iterator = np.arange(0, numComparison * step, step)

    # Initialize log dictionary for information tracking
    log = {}
    log["gw_dist"] = {}
    log["gw_dist_std"] = {}
    log["base_sizes"] = base_sizes
    log["step"] = step
    log["numComparison"] = numComparison
    if save_transport: 
        log["trans"] = {}

    # Create a separate line plot for every base_size
    for base_id in range(len(base_sizes)): 
        if verbose >= 1: 
            start = time.time()
        # Initialize a storage vector for gromov-wasserstein distances
        res_mean = np.zeros(numComparison)
        res_std = np.zeros(numComparison)
        # Extract node_dist and cost from base graph
        base_size = base_sizes[base_id]
        base_graph = GraphOT(generator(base_size))
        p_s, cost_s = base_graph.extract_info()
        if verbose >= 2: 
            print(f"Processing base graph size = {base_size}")
        # Generate all base, alt graph pairs
        for i, diff in enumerate(iterator): 
            alt_size = base_size + diff
            alt_graph = GraphOT(generator(alt_size))
            p_t, cost_t = alt_graph.extract_info()
            # compute the gromov-wasserstein distance between the two
            d_gw = []
            if save_transport: 
                best_gwd = float('inf')
                best_trans = None
            # compute gromov-wasserstein distance for `iter` times under random initialization
            for j in range(iter): 
                gw_dist, trans, conv = entropic_gw(cost_s, cost_t, p_s, p_t, random_init=True, sinkhorn_warn=False)
                if verbose == 3 and not conv: 
                    print(f"WARNING: {(base_graph, alt_graph)} failed to converge at iteration j")
                d_gw.append(gw_dist)
                if save_transport and gw_dist < best_gwd:
                    best_trans = trans
            # compute the average and standard deviation
            d_gw = np.array(d_gw)
            res_mean[i] = np.mean(d_gw) 
            res_std[i] = np.std(d_gw)
            # save information to the log 
            log["gw_dist"][(base_size, alt_size)] = res_mean[i]
            log["gw_dist_std"][(base_size, alt_size)] = res_std[i]
            if save_transport: 
                log["trans"][(base_size, alt_size)] = best_trans
            # print progression bars 
            if verbose >= 2: 
                if np.round(numComparison * 0.25) == i: 
                    print("25%===>")
                elif np.round(numComparison * 0.50) == i: 
                    print("50%=========>")
                elif np.round(numComparison * 0.75) == i: 
                    print("75%=================>")
        if verbose >= 1: 
            end = time.time()
            print(f"Base Size {base_size} Elapsed Time: {end - start}")
        # visualize results for current base size 
        plt.plot(iterator, res_mean, label=f"Base size = {base_size}", color=colors[base_id])
        plt.fill_between(iterator, res_mean - res_std, res_mean + res_std,
                            color=colors[base_id], alpha=0.2)
    # display the label for each plot
    plt.legend()
    # store the results
    if save_path is None: 
        save_path = title.replace(" ", "_")
    else: 
        save_path = os.path.join(save_path, title.replace(" ", "_"))
    if save_fig:
        plt.savefig(save_path + ".png")
    if save_log: 
        filename_pkl = save_path + "_log.pkl"
        with open(filename_pkl, 'wb') as f:
            pickle.dump(log, f)
    # output the log information
    return log



        








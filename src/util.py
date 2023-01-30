"""
This script contains a utility functions that are helpful for various experiments
related to this project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os
sys.path.insert(0, '../../')
from constants import SCRATCH_PATH, SNS_CMAP


def vis_transport(T, title=None, percent=True, 
                    cmap=SNS_CMAP, save_fig=False, save_path=SCRATCH_PATH): 
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
        Location to store the output figure

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
        vis_title = os.join(save_path, vis_title + ".png")
        plt.savefig(vis_title)


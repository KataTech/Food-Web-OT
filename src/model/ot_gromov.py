"""
This python file contains useful functions for computing gromov-wasserstein distance 
and gromov-wasserstein discrepancies.
"""
import numpy as np
from ot.bregman import sinkhorn_log
from ot.utils import list_to_array
from ot.backend import get_backend
from ot.gromov import gwggrad, gwloss, init_matrix


def entropic_gw(C1, C2, p, q, loss_fun="square_loss", epsilon=0.01, max_iter=1000, tol=1e-9, verbose=False, 
                log=False, random_init=False, init_trans=None, sinkhorn_warn=True):
    """
    Compute the entropically regularized gromov-wasserstein distance 
    between source object and target object in the log space. 

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p :  array-like, shape (ns,)
        Distribution in the source space
    q :  array-like, shape (nt,)
        Distribution in the target space
    loss_fun :  string
        Loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    random_init : bool
        Initialize the transport matrix randomly
    init_trans : array-like, shape (ns, nt)
        Initial transport for optimization. If none, take the outer product 
        of p and q (uniform mass distribution)
    sinkhorn_warn : bool
        Whether the sinkhorn convergence warning is turned on

    Returns
    -------
    gw_dist : float
        Gromov-Wasserstein distance
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces
    converged : boolean
        Whether Sinkhorn algorithm has converged

    References
    ----------
    - Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    - POT Documentation:
        https://pythonot.github.io/_modules/ot/gromov.html#entropic_gromov_wasserstein
    """

    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    if init_trans is None:
        if random_init: 
            T = np.random.rand(p.shape[0], q.shape[0])
        else: 
            T = nx.outer(p, q)
    else: 
        T = init_trans

    constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}

    while (err > tol and cpt < max_iter):

        Tprev = T

        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)

        # perform sinkhorn update in the log-space
        T = sinkhorn_log(p, q, tens, epsilon, method='sinkhorn', warn=sinkhorn_warn)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    gw_dist = gwloss(constC, hC1, hC2, T)
    converged = cpt != max_iter
    return gw_dist, T, converged




def corheatmap(Xc, n_max=600, val_excl=0, ct='rho', fthresh=0.8, title=''):
    """
        Correlation Heatmap
        Args:
            Xc: Pandas or Numpy object of rank 2 with features in columns
            n_max: Maximum number of allowed missing/excluded observations
            val_excl: Observations values that are excluded from analysis (0 for Bruker fits)
            ct: Correlation type (rho for rank correlation or r for Pearson's correlation)
        Returns:
            Tuple of two: 1. tuple of ax, fig, 2: correlation matrix (numpy rank 2)
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import pandas as pd
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from utility import _rho, _cov_cor

    if isinstance(Xc, pd.DataFrame):
        labs = Xc.columns.values
        Xc = Xc.to_numpy()
    else:
        if not isinstance(Xc, np.ndarray):
            raise ValueError('Provide pandas DataFrame or numpy ndarray')
        labs = np.array(['Feat ' + str(x) for x in range(Xc.shape[0])])

    if not Xc.dtype.kind in set('buifc'):
        raise ValueError('Provide numeric values')

    idx_keep = np.where(np.sum((Xc == val_excl) | (Xc == np.nan), 0) <= n_max)[0]
    if len(idx_keep) < 2:
        raise ValueError('Number of features with missing/excluded values below n_min')
    Xc = Xc[:, idx_keep]
    labs = labs[idx_keep]

    cc = np.zeros((Xc.shape[1], Xc.shape[1]))
    tups = itertools.combinations(range(Xc.shape[1]), r=2)

    if ct == 'rho':
        cfun = _rho
    else:
        cfun = _cov_cor

    for i in tups:
        x = Xc[:, i[0]]
        y = Xc[:, i[1]]
        idx = np.where(~(x == val_excl) & ~(y == val_excl))[0]
        xcov, xcor = cfun(np.log(x[idx]), np.log(y[idx]))
        cc[i[0], i[1]] = xcor

    cs = cc + cc.T
    ft = fthresh * cs.shape[0]
    idx_keep = np.where(np.nansum(np.isnan(cs), 1) < ft)[0]
    if len(idx_keep) < 2:
        raise ValueError('Number of selected features < 2: Decrease ftrehs parameter')
    ps = cs[np.ix_(idx_keep, idx_keep)]
    np.fill_diagonal(ps, 1)
    ps1 = squareform(1-ps)

    # reorder based on ward linkage
    Z = linkage(ps1, method='ward')
    cord = Z[:, 0:2].ravel()
    cord = cord[cord < ps.shape[0]].astype(int)
    labs = labs[cord]
    ps = ps[np.ix_(cord, cord)]

    fig, ax = plt.subplots(tight_layout=True)
    heatmap = ax.pcolor(ps, cmap=plt.cm.rainbow, norm=Normalize(vmin=-1, vmax=1))
    fig.colorbar(heatmap)
    ax.set_title(title)

    ax.set_xticks(np.arange(ps.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(ps.shape[1]) + 0.5, minor=False)
    ax.set_xticklabels(labs[idx_keep], minor=False, rotation=90)
    ax.set_yticklabels(labs[idx_keep], minor=False)

    fig.show()

    return (fig, ax), ps
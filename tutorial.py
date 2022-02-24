import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mva


# load data set and create X and Y variable
iris = load_iris()

X = iris['data']
Y = iris['target']

an=pd.DataFrame({'Y': Y})

# perform correlation analysis
S =np.concatenate([X, Y[..., np.newaxis]], axis=1)
mva.cor_heatmap(S, n_max=600, val_excl=0, ct='rho', title='')



# perform pca
mod=mva.pca(X, ppm=None, nc=2, center=True, scale='uv')
mod.plot_scores(an=an, hue='Y')

# create two class scenario and perform opls
# do discriminant analysis (categorical Y variable)
# do regression analysis (continuous Y variable)
idx = np.where((Y == 1) | (Y == 2))[0]

X = X[idx]
Y = Y[idx] # regression
Yc = iris.target_names[iris.target][idx] # DA


# perform opls with mc and k-fold resampling
ss=mva.o_pls(X, Yc, ctype='opls', cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=mva.o_pls(X, Y, ctype='opls', cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=mva.o_pls(X, Yc, ctype='opls', cvtype='k-fold', pars={'k': 7, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=mva.o_pls(X, Y, ctype='opls', cvtype='k-fold', pars={'k': 4, 'split_train': 2 / 3})
x, y = ss.plot_scores()
self = ss
ss.plot_load_ms()



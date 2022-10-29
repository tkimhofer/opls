import pandas as pd
from sklearn.datasets import load_iris, load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utility import corheatmap
from pca import Pca
from opls import O_pls


# load wine toy data set
wine  = load_wine()
X=pd.DataFrame(wine['data'], columns=wine['feature_names'])
Y = wine['target']

# correlation heatmap
corheatmap(X)

# calculate Principal Components Analysis (PCA)
mod=Pca(X, ppm=None, nc=2, center=True, scale='uv')

# visualise PCA scores and loadings
an = pd.DataFrame(wine['target'], columns=['target'])
mod.plot_scores(an=an, hue='target')
mod.plot_load(pc=1, ids=wine['feature_names'])
mod.plot_load(pc=2, ids=wine['feature_names'])


# create two class scenario and perform O-PLS regression (R)
idx = np.where((Y == 1) | (Y == 2))[0]

X = np.array(X.iloc[idx])
Y = Y[idx] # regression

# calculate O-PLSR with k-fold cross validation
ss=O_pls(X, Y, ctype='opls', cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3})

# visualise O-PLS scores and loadings
ss.plot_scores()
ss.plot_load(ids=wine['feature_names'])

# for O-PLS discriminant analysis (DA) or Monte Carlo - cross validation see `help(O_pls)`
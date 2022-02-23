# Partial Least Squares and Orthogonal Partial Least Squares
# Autoselect number of components using k-fold or Monte Carlo cross-validation

#

# import example data
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pls

iris = load_iris()

X = iris['data']
Y = iris['target']
idx = np.where((Y == 1) | (Y == 2))[0]

X = X[idx]
Y = Y[idx]
#Y = np.array([str(x) for x in Y])

Yc = iris.target_names[iris.target][idx]

# test=pls.comp_data(X, Yc, x_center=True, x_stype='uv', y_center=False, y_stype=None)
# plt.scatter(test.Xsc[:,0], test.Xsc[:,1], c=Y)


ss=pls.o_pls(X, Yc, ctype='opls', cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=pls.o_pls(X, Y, ctype='opls', cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=pls.o_pls(X, Yc, ctype='opls', cvtype='k-fold', pars={'k': 7, 'split_train': 2 / 3})
x, y = ss.plot_scores()
ss.plot_load_ms()

ss=pls.o_pls(X, Y, ctype='opls', cvtype='k-fold', pars={'k': 4, 'split_train': 2 / 3})
x, y = ss.plot_scores()
self = ss
ss.plot_load_ms()




### import pym8 package

# determine directory of pym8 clone
d='/Users/tk2812/py/pym8'

# add pym8 directory to python library path variable
import sys
sys.path.append(d)

# import pym8 package
from mm8 import *

# pym8 package is structured in 5 modules called: reading, preproc, analyse, plotting, utility
# each module contains code for specific tasks
# for example, the module reading contains code to read in 1 or 2D spectra


### read-in 1D spectra

# define directory of NMR experiments
path='/Users/tk2812/Downloads'

# determine experiment type and number in directory
# the following code prints out a summary of experiments into the console
# copy 1D NMR experiment type(s) (e.g. PROF_PLASMA_NOESY and PROF_PLASMA_CPMG)
exp = reading.list_exp(path, return_results=True, pr=True)

# import experiments, the following function generates 3 variables:
# X - NMR spectral matrix,
# ppm - matching ppm array,
# meta - metadata and TopSpin acquisition, processing information (files acqus and procs)
# every function comes with documentation, eg. type `help(reading.import1d_procs)` into the console
X, ppm, meta =  reading.import1d_procs(flist=exp, exp_type=['PROF_PLASMA_NOESY'], eretic=True)

# interactive visualisation of spectra
# 1. IDE supported plotting
# On normal desktop computer, up to 1000 spectra should work without much latency
plotting.spec(X, ppm) # full ppm range
plotting.spec(X, ppm, shift=[-0.1, 0.1]) # selected ppm range

# 2. Browser supported plotting
# this is computationally less efficient - better to reduce nb of spectra or/and chemical shift area
plotting.ispec(X, ppm, shift=[-0.1, 0.1])



### preprocessing: remove res water signal and cap up and lowfield ends
# calibrate blood derived spectra using glucose, alanine or lactate
# select doublet position and J constant, both values with be used to identify the right doublet in each spectrum
# you can use the plotting functions to get these parameters

# lets calibrate to alanine
plotting.spec(X, ppm, shift=[1.3, 1.45])
cent_ppm = 1.354
j_hz = 8

# function returns calibrated spectra (Xc)
Xc = preproc.calibrate_doubl(X = X, ppm = ppm, cent_ppm = cent_ppm, j_hz = j_hz, lw=10, tol_ppm=0.015,  niter_bl=5)

# check if calibration was successfull
# if double is not aligned, adjust parameters j_hz and location
# typically j_hz can be relaxed to improve calibration
plotting.spec(Xc, ppm, shift=[1.3, 1.45])

# excise spectral regions (cap up/lowfield, residual water, edta signal, etc)


# define spectral intervals that you want to keep (rather than remove!)
# then excise unwanted regions and visualise results
upfield = [0.25, 4.5]
downfield = [5, 9.5]

Xe, ppe = preproc.excise1d(Xc, ppm, shifts=[upfield, downfield])
plotting.spec(Xe, ppe)

# basline correction for std 1d experiment (generally not needed from CPMG)
Xb = preproc.bline(Xe, multiproc=True)

# compare a spectrum before after bl correction
# first spectrum is used (python starts indexing at 0)
ax1 = plotting.spec(Xe[0], ppe, label='raw')
fig, ax1 = plotting.spec(Xb[0], ppe, ax=ax1, label='bline corrected')
fig.legend()


# perform OPLS
Y=np.sum(X[:,utility.get_idx(ppm, [2.6, 2.55])], axis=1)

ss=o_pls(Xb, Y, ctype='opls', cvtype='mc', pars={'k': 20, 'split_train': 2 / 3})
ss.nc

ss.plot_load(ppe, 'o0')


plt.scatter(ss.c_full[0].comp.t, ss.c_full[0].comp.to)
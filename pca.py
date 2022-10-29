import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from utility import _cov_cor


class Pca:
    # methods: plot_scores, plot_load
    """
    Create PCA class

    Args:
          X: np matrix rank 2
          ppm: np array rank 1 if NMR analysis, else None
          pc: Number of desired principal components
          center: bool, mean centering
          scale: scaling type, currently only unit variance scaling implemented ('uv')
    Returns:
          pca class
    """

    def __init__(self, X, ppm, nc=2, center=True, scale='uv'):
        # from matplotlib.pyplot import plt
        self.X = X if isinstance(X, np.ndarray) else np.array(X)
        self.ppm = ppm
        self.nc = nc
        self.center = center
        self.scale = scale
        self.means = np.mean(self.X, 0, keepdims=True)
        self.std = np.std(self.X, 0, keepdims=True)

        if (self.std == 0).any():
            print('Matrix contains columns with zero standard deviation - replacing these with eps = 1e-7')
            self.std[self.std == 0] = 1e-7

        self.Xsc = (self.X - self.means) / self.std

        if self.center and (self.scale == 'uv'):
            X = self.Xsc
        else:
            if center:
                X = self.X
            if (scale == 'uv'):
                X = X / self.std
        self.ss_tot = np.sum((X) ** 2)
        self.pca_mod = PCA(n_components=nc).fit(X)
        self.t = self.pca_mod.transform(X)
        self.p = self.pca_mod.components_

        tvar = np.sum(X ** 2)
        r2 = []
        for i in range(self.t.shape[1]):
            xc = np.matmul(self.t[:, i][np.newaxis].T, self.p[i, :][np.newaxis])
            r2.append((np.sum(xc ** 2) / tvar) * 100)
        self.r2 = r2

        xcov, xcor = _cov_cor(self.t, self.X)

        self.Xcov = xcov
        self.Xcor = xcor

    def plot_scores(self, an, pc=[1, 2], hue=None, labs=None, legend_loc='right'):
        # methods: plot_scores, plot_load
        import seaborn as sns
        """
        Plot PCA scores (2D)

        Args:
              an: Pandas DataFrame where colouring variable given as column
              pc: List of principal component indices, start counting at 1 (not zero) and with a length of two
              hue: Column name of colouring variable in an
              labs: None or list of strings containing scores plot labels
              legend_loc: Legend location given as string ('right', 'left') 
        Returns:
              sns plotting object
        """
        self.an = an
        pc = np.array(pc)
        cc = ['t' + str(sub) for sub in np.arange(self.t.shape[1]) + 1]
        df = pd.DataFrame(self.t, columns=cc)

        if self.an.shape[0] != df.shape[0]:
            raise ValueError('Dimensions of PCA scores and annotation dataframe don\'t match.')
            # return Null

        ds = pd.concat([df.reset_index(drop=True), an.reset_index(drop=True)], axis=1)
        # ds=ds.melt(id_vars=an.columns.values)
        # ds=ds.loc[ds.variable.str.contains('t'+str(pc[0])+"|t"+str(pc[1]))]

        # calculate Hotelling`s T2 ellipse
        x = ds.loc[:, 't' + str(pc[0])]
        y = ds.loc[:, 't' + str(pc[1])]
        theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
        circle = np.array((np.cos(theta), np.sin(theta)))
        cov = np.cov(x, y)
        ed = np.sqrt(chi2.ppf(0.95, 2))
        ell = np.transpose(circle).dot(np.linalg.cholesky(cov) * ed)
        a, b = np.max(ell[:, 0]), np.max(ell[:, 1])  # 95% ellipse bounds
        t = np.linspace(0, 2 * np.pi, 100)

        el_x = a * np.cos(t)
        el_y = b * np.sin(t)

        fg = sns.FacetGrid(ds, hue=hue)
        fg.axes[0][0].axvline(0, color='black', linewidth=0.5, zorder=0)
        fg.axes[0][0].axhline(0, color='black', linewidth=0.5, zorder=0)

        ax = fg.facet_axis(0, 0)

        # fg.xlabel('t'+str(pc[0])+' ('+str(self.r2[pc[0]-1])+'%)')
        # fg.ylabel('t'+str(pc[1])+' ('+str(self.r2[pc[1]-1])+'%)')
        ax.plot(el_x, el_y, color='gray', linewidth=0.5, )
        fg.map(sns.scatterplot, 't' + str(pc[0]), 't' + str(pc[1]), palette="tab10")

        fg.axes[0][0].set_xlabel('t' + str(pc[0]) + ' (' + str(np.round(self.r2[pc[0] - 1], 1)) + '%)')
        fg.axes[0, 0].set_ylabel('t' + str(pc[1]) + ' (' + str(np.round(self.r2[pc[1] - 1], 1)) + '%)')

        if labs is not None:
            if (len(labs) != len(x)):  raise ValueError('Check length of labs')
            for i in range(len(labs)):
                fg.axes[0, 0].annotate(labs[i], (x[i], y[i]))
        fg.add_legend(loc=legend_loc)

        return fg

    def plot_load_nmr(self, pc=1, shift=[0, 10]):
        from matplotlib.collections import LineCollection
        # from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.pyplot as plt
        """
        Plot statistical reconstruction of PCA loadings 

        Args:
              pc: Index of principal components, starting at 1
              shift: Chemical shift range (list of 2)
        Returns:
              plotting object
        """

        # print(shift)
        shift = np.sort(shift)
        x = self.ppm
        y = self.Xcov[pc, :]
        z = self.Xcor[pc, :]
        idx = np.where((x >= shift[0]) & (x <= shift[1]))[0]
        x = x[idx]
        y = y[idx]
        z = np.abs(z[idx])
        xsub = self.X[:, idx]

        fig, axs = plt.subplots(2, 1, sharex=True)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(z.min(), z.max())
        lc = LineCollection(segments, cmap='rainbow', norm=norm)
        # Set the values used for colormapping
        lc.set_array(z)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs)
        dd = (x.max() - x.min()) / 30
        axs[0].set_xlim(x.max() + dd, (x.min() - dd))
        axs[0].set_ylim(y.min() * 1.1, y.max() * 1.1)

        axs[1].plot(x, xsub.T, c='black', linewidth=0.3)

        return (axs, fig)

        #
        # df=pd.DataFrame({'ppm':x, 'cov':y, 'cor':np.abs(z)})
        # df['fac']='PCA: p'+str(pc)
        #
        # rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
        # g=pn.ggplot(pn.aes(x='ppm', y='cov', color='cor'), data=df)+pn.geom_line()+pn.scale_colour_gradientn(colors=rainbow, limits=[0,1])+pn.theme_bw()+pn.scale_x_reverse()+ pn.scale_y_continuous(labels=scientific_format(digits=2))+pn.facet_wrap('~fac')+pn.labs(color='|r|')
        # return(g)

    def plot_load(self, pc=1, ids=None):
        import matplotlib.pyplot as plt
        import matplotlib._color_data as mcd
        import numpy.ma as ma
        import numpy as np

        if not isinstance(ids, type(None)):
            if len(ids) != len(self.p[0]):
                raise ValueError('Provided variable IDs are not matching to the dimension of X')
            vids = ids
        else:
            vids = ['f'+str(x) for x in range(1, 1+self.p.shape[1])]

        fig, ax = plt.subplots(tight_layout=True)
        ax.bar(vids, np.squeeze(self.p[pc-1]), align='center', color='violet')
        ax.set_ylabel('p'+str(pc))
        # ax.set_xticklabels(vids, rotation=90)
        # ax.set_xticklabels(vids, rotation=90)
        ax.tick_params('x', labelrotation=90)
        ax.set_xlabel('Feature')



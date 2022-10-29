import numpy as np
# import plotly.graph_objects as go
import plotly.io as pio
from utility import _cov_cor

class Stocsy:
    """
    Create STOCSY class

    Args:
        X: NMR matrix rank 2
        ppm: chemical shift vector rank 1
    Returns:
        class stocsy
    """
    def __init__(self, X, ppm):
        self.X = X
        self.ppm = ppm

    def trace(self, d, shift=[0, 10], interactive=False, spectra=True):
        """
        Perform STOCSY analysis
        Args:
            d: Driver peak position (ppm)
            shift: Chemical shift range as list of length two
            interactive: boolean, True for plotly, False for plotnine
        Returns:
            graphics object
        """
        shift = np.sort(shift)
        idx = np.argmin(np.abs(self.ppm - d))
        y = np.reshape(self.X[:, idx], (np.shape(self.X)[0], 1))
        xcov, xcor = _cov_cor(y, self.X)

        if interactive:
            import plotly.io as pio
            import plotly.graph_objects as go
            pio.renderers.default = "browser"
            idx_ppm = np.where((self.ppm >= shift[0]) & (self.ppm <= shift[1]))[0]
            t = xcor[0][idx_ppm]
            x, y = self.ppm[idx_ppm], xcov[0][idx_ppm]

            fig = go.Figure(data=go.Scatter(x=x, y=y, mode='markers+lines',
                                            marker={'color': t, 'colorscale': 'Rainbow', 'size': 5,
                                                    'colorbar': dict(title="|r|")}, line={'color': 'black'}))
            fig.update_xaxes(autorange="reversed")
            fig.show()
            return fig

        else:
            from matplotlib.collections import LineCollection
            # from matplotlib.colors import ListedColormap, BoundaryNorm
            import matplotlib.pyplot as plt
            x = np.squeeze(self.ppm)
            y = np.squeeze(xcov)
            z = np.abs(np.squeeze(xcor))
            xsub = self.X

            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            # Create a continuous norm to map from data points to colors
            norm = plt.Normalize(z.min(), z.max())
            lc = LineCollection(segments, cmap='rainbow', norm=norm)
            # Set the values used for color mapping
            lc.set_array(z)
            lc.set_linewidth(2)

            dd = (x.max() - x.min()) / 30
            if spectra:
                fig, axs = plt.subplots(2, 1, sharex=True)
                line = axs[0].add_collection(lc)
                fig.colorbar(line, ax=axs)
                axs[0].set_xlim(x.max() + dd, (x.min() - dd))
                axs[0].set_ylim(y.min() * 1.1, y.max() * 1.1)
                axs[0].vlines(d, ymin=(y.min() * 1.1), ymax=(y.max() * 1.1), linestyles='dotted', label='driver')
                axs[1].plot(x, xsub.T, c='black', linewidth=0.3)
                axs[1].vlines(d, ymin=(xsub.min() * 1.1), ymax=(xsub.max() * 1.1), linestyles='dotted', label='driver',
                              colors='red')
            else:
                fig, axs = plt.subplots(1, 1)
                line = axs.add_collection(lc)
                fig.colorbar(line, ax=axs)
                axs.set_xlim(x.max() + dd, (x.min() - dd))
                axs.set_ylim(y.min() * 1.1, y.max() * 1.1)
                axs.vlines(d, ymin=(y.min() * 1.1), ymax=(y.max() * 1.1), linestyles='dotted', label='driver')

            return (axs, fig)
            #
            # dd=pd.DataFrame({'ppm':np.squeeze(self.ppm), 'cov':np.squeeze(xcov), 'cor':np.abs(np.squeeze(xcor))})
            # idx_ppm=np.where((dd.ppm>=shift[0]) & (dd.ppm <= shift[1]))[0]
            # dd=dd.iloc[idx_ppm]
            # dd['fac']='STOCSY: d='+str(d)+' ppm'+', n='+ str(self.X.shape[0])
            #
            # rainbow=["#0066FF",  "#00FF66",  "#CCFF00", "#FF0000", "#CC00FF"]
            #
            # g=pn.ggplot(dd, pn.aes(x='ppm', y='cov', color='cor'))+pn.geom_line()+pn.scale_x_reverse()+pn.scale_colour_gradientn(colors=rainbow, limits=[0,1])+pn.theme_bw()+pn.labs(color='r(X,d)')+ pn.scale_y_continuous(labels=scientific_format(digits=2))+pn.facet_wrap('~fac')
            #
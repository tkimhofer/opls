
import numpy as np


class comp_data:
    # X: spectral data, Y = Outcome (cat/cont), x/y centering, x/y scaling (uv/pareto/none)
    def __init__(self, X, Y, x_center=True, x_stype='uv', y_center=False, y_stype=None, ):

        self.component_type = None
        if Y.ndim == 1:
            Y = Y[..., np.newaxis]

        if X.ndim != 2:
            raise ValueError('Check X dimensions')

        self.x_n, self.x_m = X.shape
        self.y_n, self.y_m = Y.shape

        if self.y_m != 1:
            raise ValueError('OPLS currently does not supports multy-column Y')

        if self.y_n != self.x_n:
            raise ValueError('X and Y dimensions do not match')

        if Y.dtype.kind.lower() not in 'fibuO':
            print(Y.shape)
            print(Y.dtype.kind.lower())
            raise ValueError('Unknown Y data type')

        if Y.dtype.kind.lower() in 'f':
            self.Y = Y
            self.ytype = 'R'
            self.Ylev = None
        if Y.dtype.kind.lower() in 'i':
            self.Y = Y.astype(float)
            self.ytype = 'R'
            self.Ylev = None
        if Y.dtype.kind.lower() in 'bu0':
            self.ytype = 'DA'
            self.Ylev, self.Y = np.unique(Y, return_inverse=True)
            self.Y = self.Y[..., np.newaxis]
            self.Y = self.Y.astype(float)

        if X.dtype.kind.lower() not in 'if':
            raise ValueError('Unknown X data type')

        if X.dtype.kind.lower() in 'i':
            self.X = self.X.astype(float)
        else:
            self.X = X

        self.x_center = x_center
        self.x_stype = x_stype
        self.y_center = y_center
        self.y_stype = y_stype
        self.x_mean, self.x_sc, self.Xsc = self.center_scale(self.X, self.x_center, self.x_stype)
        self.y_mean, self.y_sc, self.Ysc = self.center_scale(self.Y, self.y_center, self.y_stype)

        self.x_tss = np.sum((self.Xsc - np.mean(self.Xsc, 0)) ** 2)
        self.y_tss = np.sum((self.Ysc - np.mean(self.Ysc, 0)) ** 2)

        self.t = None
        self.p = None
        self.b = None
        self.w = None
        self.c = None
        self.e = None
        self.R = None

        self.to = None
        self.po = None
        self.wo = None
        self.Xo = None
        self.eo = None
        self.Xo = None

        self.x_new = None
        self.t_new = None
        self.to_new = None
        self.x_res_new = None
        self.y_hat = None

        self.y_cv = None
        self.t_cv = None
        self.to_cv = None
        self.p_cv = None

        self.r2x = None
        self.r2x_orth = None
        self.r2x_cv = None
        self.r2y = None
        self.r2y_cv = None

    def center_scale(self, var, cent, stype):
        import numpy as np

        if cent:
            mean = np.mean(var, 0, keepdims=True)
            var = var - mean
        else:
            mean = np.zeros(var.shape[1])

        if stype == 'uv':
            sc = np.std(var, 0, keepdims=True)
            var = var / sc

        if stype == 'pareto':
            sc = np.sqrt(np.std(var, 0, keepdims=True))
            var = var / sc

        if isinstance(stype, type(None)):
            sc = np.zeros(var.shape[0])

        return (mean, sc, var)

    # TODO: remove zero /null entries prior or after scaling/centering
    def handle_nas_zeros(self):
        pass


class nipals(comp_data):
    def __init__(self, Xsc, y, x_center, x_stype, y_center, y_stype, eps=1e-10):
        super().__init__(Xsc, y, x_center=x_center, x_stype=x_stype, y_center=y_center, y_stype=y_stype)
        self.eps = eps
        self.component_type = 'nipals'
        self.comp_nipals()
        self.X_residual()

    def comp_nipals(self):
        import numpy as np
        u = self.Ysc
        e = 1
        while e > self.eps:
            w_t = (u.T @ self.Xsc / (u.T @ u))
            w_t = w_t / np.linalg.norm(w_t)
            t = (self.Xsc @ w_t.T) / (w_t @ w_t.T)
            c_t = (t.T @ u) / (t.T @ t)  # aka w_y
            u_upd = (u @ c_t.T) / (c_t @ c_t.T)  # aka t_y
            e = np.linalg.norm(u_upd - u) / np.linalg.norm(u_upd)
            u = u_upd

        p_t = (t.T @ self.Xsc) / (t.T @ t)
        p_yt = (u.T @ self.Ysc) / (u.T @ u)
        self.t = t
        self.p = p_t.T
        self.b = p_yt.T
        self.w = w_t.T
        self.c = c_t.T
        self.e = e

    def X_residual(self):
        self.R = self.Xsc - (self.t @ self.p.T)


class orth_component(nipals):
    def __init__(self, Xsc, y, x_center, x_stype, y_center, y_stype, eps=1e-10):
        super().__init__(Xsc, y, x_center=x_center, x_stype=x_stype, y_center=y_center, y_stype=y_stype)
        self.eps = eps
        self.component_type = 'orthogonal'
        self.comp_orth()
        self.comp_nipals()

    def comp_orth(self):
        w_o = self.p - (((self.w.T @ self.p) / (self.w.T @ self.w)) * self.w)
        w_o = w_o / np.linalg.norm(w_o)
        t_o = (self.Xsc @ w_o) / (w_o.T @ w_o)
        p_o_t = t_o.T @ self.Xsc / (t_o.T @ t_o)

        self.wo = w_o
        self.to = t_o
        self.po = p_o_t.T
        self.Xo = t_o @ p_o_t
        self.Xsc = self.Xsc - self.Xo


# define object that combines pls and opls
# component class is composed of pls or opls
class component:
    def __init__(self, X, Y, ctype, x_center=True, x_stype='uv', y_center=True, y_stype=None, eps=1e-10, x_new=None,
                 ysc_new=None, tss_x=None, tss_y=None):
        self.ctype = ctype
        self.ysc_new = ysc_new

        if self.ctype == 'pls':
            self.comp = nipals(X, Y, x_center, x_stype, y_center, y_stype, eps)
        if self.ctype == 'opls':
            self.comp = orth_component(X, Y, x_center, x_stype, y_center, y_stype, eps)
            self.comp.r2x_orth = self.r2((self.comp.to @ self.comp.po.T), tss_x=tss_x)  # ss in x that is Y-orthogonal

        # r2x of this component using tss_x
        # if not isinstance(tss_x, type(None)):
        self.comp.r2x = self.r2((self.comp.t @ self.comp.p.T), tss_x=tss_x)
        # else:
        #     self.comp.r2x = self.r2((self.comp.t @ self.comp.p.T), self.comp.Xsc)

        # predictions using same samples as mode was created with
        y_pred_mc_sc = self.comp.b * self.comp.t
        self.comp.r2y = self.r2(np.squeeze(y_pred_mc_sc), tss_x=tss_y)
        # new data predictions
        if not isinstance(x_new, type(None)):
            self.comp.x_new = x_new
            self.predictions()
            self.comp.r2y_cv = self.r2(np.squeeze(self.comp.y_hat), np.squeeze(self.ysc_new))
            self.comp.r2x_cv = self.r2((self.comp.t_new @ self.comp.p.T), self.comp.x_new)

    def predictions(self):
        # center & scaling
        if self.comp.x_center:
            self.comp.x_new = (self.comp.x_new - self.comp.x_mean)

        if not isinstance(self.comp.x_stype, type(None)):
            self.comp.x_new = self.comp.x_new / self.comp.x_sc

        if self.ctype == 'opls':
            self.comp.to_new = (self.comp.x_new @ self.comp.wo) / (self.comp.wo.T @ self.comp.wo)
            self.comp.x_res_new = self.comp.x_new - (self.comp.to_new @ self.comp.po.T)
            self.comp.t_new = self.comp.x_res_new @ self.comp.w
        if self.ctype == 'pls':
            self.comp.t_new = self.comp.x_new @ self.comp.w

        # predict y / T
        y_hat_mc_sc = self.comp.b * self.comp.t_new
        # backscale y_hat to derive original Y values
        if not isinstance(self.comp.y_stype, type(None)):
            y_hat_mc_sc = y_hat_mc_sc * self.comp.y_sc
        self.comp.y_hat = y_hat_mc_sc + self.comp.y_mean
        # self.pred = {'t_orth': t_orth, 'x_res': x_res, 't_pred': t_pred, 'y_hat': y_hat}

    def r2(self, x_hat, x=0, tss_x=None):
        if not isinstance(tss_x, type(None)):
            return 1 - (np.sum((x - x_hat) ** 2) / tss_x)
        else:
            return 1 - (np.sum((x - x_hat) ** 2) / np.sum((x - np.mean(x_hat)) ** 2))

    def ssq(self, x):
        return np.sum((x - np.mean(x, 0, keepdims=True)) ** 2)


class cv_sets:
    def __init__(self, n, cvtype='mc', pars={'k': 1000, 'split_train': 2 / 3}):
        self.n = n
        self.cv_type = cvtype
        self.pars = pars
        if cvtype == 'mc':
            self.idc_train, self.idc_test = self.mc_cvset()
        elif cvtype == 'k-fold':
            self.idc_train, self.idc_test = self.kfold_cvset()
        else:
            raise ValueError('check type value')

    def mc_cvset(self):
        import numpy as np
        # sample split from data.n for k times
        n_sample = round(self.n * self.pars['split_train'])

        s = lambda: np.random.choice(self.n, n_sample, replace=False)
        idc_train = [s() for _ in range(self.pars['k'])]

        s = lambda A: set(np.arange(self.n)) - set(A)
        idc_test = [list(s(A)) for A in idc_train]

        return (idc_train, idc_test)

    def kfold_cvset(self):
        import numpy as np
        import math

        if self.pars['k'] > self.n:
            raise ValueError('k-fold cv: k exceeding n')

        idc = list(range(self.pars['k'])) * math.floor(self.n / self.pars['k'])
        idc = idc + list(range(self.n - len(idc)))
        np.random.shuffle(idc)

        if len(idc) != self.n:
            raise ValueError('kfold_cv fct: unequal array shape')

        s = lambda k: [i == k for i in idc]
        idc_train = [np.where(np.invert(s(k)))[0] for k in range(self.pars['k'])]
        idc_test = [np.where(s(k))[0] for k in range(self.pars['k'])]

        return (idc_train, idc_test)


class Qcomp:

    # two cases: k-fold (easy) and mc (take mean value)
    def __init__(self, y, y_hat, ytype, prior=100):
        self.q2 = None
        self.auroc = None
        self.y = y
        self.y_hat = y_hat
        self.ytype = ytype
        self.cont = False

        if self.ytype == 'R':
            self.q2 = 1 - (np.sum((self.y - self.y_hat) ** 2) / np.sum((self.y - np.mean(self.y)) ** 2))
            if (self.q2 > 0.15) & ((self.q2 - prior) > 0.05): self.cont = True

        if self.ytype == 'DA':
            self.roc()
            if (self.auroc > 0.7) & ((self.auroc - prior) > 0.05): self.cont = True

    def confusion(self, cp):
        # two classes scenario
        y_lev = np.unique(self.y)
        y_pos = self.y == y_lev[1]
        y_neg = self.y == y_lev[0]

        yh_pos = self.y_hat >= cp
        yh_neg = self.y_hat < cp

        tpr = np.sum(yh_pos & y_pos) / np.sum(y_pos)
        tnr = np.sum(yh_neg & y_neg) / np.sum(y_neg)

        fpr = np.sum(yh_pos & y_neg) / np.sum(y_neg)
        fnr = np.sum(yh_neg & y_pos) / np.sum(y_pos)

        return (tpr, tnr, fpr, fnr)

    def roc(self):
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(self.y, self.y_hat, pos_label=np.max(self.y))
        self.auroc = metrics.auc(fpr, tpr)
        # np.array([self.confusion(cp=x) for x in cp])
        # # two class scenario
        # cp = np.unique(self.y_hat)
        # out = np.array([self.confusion(cp=x) for x in cp])
        # self.auroc = np.sum([x > y for x in out[:, 0] for y in out[:, 2]]) / out.shape[0] ** 2
        # plt.scatter(out[:, 2], out[:, 0])


class O_pls(comp_data, cv_sets):
    def __init__(self, X, Y, ctype='opls', x_center=True, x_stype='uv', y_center=True, y_stype=None, cvtype='k-fold',
                 pars={'k': 10}, eps=1e-10):
        comp_data.__init__(self, X=X, Y=Y, x_center=x_center, x_stype=x_stype, y_center=y_center, y_stype=y_stype)
        cv_sets.__init__(self, self.x_n, cvtype, pars)
        self.n_oc = 0
        self.n_pd = 0
        self.ctype = ctype  # declared in fit fct
        self.eps = eps
        print('data, cv sets done')
        self.fit(self.ctype)

    def fit(self, ctype, autostop=True, nc_=50):
        import numpy.ma as ma
        import numpy as np

        self.ctype = ctype
        self.c = []
        self.de = []
        cont = True
        nc = 0

        self.t_pred = []
        self.p_pred = []
        self.y_pred = []
        self.c_full = []

        while ((cont & autostop) and (nc < nc_)):
            print('component' + str(nc))
            print('tss_x' + str(self.x_tss))
            tpred = np.full((self.x_n, self.pars['k']), None)  # full matrix to be used in mccv and k-fold cv
            ypred = np.full((self.x_n, self.pars['k']), None)  # full matrix to be used in mccv and k-fold cv
            topred = np.full((self.x_n, self.pars['k']), None)
            ppred = np.full((self.x_m, self.pars['k']), None)

            if nc == 0:
                'start with ori data'
                Xs = self.X  # scale/center in cv functions
            else:
                if self.ctype == 'pls':
                    # print('residual data pls')
                    Xs = c_full.comp.R

                if self.ctype == 'opls':
                    # print('Xs opls')
                    Xs = c_full.comp.Xsc

            print('calculating cv components')

            for i in range(self.pars['k']):

                # print('cv set')
                xtr = Xs[self.idc_train[i]]
                ytr = self.Y[self.idc_train[i]]
                yte = self.Ysc[self.idc_test[i]]
                xte = Xs[self.idc_test[i]]

                if nc == 0:
                    # print('component cv nc = 0')
                    self.c = component(xtr, ytr, ctype=self.ctype, x_center=self.x_center, \
                                       x_stype=self.x_stype, y_center=self.y_center, y_stype=self.y_stype, eps=self.eps,
                                       x_new=xte, ysc_new=yte, tss_x=self.x_tss, tss_y=self.y_tss)
                else:
                    self.c = component(xtr, ytr, ctype=self.ctype, x_center=None, \
                                       x_stype=None, y_center=False, y_stype=None, eps=self.eps,
                                       x_new=xte, ysc_new=yte, tss_x=self.x_tss, tss_y=self.y_tss)

                # pred = predictions(c.comp, xte, self.ctype)
                tpred[self.idc_test[i], i] = np.squeeze(self.c.comp.t_new)
                topred[self.idc_test[i], i] = np.squeeze(self.c.comp.to_new)
                ypred[self.idc_test[i], i] = np.squeeze(self.c.comp.y_hat)
                ppred[:, i] = np.squeeze(self.c.comp.p)
                # print(ppred.shape)

            # cont = False
            print('determine overfitting')
            mx = ma.masked_values(ypred, None)
            mm = mx.mean(axis=1, dtype=float)

            self.y_hat = mm.data[~mm.mask]
            self.y = np.squeeze(self.Y[~mm.mask])

            # decide if next component or not
            de = Qcomp(self.y, self.y_hat, self.ytype)
            self.de.append(de)

            self.t_pred.append(tpred)
            self.y_pred.append(ypred)

            # print('appending c_full')
            # print(Xs[0:4, 0:4])
            if nc == 0:
                c_full = component(Xs, self.Ysc, ctype=self.ctype, x_center=self.x_center, \
                                   x_stype=self.x_stype, y_center=self.y_center, y_stype=self.x_stype, eps=self.eps,
                                   x_new=Xs,
                                   ysc_new=self.Ysc, tss_x=self.x_tss, tss_y=self.y_tss)
            else:
                c_full = component(Xs, self.Ysc, ctype=self.ctype, x_center=False, x_stype=None, y_center=False,
                                   y_stype=None, eps=self.eps, x_new=Xs, ysc_new=self.Ysc, tss_x=self.x_tss,
                                   tss_y=self.y_tss)

            print('r2x_c_full: ' + str(c_full.comp.r2x))
            # save cv t and cv y pred in c_full
            c_full.comp.y_cv = ypred
            c_full.comp.t_cv = tpred
            c_full.comp.p_cv = ppred
            c_full.comp.to_cv = topred
            self.c_full.append(c_full)
            print(len(self.c_full))
            # print('calculate full model')
            # if next component: calcluate model using all data
            if not de.cont:
                cont = False
            else:
                nc += 1

        self.nc = nc + 1

        # collect cv scores, loadings to visualise in
        # full model component should also contain cv t, cv to, cv p, cv po, cv y pred, y_pred, and rx2 ry2
        # TODO: calc R2X cv after components are fixed, use this to calc SE/CI for r2X of full model

    def plot_load_nmr(self, ppm, pc='p1', shift=[0, 10]):
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap, BoundaryNorm
        import matplotlib.pyplot as plt
        """
        Plot statistical reconstruction of PLS/OPLS component loadings 

        Args:
              pc: Index of components, for OPLS: 'p0' for predictive component and 'o[n]' for the n-th orthogonal component, index starts at zero
              shift: Chemical shift range (list of 2)
        Returns:
              plotting object
        """

        if len(pc) != 2:
            raise ValueError(
                'Check pc argument - string of indicators for component type (p/o) and component id, e.g. `p0` for predictive component of an OPLS-model')

        try:
            cid = int(pc[1])
        except:
            raise ValueError('Second string element should be a number starting from zero, indicating the component id')

        if pc[0] not in 'po':
            raise ValueError('First string element should be character [o] or [p], indicating the component type')

        if (pc in 'o') and (self.component_type != 'orthogonal'):
            raise ValueError('Check pc argument - no orthogonal model')

        if (self.component_type != 'orthogonal') & (pc in 'o') & (cid > (self.nc - 1)):
            raise ValueError('Check pc argument - component id is too high')

        if (self.component_type != 'orthogonal') & (pc in 'p') & (cid > 0):
            raise ValueError('Check pc argument - OPLS model has a single predictive component')

        # print(shift)
        shift = np.sort(shift)

        if pc[0] == 'p':
            if self.component_type == 'orthogonal':
                t = self.c_full[self.nc - 1].comp.t
            else:
                t = self.c_full[cid].comp.t
        if pc[0] == 'o':
            t = self.c_full[cid].comp.to

        idx = np.where((ppm >= shift[0]) & (ppm <= shift[1]))[0]
        xsub = self.X[:, idx]
        x = ppm[idx]

        print(xsub.shape)
        print(x.shape)

        y, z, = _cov_cor(t, xsub)
        z = np.abs(z[0])

        fig, axs = plt.subplots(2, 1, sharex=True)
        points = np.array([x, y[0]]).T.reshape(-1, 1, 2)
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

    def plot_load(self, ids=None):
        import matplotlib.pyplot as plt
        import matplotlib._color_data as mcd
        import numpy.ma as ma
        import numpy as np

        cid = len(self.c_full) - 1

        if self.cv_type == 'mc':
            def mc_estimates(x):
                t_cv = ma.masked_values(x, value=None)
                if not isinstance(type(t_cv.mask), type(bool)):
                    x_sd = []
                    for i in range(t_cv.shape[0]):
                        x_sd.append(1 / np.std(t_cv[i][~t_cv.mask[i]], dtype=float))
                else:
                    x_sd = np.std(x.astype(float), 1)
                x = t_cv.mean(axis=1, dtype=float)
                return (x, np.array(x_sd))

            x, x_sd = mc_estimates(self.c_full[cid].comp.p_cv)

        if self.cv_type == 'k-fold':
            def mc_estimates(x):
                t_cv = ma.masked_values(x, value=None)
                if not isinstance(type(t_cv.mask), type(bool)):
                    x_sd = []
                    for i in range(t_cv.shape[0]):
                        x_sd.append(1 / np.std(t_cv[i][~t_cv.mask[i]], dtype=float))
                else:
                    x_sd = np.std(x.astype(float), 1)
                x = t_cv.mean(axis=1, dtype=float)
                return (x, np.array(x_sd))

            x_sd = np.std(self.c_full[cid].comp.p_cv.astype(float), axis=1)
            x = self.c_full[cid].comp.p

        ci = 0.95
        x_ci = (x_sd / np.sqrt(self.pars['k'])) * ci
        if not isinstance(ids, type(None)):
            if len(ids) != len(x_sd):
                raise ValueError('Provided variable IDs are not matching to the dimension of X')
            vids = ids
        else:
            vids = np.arange(self.X.shape[1])

        # plt.figure()
        # plt.bar(vids, np.squeeze(x), yerr=x_ci, align='center', color='violet')
        # plt.ylabel('p')
        # vids = ['f' + str(x) for x in range(1, 1 + self.p.shape[1])]

        fig, ax = plt.subplots(tight_layout=True)
        ax.bar(vids, np.squeeze(x), yerr=x_ci, align='center', color='violet')
        ax.set_ylabel('p' )
        ax.tick_params('x', labelrotation=90)
        ax.set_xlabel('Feature')

    def plot_scores(self):
        # currently only impolemented for opls not pls
        import matplotlib.pyplot as plt
        import matplotlib._color_data as mcd
        import numpy.ma as ma
        import numpy as np

        #
        cid = len(self.c_full) - 1

        def ellipse(x, y, alpha=0.95):
            from scipy.stats import chi2
            theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), np.linspace(np.pi, -np.pi, 50)))
            circle = np.array((np.cos(theta), np.sin(theta)))
            cov = np.cov(x, y)
            ed = np.sqrt(chi2.ppf(alpha, 2))
            ell = circle.T.dot(np.linalg.cholesky(cov).T * ed)
            a, b = np.max(ell[:, 0]), np.max(ell[:, 1])  # 95% ellipse bounds
            t = np.linspace(0, 2 * np.pi, len(x))

            el_x = a * np.cos(t)
            el_y = b * np.sin(t)
            return el_x, el_y

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "ridx: {}".format(" ".join([str(n) for n in ind["ind"]]))
            annot.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            # annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        if self.cv_type == 'mc':
            # get cv-scores
            def mc_estimates(x):
                t_cv = ma.masked_values(x, None)
                x = t_cv.mean(axis=1, dtype=float)
                x_sd = []
                for i in range(t_cv.shape[0]):
                    x_sd.append(1 / np.std(t_cv[i][~t_cv.mask[i]], dtype=float))
                return (x, np.array(x_sd))

            x, x_sd = mc_estimates(self.c_full[cid].comp.t_cv)
            y, y_sd = mc_estimates(self.c_full[cid].comp.to_cv)

        if self.cv_type == 'k-fold':
            x_cv = ma.masked_values(self.c_full[cid].comp.t_cv, None)
            x = x_cv.data[~x_cv.mask]
            y_cv = ma.masked_values(self.c_full[cid].comp.to_cv, None)
            y = y_cv.data[~y_cv.mask]

        # calculate HT2 ellipse
        el = ellipse(np.array(x).astype(float), np.array(y).astype(float), alpha=0.95)

        fig, ax = plt.subplots()
        ax.axhline(0, color='black', linewidth=0.3)
        ax.axvline(0, color='black', linewidth=0.3)
        ax.plot(el[0], el[1], color='gray', linewidth=0.5, linestyle='dotted')
        ax.set_xlabel('t_pred')  # +' ('+str(self.r2[pc[0]-1])+'%)')
        ax.set_ylabel('t_orth')  # +' ('+str(self.r2[pc[1]-1])+'%)')

        # color specification:
        # categorical data
        if self.ytype == 'DA':
            cdict = dict(zip(self.Ylev, list(mcd.TABLEAU_COLORS)[0:len(self.Ylev)]))
            plab = np.array([(cdict[self.Ylev[int(i)]], self.Ylev[int(i)]) for i in self.Y])
            if self.cv_type == 'mc':
                for i in self.Ylev:
                    idx = np.where(plab[:, 1] == i)[0]
                    ax.scatter(x[idx], y[idx], c=plab[idx, 0][0], label=i, s=x_sd[idx])
                ax.legend()
            if self.cv_type == 'k-fold':
                for i in self.Ylev:
                    idx = np.where(plab[:, 1] == i)[0]
                    ax.scatter(x[idx], y[idx], c=plab[idx, 0][0], label=i)
                # sc = ax.scatter(x, y, c=plab[:, 1], cmap=cmap)
                ax.legend()

                # ax.legend(plab[1])
                # ax.legend()

        if self.ytype == 'R':
            cm = plt.cm.get_cmap('RdYlBu')
            if self.cv_type == 'mc':
                sc = plt.scatter(x, y, c=np.squeeze(self.Y), s=x_sd, cmap=cm)
            if self.cv_type == 'k-fold':
                sc = plt.scatter(x, y, c=np.squeeze(self.Y), cmap=cm)
            plt.colorbar(sc)
        # annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
        #                         bbox=dict(boxstyle="round", fc="w"),
        #                         arrowprops=dict(arrowstyle="->"))
        # annot.set_visible(False)
        # fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
        # return (fig, ax, sc)
        return (fig, ax)

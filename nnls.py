import numpy as np

def nnlsq(X, Y, eps=1e-10):
    """
    Non-negative constrained least squares regression: Xc=Y

    Solve ||Xb-Y||2A wrt b>=0.  Algorithm extracted from `Fast Non-negativity-constrained Least Squares Algorithm`
                                (Rasmus Bro, Sijmen De Jong, Journal of Chemometrics, 1997)

    Args:
          X (np.array, rank 2) - independent variables in column format (n x m)
          Y ( np.array, rank 1 or 2) - dependent variable, if rank 1: conversion to rank 2 in column format (n x 1)
          eps -  stopping criterion of lagrange being > 0
    Returns:
          c (np.array, rank 1): Non-negative regression coefficients
    """

    if Y.ndim == 1:
        Y = Y[..., np.newaxis]

    A = X
    P = []  # positive coef
    R = np.arange(X.shape[1]).tolist()  # residual coef
    x = np.zeros(X.shape[1])[..., np.newaxis]  # init

    w = A.T @ (Y - (A @ x))  # weighting (ss)
    c = 1
    while ((len(R) > 0) | (np.max(w) > eps)):
        print('outer')
        print(len(R))
        print(np.max(w))
        j = R[np.argmax(w[R])]
        P.append(j)  # get most positive coef, place in passive set
        R.remove(j)  # remove coef, remove from active set

        AP = A[:, P]  # update A with set P
        s = np.zeros((len(x), 1))
        s[P, 0] = np.squeeze((np.linalg.inv(AP.T @ AP) @ AP.T) @ Y)  # update coef for passive set P

        # in case unconstrained coefs turn neg: reduce coef magnitude or remove from active set
        while np.min(s[P, 0]) <= 0:
            print('s')
            idc = (np.where(s[P, 0] <= 0)[0]).tolist()
            xx = np.array([x[P, 0][i] for i in idc])
            ss = np.array([s[P, 0][i] for i in idc])
            alpha = - np.min(xx / (xx - ss))
            x = x + alpha * (s - x)
            [R.append(P[i]) for i in (np.where(x[P] <= 0)[0])]
            [P.remove(P[i]) for i in (np.where(x[P] <= 0)[0])]

            AP = A[:, P]  # update A and s
            s[P] = (np.linalg.inv(AP.T @ AP) @ AP.T) @ Y
            s[R] = 0

        x = s
        w = A.T @ (Y - (A @ x))
        c += 1

    return np.squeeze(x)
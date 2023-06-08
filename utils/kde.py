"""
source from: https://gist.github.com/jayaram-r/ae4a613f2525499ea378f4f8bd4774b9#file-kernel_density_estimation-py

Kernel density estimation with bandwidth selection. The kernel bandwidth is selected using cross-validation to
maximize the cross-validated log-likelihood.
Some of the constants for the scikit learn implementation are set based on:
https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def helper_bw_search(samples, kernel, params, n_folds, rtol, n_jobs):
    grid = GridSearchCV(KernelDensity(kernel=kernel, rtol=rtol), params, cv=n_folds, n_jobs=n_jobs)
    grid.fit(samples)
    return grid.best_estimator_


def wrapper_kde(samples, kernel='gaussian', bandwidth=None, n_jobs=1):
    if len(samples.shape) == 1:
        samples = samples[:, np.newaxis]

    n_samp = samples.shape[0]
    if n_samp <= 1000:
        rtol = 1e-12
    elif n_samp <= 20000:
        rtol = 1e-8
    else:
        rtol = 1e-6

    if bandwidth is None:
        # Use cross-validation to find a suitable bandwidth
        n_folds = 10 if n_samp <= 500 else 5
        if n_samp <= 1000:
            n_bandwidth = 20
        elif n_samp <= 5000:
            n_bandwidth = 15
        else:
            n_bandwidth = 10

        params = {'bandwidth': np.logspace(-4, -1, num=n_bandwidth)}
        model_kde = helper_bw_search(samples, kernel, params, n_folds, rtol, n_jobs)
        bw_kde = model_kde.bandwidth
        if bw_kde < 0.0001001:
            # If the best bandwidth found is close to the lower end, then decrease the search set
            params = {'bandwidth': np.logspace(-5, np.log10(bw_kde), num=5)}
            model_kde = helper_bw_search(samples, kernel, params, n_folds, rtol, n_jobs)
            bw_kde = model_kde.bandwidth
        elif bw_kde > 0.099:
            # If the best bandwidth found is close to the upper end, then increase the search set
            params = {'bandwidth': np.logspace(np.log10(bw_kde), 1, num=5)}
            model_kde = helper_bw_search(samples, kernel, params, n_folds, rtol, n_jobs)
            bw_kde = model_kde.bandwidth

        print("Best KDE bandwidth = {:.6f}".format(bw_kde))
    else:
        model_kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, rtol=rtol).fit(samples)

    return model_kde

from scipy.stats import combine_pvalues #, ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils.kde import wrapper_kde
from utils.plot_utils import plot_stats

import tensorflow as tf


def hellinger(p, q):
    # code snippet from https://jamesmccaffrey.wordpress.com/2021/06/07/the-hellinger-distance-between-two-probability-distributions-using-python/
    # measuring Hellinger distance between p and q
    # p and q are np array probability distributions
    n = len(p)
    _sum = 0.0
    for i in range(n):
        _sum += (np.sqrt(p[i]) - np.sqrt(q[i]))**2
    result = (1.0 / np.sqrt(2.0)) * np.sqrt(_sum)
    return result


def density_estimation(X_in, X_out, modeling_type='kde', bw_in=None, bw_out=None):
    """
    :param modeling_type: kde (kernel density estimation) or gm (gaussian mixture)
    """
    if modeling_type == 'kde':
        print('estimating score distribution of ID data.......')
        model_in = wrapper_kde(X_in, bandwidth=bw_in) # estimates Pr(s|ID)
        print('estimating score distribution of OOD data.......')
        model_out = wrapper_kde(X_out, bandwidth=bw_out) # estimate Pr(s|OOD)
        #log_dens = model.score_samples(X)
        #dens = np.exp(log_dens)
        
    return model_in, model_out

def bayes_posterior(s_in, s_out, modeling_type='kde', bw_in=None, bw_out=None):
    """
    Estimate the Bayes posterior distribution: Pr(ID|s) -- the lower, the more likely to be OOD
    :param s_in: OOD scores of ID data
    :param s_out: OOD scores of OOD data 
    """
    # for train set
    num_in = len(s_in) # AwA train: 29841 images
    num_out = len(s_out) # MSCOCO train: 10452 images

    prior_in = num_in / (num_in + num_out) # Pr(ID)
    prior_out = 1 - prior_in               # Pr(OOD)

    model_in, model_out = density_estimation(s_in[:,None], s_out[:,None], modeling_type, bw_in, bw_out)

    s_all = np.r_[s_in, s_out]
    dens_in = np.exp(model_in.score_samples(s_all[:,None])) # Pr(s|ID)
    dens_out = np.exp(model_out.score_samples(s_all[:,None])) # Pr(s|OOD)
    post_in = prior_in*dens_in / (prior_in*dens_in + prior_out*dens_out) # Pr(ID|s) = Pr(ID)*Pr(s|ID)/(Pr(ID)*Pr(s|ID)+Pr(OOD)*Pr(s|OOD))
    post_out = 1 - post_in # Pr(OOD|s)
    posterior = np.c_[post_out, post_in]
    print(f'dens_in: mean {np.mean(dens_in)} | std {np.std(dens_in)} | min {np.min(dens_in)} | max {np.max(dens_in)}')
    print(f'post_in: mean {np.mean(post_in)} | std {np.std(post_in)} | min {np.min(post_in)} | max {np.max(post_in)}')
    
    #### plot estimated distributions
    idx_in, idx_out = np.argsort(s_in), np.argsort(s_out)
    plt.close()
    plt.fill_between(s_in[idx_in], dens_in[:num_in][idx_in], alpha=0.5, color='b')
    plt.fill_between(s_out[idx_out], dens_out[num_in:][idx_out], alpha=0.5, color='r')
    plt.legend(['in-distribution (AwA)', 'out-of-distribution (MSCOCO)'])
    plt.title('Estimated Distribution')
    plt.savefig('results/OOD_baselines/{}_MSP_distribution.jpg'.format(modeling_type))
    
    plt.close()
    plt.fill_between(s_in[idx_in], dens_in[:num_in][idx_in], alpha=0.5, color='b')
    plt.fill_between(s_out[idx_out], dens_in[num_in:][idx_out], alpha=0.5, color='r')
    plt.legend(['in-distribution (AwA)', 'out-of-distribution (MSCOCO)'])
    plt.title('Estimated Distribution')
    plt.savefig('results/OOD_baselines/{}_MSP_distribution_IN.jpg'.format(modeling_type))

    plt.close()
    plt.fill_between(s_in[idx_in], dens_out[:num_in][idx_in], alpha=0.5, color='b')
    plt.fill_between(s_out[idx_out], dens_out[num_in:][idx_out], alpha=0.5, color='r')
    plt.legend(['in-distribution (AwA)', 'out-of-distribution (MSCOCO)'])
    plt.title('Estimated Distribution')
    plt.savefig('results/OOD_baselines/{}_MSP_distribution_OOD.jpg'.format(modeling_type))

    plt.close()
    plt.fill_between(s_in[idx_in], post_in[:num_in][idx_in], alpha=0.5, color='b')
    plt.fill_between(s_out[idx_out], post_in[num_in:][idx_out], alpha=0.5, color='r')
    plt.legend(['in-distribution (AwA)', 'out-of-distribution (MSCOCO)'])
    plt.title('Bayesian Posterior Distribution')
    plt.savefig('results/OOD_baselines/{}_MSP_posterior.jpg'.format(modeling_type))
    
    plt.close()
    plt.fill_between(s_in[idx_in], post_in[:num_in][idx_in], alpha=0.5, color='b')
    plt.fill_between(s_out[idx_out], post_out[num_in:][idx_out], alpha=0.5, color='r')
    plt.legend(['in-distribution (AwA)', 'out-of-distribution (MSCOCO)'])
    plt.title('Bayesian Posterior Distribution')
    plt.savefig('results/OOD_baselines/{}_MSP_posterior_IDvsOOD.jpg'.format(modeling_type))
    return posterior

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    # p = np.asarray(p, dtype=np.float)
    # q = np.asarray(q, dtype=np.float)

    div = np.where(p*q > 0, p * np.log(p / q), 0)
    return div

def mahalanobis(x,data):
    x_minus_mu = x - np.mean(data,axis=0)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal #.diagonal()

def aggregate_pval(pvalues, method='fisher'):
    """pvalues: dim=(n_concepts,), to be aggregated across columns"""

    if method == 'fisher':
        _, pval_agg = combine_pvalues(pvalues, method)
        # pval_agg = -2 * np.sum(np.log(pvalues))
    elif method == 'harmonic_mean':
        n_pvals = len(pvalues)
        weights = (1. / n_pvals) * np.ones(n_pvals)
        pval_agg = np.sum(weights) / np.sum(weights/pvalues)

    return pval_agg

def compute_pval(data_null, data_obs):
    """
    :param data_null: dim=(n_profile, num_concepts) 
    :param data_obs: dim=(num_concepts,)
    """
    
    """
    num_concepts = data_null.shape[1]
    
    #pval = np.ones((data_test.shape[0],1), dtype=np.float64)
    pval = []
    for c in range(num_concepts):
        _, _pval = ttest_1samp(data_null[:,c], data_test[c]) #data_test[:,c])
        #pval = np.minimum(pval, _pval)
        pval.append(_pval)

    #return pval, np.sum(np.array(pval) > 1e-50) 
    return pval, np.sum(np.square(np.sort(pval)[:5]))
    #return np.sum(pval)
    """
    eps = 1e-16

    n_data, n_concept = data_null.shape
    pval = np.zeros(data_obs.shape)
    for i in range(n_concept):
        pval[i] = 2 * min(np.sum(data_null[:,i] <= data_obs[i]), np.sum(data_null[:,i] >= data_obs[i])) / n_data
    
    # TODO: allow bootstrap option and compare the pval estimation results
    # reference: https://github.com/jayaram-r/adversarial-detection/blob/94fd0881a3eef179e66301629c9a5e348ce46bd1/expts/detectors/pvalue_estimation.py#L33

    pval[pval < eps] = eps
    return pval

def FLD(x_in, x_out, optimal=False):
    """
    compute Fisher Linear Discriminant (FLD) to measure the separability between x_in vs x_out
    :param x_in: ID concept scores with dim=(N_in, N_concepts) if optimal=False, else ID features
    :param x_out: OOD concept scores with dim=(N_out, N_concepts) if optimal=False, else OOD features
    :param optimal: whether to find the optimal projection that maximizes the separability
    return fld: each element is FLD score w.r.t each concept vector, dim=(N_concepts,)
    """

    if optimal:
        N_in, N_h, N_w, N_feat = x_in.shape
        N_out, _, _, _ = x_out.shape
        X = np.r_[x_in.reshape(-1,N_feat), x_out.reshape(-1,N_feat)]
        y = np.r_[np.ones(N_in*N_h*N_w), np.zeros(N_out*N_h*N_w)]

        """
        height, width = X.shape
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        n_components = 1
        scatter_t = np.cov(X.T)*(height - 1)
        print(scatter_t.shape)
        scatter_w = 0
        for i in range(num_classes):
            class_items = np.flatnonzero(y == unique_classes[i])
            scatter_w = scatter_w + np.cov(X[class_items].T) * (len(class_items)-1)

        scatter_b = scatter_t - scatter_w
        _, eig_vectors = np.linalg.eigh(np.linalg.pinv(scatter_w).dot(scatter_b))
        print(eig_vectors.shape)
        pc = X.dot(eig_vectors[:,::-1][:,:n_components])
        """
        clf = LinearDiscriminantAnalysis(n_components=1)
        clf.fit(X, y)
        pc = clf.transform(X)
        #print(pc.shape)
        pc_in = pc[y == 1].reshape(N_in,N_h,N_w,-1)
        pc_out = pc[y == 0].reshape(N_out,N_h,N_w,-1)
        #print(pc_in.shape)
        #print(pc_out.shape)

        pc_max = tf.math.reduce_max(pc_in, axis=(1,2))
        pc_max_abs = tf.math.reduce_max(tf.abs(pc_in), axis=(1,2))
        pc_in = tf.where(pc_max == pc_max_abs, pc_max, -pc_max_abs).numpy() # dim=(N_in, N_concept)
        pc_max = tf.math.reduce_max(pc_out, axis=(1,2))
        pc_max_abs = tf.math.reduce_max(tf.abs(pc_out), axis=(1,2))
        pc_out = tf.where(pc_max == pc_max_abs, pc_max, -pc_max_abs).numpy() #dim=(N_out, N_concept)
        

        mu_in = np.mean(pc_in, axis=0, keepdims=True)
        mu_out = np.mean(pc_out, axis=0, keepdims=True)
        s_in = np.sum((pc_in - mu_in)**2, axis=0)
        s_out = np.sum((pc_out - mu_out)**2, axis=0)
        #fld = np.mean((mu_in-mu_out)**2/(s_in+s_out))
        fld = (mu_in-mu_out)**2/(s_in+s_out)[0]
        """
        x_in, x_out = x_in.T, x_out.T
        #x_in = x_in/(np.linalg.norm(x_in,axis=0,keepdims=True)+1e-9)
        #x_out = x_out/(np.linalg.norm(x_out,axis=0,keepdims=True)+1e-9)
        N_in = x_in.shape[1]
        N_out = x_out.shape[1]
    
        mu_in = np.mean(x_in, axis=1, keepdims=True)
        mu_out = np.mean(x_out, axis=1, keepdims=True)
        S_w = N_in*np.cov(x_in, bias=True) + N_out*np.cov(x_out, bias=True)
        S_b = (mu_in-mu_out)*(mu_in-mu_out).T
        v = np.matmul(np.linalg.inv(S_w), (mu_in - mu_out)) # found a vector that make the projected values to have maximum separability
        v /= (np.linalg.norm(v)+1e-9) #normalize
        fld = np.matmul(np.matmul(v.T,S_b),v)/np.matmul(np.matmul(v.T,S_w),v)
        fld = fld[0,0]
        print(fld)
        """

    else:
        mu_in = np.mean(x_in, axis=0, keepdims=True)
        mu_out = np.mean(x_out, axis=0, keepdims=True)
        s_in = np.sum((x_in - mu_in)**2, axis=0)
        s_out = np.sum((x_out - mu_out)**2, axis=0)
        #fld = np.mean((mu_in-mu_out)**2/(s_in+s_out))
        fld = (mu_in-mu_out)**2/(s_in+s_out)[0]
    
    return fld



def multivar_separa(x_in, x_out):
    """
    compute separability using between-class and inter-class scatters
    :param x_in: dim=(N_in, data_dim)
    :param x_out: dim=(N_out, data_dim)
    :return: multivariate separability in tensor
    """
    mu_in = tf.transpose(tf.reduce_mean(x_in, axis=0, keepdims=True)) # dim=(num_concept,1)
    mu_out = tf.transpose(tf.reduce_mean(x_out, axis=0, keepdims=True))
    Sw_in = (tf.transpose(x_in)-mu_in) @ tf.transpose(tf.transpose(x_in)-mu_in) # dim=(data_dim, data_dim)           
    Sw_out = (tf.transpose(x_out)-mu_out) @ tf.transpose(tf.transpose(x_out)-mu_out)
    Sw = Sw_in + Sw_out

    #Sw_inv = tf.linalg.inv(Sw)
    e, v = tf.linalg.eigh(Sw)
    Sw_inv = v @ tf.linalg.diag(1/e) @ tf.transpose(v)
    separa = tf.transpose(mu_in-mu_out) @ Sw_inv @ (mu_in - mu_out)
    return separa[0,0]


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE=12
MEDIUM_SIZE=15
BIGGER_SIZE=20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_stats(scores_in, scores_out, savename):
    scores_in, scores_out = scores_in.reshape(-1,), scores_out.reshape(-1,)
    
    plt.figure(figsize=(12, 4))
    fig = plt.figure()
    ax = plt.subplot(111)
    sns.kdeplot(scores_in, color='blue')
    sns.kdeplot(scores_out, color='red')
    #fig = plt.gcf()
    ax.legend(['in-distribution (test)', 'out-of-distribution'], frameon=False)
    plt.tight_layout()
    fig.savefig(savename)
    plt.close()

def plot_per_class_stats(num_classes, scores_in, preds_in, scores_out, preds_out, savename):
    """
    ::param scores_in:: dim=(N_in,)
    ::param preds_in:: dim=(N_in,)
    ::param scores_out:: dim=(N_out,)
    ::param preds_out:: dim=(N_out,)
    """
    scores_in, scores_out = scores_in.reshape(-1,), scores_out.reshape(-1,)
    assert len(scores_in) == len(preds_in)
    assert len(scores_out) == len(preds_out)

    n = int(np.ceil(np.sqrt(num_classes)))
    fig, axes = plt.subplots(n-1 if n*(n-1) > num_classes else n, n, \
                            figsize=(30, 25), sharex=False, sharey=False)
    fig.suptitle('ID vs OOD Mahalanobis distance')

    for c in range(num_classes):
        idx_in = np.where(preds_in == c)[0]
        idx_out = np.where(preds_out == c)[0]
        scores_in_ = scores_in[idx_in]
        scores_out_ = scores_out[idx_out]

        axes[c//n, c%n].set_title('Class {}'.format(c))
        sns.kdeplot(scores_in_, color='blue', ax=axes[c//n, c%n])
        sns.kdeplot(scores_out_,color='red', ax=axes[c//n, c%n])
        """
        sns.histplot(profile_scores[:,i], color='grey', ax=axes[i//n, i%n])#, fit=norm, kde=False)
        sns.histplot(scores_in_[:,i], color='blue', ax=axes[i//n, i%n])#, fit=norm, kde=False)
        sns.histplot(scores_out_[:,i],color='red', ax=axes[i//n, i%n])#, fit=norm, kde=False)
        """
        axes[c//n, c%n].legend(['ID: mean({:.3f}), std({:.3f})'.format(np.mean(scores_in_), np.std(scores_in_)), \
                                'OOD: mean({:.3f}), std({:.3f})'.format(np.mean(scores_out_), np.std(scores_out_))])
    # save plot
    #fig = plt.gcf()
    #fig.legend(['in-distribution (test)', 'out-of-distribution'])
    fig.savefig(savename)
    plt.close()



def plot_score_distr(concept_dict, scores_in, preds_in, scores_out, preds_out, save_plot):
    # concept_dict has 'scores', 'mean', 'std'
    # concept_dict['scores'], scores_IN, scores_OOD: dim=(n_features, n_concepts)
    n_classes = 50 #len(concept_dict)
    n_concepts = np.shape(scores_in)[1]

    for c in range(n_classes):
        idx_in = np.where(preds_in == c)[0]
        idx_out = np.where(preds_out == c)[0]
        scores_in_ = scores_in[idx_in,:]
        scores_out_ = scores_out[idx_out,:]

        profile_scores = concept_dict[c]['scores']
        #conf_interval = scipy.stats.t.interval(confidence, np.shape(profile_scores)[0]-1, loc=np.mean(profile_scores, axis=0), scale=scipy.stats.sem(profile_scores))


        n = int(np.ceil(np.sqrt(n_concepts)))
        fig, axes = plt.subplots(n-1 if n*(n-1) > n_concepts else n, n, \
                                figsize=(30, 25), sharex=False, sharey=True)
        fig.suptitle('Class {}: distribution of {} concept scores'.format(c, n_concepts))
        for i in range(n_concepts):
            #plt.figure()
            axes[i//n, i%n].set_title('Concept {}'.format(i))

            #sns.kdeplot(profile_scores[:,i], color='grey', ax=axes[i//n, i%n])
            sns.kdeplot(scores_in_[:,i], color='blue', ax=axes[i//n, i%n])
            sns.kdeplot(scores_out_[:,i],color='red', ax=axes[i//n, i%n])
            """
            sns.histplot(profile_scores[:,i], color='grey', ax=axes[i//n, i%n])#, fit=norm, kde=False)
            sns.histplot(scores_in_[:,i], color='blue', ax=axes[i//n, i%n])#, fit=norm, kde=False)
            sns.histplot(scores_out_[:,i],color='red', ax=axes[i//n, i%n])#, fit=norm, kde=False)
            """

            """
            # draw threshold lines of confidence interval
            low = conf_interval[0][i] # left-most point of interval
            high = conf_interval[1][i] # right-most point of interval
            low = (thresh/2+0.5)*low - (thresh/2-0.5)*high
            high = (thresh/2+0.5)*high - (thresh/2-0.5)*low
            plt.axvline(low, color='k', linestyle='--')
            plt.axvline(high, color='k', linestyle='--')
            """
        # save plot
        #fig = plt.gcf()
        #fig.legend(['in-distribution (train)', 'in-distribution (test)', 'out-of-distribution'])
        fig.legend(['in-distribution (test)', 'out-of-distribution'])
        #fig.savefig("{}/scores_class{}.jpg".format(save_plot,c))
        fig.savefig("{}/scores_class{}_pval.jpg".format(save_plot,c))
        plt.close()


def plot_tsne(X_in, X_out, perplexity=50, run_pca=False, figtitle=None, savepath=None):
    """
    tSNE plots with ID and OOD data
    :param X_in: ID data (e.g. intermediate representations, concept scores, recovered representations), dim=(num_ID, ?)
    :param X_out: OOD data, dim=(num_OOD,?)
    :run_pca: whether to first reduce the dimensionality using PCA to make tSNE run faster.
    """
    num_in, num_out = X_in.shape[0], X_out.shape[0]
    n_components = 3
    
    if run_pca:
        pca_in, pca_out = PCA(n_components=50), PCA(n_components=50)
        X_in = pca_in.fit_transform(X_in.reshape(num_in,-1))
        X_out = pca_out.fit_transform(X_out.reshape(num_out,-1))

    tsne_in = TSNE(n_components, verbose=0, perplexity=perplexity, n_iter=1000, learning_rate=200)
    X_in_tsne = tsne_in.fit_transform(X_in.reshape(num_in,-1))

    tsne_out = TSNE(n_components, verbose=0, perplexity=perplexity, n_iter=1000, learning_rate=200)
    X_out_tsne = tsne_out.fit_transform(X_out.reshape(num_out,-1))

    # visualize
    if n_components==2:
        df_tsne = pd.DataFrame(np.r_[X_in_tsne, X_out_tsne], columns=['comp1', 'comp2'])
        df_tsne['label'] = ['ID']*num_in + ['OOD']*num_out
        #sns.lmplot(x='comp1', y='comp2', data=df_tsne, hue='label', fit_reg=False)
        sns.scatterplot(x='comp1', y='comp2', data=df_tsne, hue='label', alpha=0.5)
    elif n_components==3:
        plt.figure(figsize=(15,10))
        ax = plt.axes(projection = '3d')
        ax.scatter(X_in_tsne[:,0], X_in_tsne[:,1], X_in_tsne[:,2], alpha=0.5, c='r', label='ID')
        ax.scatter(X_out_tsne[:,0], X_out_tsne[:,1], X_out_tsne[:,2], alpha=0.5, c='b', label='OOD')
        ax.legend()
        
    plt.title(figtitle)
    plt.savefig(savepath)
    plt.close()



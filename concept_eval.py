"""Code to evaluate detection completeness and concept separability of learn concepts"""
## plot concept sensitivity statistics across concepts
import sys
import os
import time
import argparse
import numpy as np
import random
import pickle
import joblib
import itertools
import copy
import pandas as pd
import scipy.stats
import scipy.io as sio
from scipy.special import comb
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from scipy.stats import norm
sns.set_style("whitegrid") #darkgrid
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
SMALL_SIZE=12
MEDIUM_SIZE=15
BIGGER_SIZE=20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import ipdb

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.metrics as metrics

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import concept_model
import helper
from test_baselines import run_eval


from utils.test_utils import arg_parser, prepare_data, get_measures
from utils.test_utils import ConceptProfiles
from utils.test_utils import get_recovered_features
from utils.ood_utils import run_ood_over_batch
from utils.stat_utils import hellinger, compute_pval, bayes_posterior, FLD, multivar_separa
from utils.plot_utils import plot_stats, plot_per_class_stats, plot_score_distr
from utils import log


softmax = layers.Activation('softmax')

def remove_duplicate_concepts(topic_vec, return_mapping=False):
    # Remove one concept vector if there are two vectors where the dot product is over 0.95
    # topic_vec: dim=(dim_features, n_concepts) (2048, 70)
    # print(np.shape(topic_vec))
    n_concept = topic_vec.shape[1]
    thresh = 0.95
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)

    topic_vec_n_dot = np.transpose(topic_vec_n) @ topic_vec_n - np.eye(n_concept)
    dict_similar_topic = {}
    idx_delete = set()
    for i in range(n_concept):
        ith_redundant_concepts = [j for j in range(n_concept) if topic_vec_n_dot[i][j] >= 0.95]
        dict_similar_topic[i] = ith_redundant_concepts
        
        ith_redundant_concepts = [x for x in ith_redundant_concepts if x > i]
        idx_delete.update(ith_redundant_concepts)
    idx_delete = list(idx_delete)

    print(dict_similar_topic)
    print(idx_delete)

    topic_vec = np.delete(topic_vec, idx_delete, axis=1)


    dict_topic_mapping = {}
    count = 0
    for i in range(n_concept):
        if i in idx_delete:
            dict_topic_mapping[i] = None
        else:
            dict_topic_mapping[i] = count
            count += 1
    print('concept mapping between before/after duplicate removal......')
    print(dict_topic_mapping)
    if return_mapping:
        return topic_vec, dict_similar_topic, dict_topic_mapping
    else:
        return topic_vec, dict_similar_topic

def visualize_nn(test_loader, topic_vec, f_test, save_dir, logger, out_data=None):
    num_concept = topic_vec.shape[1]

    f_test_n = f_test/(np.linalg.norm(f_test,axis=3,keepdims=True)+1e-9)
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
    topic_prob = np.matmul(f_test_n,topic_vec_n)
    n_size = np.shape(f_test)[1]
    for i in range(num_concept):
      savepath = os.path.join(save_dir,'concept'+str(i))
      if not os.path.isdir(savepath):
        os.mkdir(savepath)

      neighbors_num = 15
      ind = np.argpartition(topic_prob[:,:,:,i].flatten(), -neighbors_num)[-neighbors_num:]
      sim_list = topic_prob[:,:,:,i].flatten()[ind]
      logger.info(f'[ID TEST: CONCEPT {i}] top-{neighbors_num} scores: {sim_list}')
      for jc,j in enumerate(ind):
        j_int = int(np.floor(j/(n_size*n_size)))
        a = int((j-j_int*(n_size*n_size))/n_size)
        b = int((j-j_int*(n_size*n_size))%n_size)
        f1 = None #savepath+'/concept_full_{}_{}.png'.format(i,jc)
        if not out_data:
            f2 = savepath+'/concept_{}_{}.png'.format(i,jc) 
        else:
            f2 = savepath+'/{}_concept_{}_{}.png'.format(out_data, i,jc)
        if sim_list[jc]>0.70:
            x_test_filename = test_loader.filepaths[j_int]
            helper.copy_save_image(x_test_filename,f1,f2,a,b)


def compute_concept_scores(topic_vec, feature, predict_model=None):
    # topic_vec: concept vectors (dim= (feature_dim, n_concepts))
    # feature: features extracted from an intermediate layer of trained model

    feature_n = tf.math.l2_normalize(feature, axis=3)
    topic_vec_n = tf.math.l2_normalize(topic_vec, axis=0)

    topic_prob = tf.matmul(feature_n, topic_vec_n) # K.dot

    prob_max = tf.math.reduce_max(topic_prob, axis=(1,2))
    prob_max_abs = tf.math.reduce_max(tf.abs(topic_prob), axis=(1,2))
    concept_scores = tf.where(prob_max == prob_max_abs, prob_max, -prob_max_abs)

    """
    ##for debugging
    n_concept = np.shape(concept_scores)[1]
    print(tf.reduce_mean(input_tensor=tf.nn.top_k(K.transpose(K.reshape(topic_prob,(-1,n_concept))),k=10,sorted=True).values))
    print(tf.reduce_mean(input_tensor=K.dot(K.transpose(K.variable(value=topic_vec_n)), K.variable(value=topic_vec_n)) - np.eye(n_concept)))
    """


    if predict_model: # in eager execution
        pred = softmax(predict_model(feature))
        #pred = tf.math.argmax(pred, axis=1)
        return concept_scores.numpy(), pred.numpy()
    else:
        return concept_scores

def prepare_profiles(feature_model, topic_vec, num_classes, args, logger):
    # profiling using validation data
    #profile_path = "{}/AwA2_train_concept_dict.pkl".format(args.result_dir)
    profile_path = "{}/AwA2_val_concept_dict.pkl".format(args.result_dir)
    if not os.path.exists(profile_path):
        logger.info("Profiling the distribution of concept scores from train set...")

        tf.random.set_seed(0)
        datagen = ImageDataGenerator(rescale=1./255.)
                                                #rotation_range=40,
                                                #width_shift_range=0.2, height_shift_range=0.2,
                                                #shear_range=0.2, zoom_range=0.2,
                                                #horizontal_flip=True)
        data_loader = datagen.flow_from_directory("/nobackup/jihye/data/Animals_with_Attributes2/val", \
                                                batch_size=350, target_size=(224,224), \
                                                class_mode='categorical', \
                                                shuffle=False)

        ConceptP = ConceptProfiles()
        ConceptP.setUp(num_classes, data_loader)
        ConceptP.prepare_concept_dict(feature_model, topic_vec)
        concept_dict = ConceptP.concept_dict

        #LOAD_DIR = 'data/Animals_with_Attributes2'
        #y_train = np.load(LOAD_DIR+'/y_train.npy')
        #y_train = np.argmax(y_train, axis=1)

        logger.info("Saving concept profiles of AwA2 train set in {}".format(profile_path))
        with open(profile_path,'wb') as f:
            pickle.dump(concept_dict, f)

    else:
        logger.info("Loading concept profiles of AwA2 train set from {}".format(profile_path))
        with open(profile_path,'rb') as f:
            concept_dict = pickle.load(f)

    return concept_dict


def compute_coherency(feature, topic_vec):
    """
    compute coherency across top-k nearest neighbors for each concept
    :param topic_vec: concept vectors, dim=(feature_dim, num_concept)
    :param feature: features extracted from an intermediate layer of trained model
    """

    # normalize
    feature_n = tf.math.l2_normalize(feature, axis=3)
    topic_vec_n = tf.math.l2_normalize(topic_vec, axis=0)
    
    topic_prob = tf.matmul(feature_n, topic_vec_n) # normalized concept scores, dim=(num_data, num_concept)
    num_concept = topic_prob.shape[1]
    coher = tf.reduce_mean(tf.nn.top_k(K.transpose(K.reshape(topic_prob,(-1,num_concept))),k=10,sorted=True).values)
    return coher.numpy()

def compute_redundancy(topic_vec):
    """
    compute similarity between concept vectors
    :param topic_vec: normalized concept vectors, dim=(dim_feat, num_concept)
    """
    num_concept = topic_vec.shape[-1]

    topic_vec_n = tf.math.l2_normalize(topic_vec, axis=0)
    redun = tf.reduce_mean(K.dot(K.transpose(topic_vec_n), topic_vec_n) - np.eye(num_concept))
    return redun.numpy()

def compute_completeness(y, yhat, yhat_recov, num_class, logger=None, label=None):
    """
    compute completeness score by Yeh et al.
    :param y: groundtruth class labels, dim=(N,)
    :param yhat: predicted class labels, dim=(N,)
    :param yhat_recov: predicted class labels using recovered features, dim=(N,).
                       If label is not None, per-class predicted labels, dim=(N',) where N' <= N
    """

    acc = np.sum(y == yhat)/len(y)
    if logger:
        logger.info(f'[ID TEST] accuracy with original features: {acc}')
    
    if label:
        acc_recov = np.sum(y[y==label] == yhat_recov)/len(yhat_recov)
        if logger:
            logger.info(f'[ID TEST] per-class accuracy with recovered features: {acc_recov}')
        acc_random = 1/num_class #0.5 #NOTE: check a_r = 0.5?
    else:
        acc_recov = np.sum(y == yhat_recov)/len(y)
        if logger:
            logger.info(f'[ID TEST] accuracy with recovered features: {acc_recov}')
        acc_random = 1/num_class
    
    # compute completeness
    completeness = (acc_recov - acc_random) / (acc - 1/num_class)
    if logger:
        logger.info(f'[ID TEST] completeness score: {completeness}')
    return completeness

def compute_detection_completeness(auroc, auroc_recov, logger=None):
    """
    compute detection completeness score
    """
    # compute completeness
    auroc_random = 1/2
    completeness = (auroc_recov - auroc_random) / (auroc - auroc_random)
    if logger:
        logger.info(f'[DETECTION] completeness score: {completeness}')
    return completeness


def compute_conceptSHAP(concept_mask, topic_vec, 
                        feat_in, feat_out, y, yhat_in, yhat_out, auroc,
                        in_loader, out_loader,
                        topic_model, feature_model, args, logger, 
                        finetune=False, labels=None):

    assert labels is not None

    num_class = 50
    num_concept = topic_vec.shape[1]

    ## modify topic model
    logger.info(f'[ConceptSHAP] using concept mask: {concept_mask}.....')
    #topic_vec_temp = np.random.rand(topic_vec.shape[0], topic_vec.shape[1]) 
    topic_vec_temp = copy.copy(topic_vec)
    topic_vec_temp[:,np.array(concept_mask)==0] = 0
    #print(topic_model.layers[0].get_weights())
    topic_model.layers[0].set_weights([topic_vec_temp])
    #print(topic_model.layers[0].get_weights())

    _, logits_in, _ = topic_model(feat_in)
    _, logits_out, _ = topic_model(feat_out)


    compl_class, compl_detect = np.array([]), np.array([])
    compl_class_2, compl_detect_2 = np.array([]), np.array([])
    for label in labels:
        # compute classification completeness
        _, logits, _ = topic_model(feat_in[np.where(y == label)[0]])
        #print(logits)
        yhat_in_recov = tf.math.argmax(logits, axis=1)
        _compl_class = compute_completeness(y, yhat_in, yhat_in_recov, num_class, logger, label)
        compl_class = np.append(compl_class, _compl_class) 
        
        # compute detection completeness
        idx_in = np.where(tf.math.argmax(logits_in, axis=1).numpy() == label)[0]
        idx_out = np.where(tf.math.argmax(logits_out, axis=1).numpy() == label)[0]
        logger.info(f'[ConceptSHAP CLASS {label}] number of ID: {len(idx_in)} | number of OOD: {len(idx_out)}')
        if len(idx_in) == 0 or len(idx_out) == 0:
            compl_detect = np.append(compl_detect, None)
            continue

        s_in = run_ood_over_batch(None, feature_model, topic_model, args, num_class, feat_in[idx_in])
        s_out = run_ood_over_batch(None, feature_model, topic_model, args, num_class, feat_out[idx_out])
        #s_in, s_out = np.random.rand(len(idx_in)), np.random.rand(len(idx_out))
        auroc_recov, aupr_in, aupr_out, fpr95, thres95 = get_measures(s_in[:,None],s_out[:,None])
        _compl_detect = compute_detection_completeness(auroc, auroc_recov, logger)
        compl_detect = np.append(compl_detect, _compl_detect) 
        logger.info(f'[ConceptSHAP CLASS {label}] auroc: {auroc_recov} | aupr_in: {aupr_in} | aupr_out: {aupr_out} | fpr95: {fpr95} | thres95: {thres95}')
        logger.info(f'[ConceptSHAP CLASS {label}] (before finetuning) classification completeness: {_compl_class} | detection completeness: {_compl_detect}')

        if finetune:
            target_size = (224, 224)
            batch_size = 200
            train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
            train_loader = train_datagen.flow_from_directory('/nobackup/jihye/data/Animals_with_Attributes2/train',
                                                    batch_size=batch_size,
                                                    target_size=target_size,
                                                    class_mode='categorical',
                                                    shuffle=True)
            datagen = ImageDataGenerator(rescale=1.0 / 255.)
            ood_loader = datagen.flow_from_directory("./data/MSCOCO",
                                                batch_size=batch_size,
                                                target_size=target_size,
                                                class_mode=None, shuffle=True)
    
            optimizer = Adam(lr=0.01)
            optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1, optimizer.beta_2, optimizer.decay]
            optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
            softmax = layers.Activation('softmax')
            #train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
            #test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        
            COEFF_CONCEPT = 10.0
        
            train_step_signature = [
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            #tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32), 
            ]
            @tf.function(input_signature=train_step_signature)
            def train_step(x_in, y_in): #, x_out=None):
                f_in = feature_model(x_in)
                f_in_n = K.l2_normalize(f_in,axis=(3))
            
                #f_out = feature_model(x_out)
                #f_out_n = K.l2_normalize(f_out,axis=(3))

                obj_terms = {} # terms in the objective function
                with tf.GradientTape() as tape:
                    f_in_recov, logits_in, topic_vec_n = topic_model(f_in, training=True)
                    pred_in = softmax(logits_in) # class prediction using concept scores
                    topic_prob_in_n = K.dot(f_in_n, topic_vec_n) # normalized concept scores

                    #_, logits_out, _ = topic_model(f_out, training=True)
                    #topic_prob_out_n = K.dot(f_out_n, topic_vec_n)

                    # baseline
                    CE_IN = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_in, pred_in))
                    loss_coherency = tf.reduce_mean(tf.nn.top_k(K.transpose(K.reshape(topic_prob_in_n,(-1,num_concept))),k=10,sorted=True).values)
                    loss_similarity = tf.reduce_mean(K.dot(K.transpose(topic_vec_n), topic_vec_n) - tf.eye(num_concept))
                    loss = CE_IN - COEFF_CONCEPT*loss_coherency + COEFF_CONCEPT*loss_similarity
                    obj_terms['[ID] CE'] = CE_IN
                    obj_terms['[ID] concept coherency'] = loss_coherency
                    obj_terms['[ID] concept similarity'] = loss_similarity

                grads = tape.gradient(loss, topic_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, topic_model.trainable_variables))
                return obj_terms

            for step, (x_in, y_in) in enumerate(train_loader):
                step += 1
                if step > 100: #len(train_loader):
                    break
                #x_out = ood_loader.next()
                obj_terms = train_step(x_in, y_in) #, x_out)
                if step % 20 == 0:
                    for term in obj_terms:
                        print(f'[STEP{step}] {term}: {obj_terms[term]}')


            #train_acc = train_acc_metric.result()
            #logger.info("Training acc over epoch: %.4f" % (float(train_acc),))

            # compute classification completeness
            _, logits, _ = topic_model(feat_in[np.where(y == label)[0]])
            #print(logits)
            yhat_in_recov = tf.math.argmax(logits, axis=1)
            _compl_class = compute_completeness(y, yhat_in, yhat_in_recov, num_class, logger, label)
            compl_class_2 = np.append(compl_class_2, _compl_class)

            # compute detection completeness
            idx_in = np.where(tf.math.argmax(logits_in, axis=1).numpy() == label)[0]
            idx_out = np.where(tf.math.argmax(logits_out, axis=1).numpy() == label)[0]
            s_in = run_ood_over_batch(None, feature_model, topic_model, args, num_class, feat_in[idx_in])
            s_out = run_ood_over_batch(None, feature_model, topic_model, args, num_class, feat_out[idx_out])
            auroc_recov, aupr_in, aupr_out, fpr95, thres95 = get_measures(s_in[:,None],s_out[:,None])
            _compl_detect = compute_detection_completeness(auroc, auroc_recov, logger)
            compl_detect_2 = np.append(compl_detect_2, _compl_detect)
            logger.info(f'[ConceptSHAP CLASS {label}] auroc: {auroc_recov} | aupr_in: {aupr_in} | aupr_out: {aupr_out} | fpr95: {fpr95} | thres95: {thres95}')
            logger.info(f'[ConceptSHAP CLASS {label}] (before finetuning) classification completeness: {_compl_class} | detection completeness: {_compl_detect}')

        logger.info('--------------------------------------------------------')

    topic_model.layers[0].set_weights([topic_vec])
    if finetune:
        return compl_class_2, compl_detect_2
    else:
        assert len(compl_class) == len(labels)
        assert len(compl_detect) == len(labels)
        return compl_class, compl_detect


def compute_separability(in_concept, out_concept, in_yhat, out_yhat, logger=None):
    # compute Multivariate Separability (global)
    separa = {'global': multivar_separa(in_concept, out_concept).numpy()}

    # compute per-class separability
    # num_classes = 50
    num_concepts = in_concept.shape[1]
    for i in range(num_classes):
        idx_in = np.where(in_yhat == i)[0]
        idx_out = np.where(out_yhat == i)[0]
        if logger:
            logger.info(f'class {i}: num IN - {len(idx_in)}, num OUT - {len(idx_out)}')

        ## explanation using groundtruth ID/OOD labels
        #sep_concept_ith = FLD(in_concept[idx_in,:], out_concept[idx_out,:], optimal=False)
        sep_concept_ith = multivar_separa(in_concept[idx_in,:], out_concept[idx_out,:]).numpy()
        if logger:
            logger.info(f'[CLASS {i}: SEPARABILITY, CONCEPTS] separability using groundtruth ID/OOD: {sep_concept_ith}')

        separa['class'+str(i)] = sep_concept_ith

    return separa


def explain_topK(scores, top_k, separa, figname=None):
    """
    Plot bar graph of top-k largest average concept scores
    :param scores: concept scores, dim=(N,num_concepts)
    :param top_k: interested in printing top-k highest concept scores
    :param separa: separability score averaged across concepts or per-class multivariate separability
    """
    s_mean = np.mean(scores, axis=0)
    concept_idx = np.argsort(np.abs(s_mean))[::-1][:top_k] 
    
    num_types = 1 
    num_concepts = top_k
    bar_width = 0.35
    index = np.arange(num_concepts) * bar_width * (num_types + 1)

    fig, ax = plt.subplots(figsize=(3*top_k/5,3))
    bar = ax.bar(index + 0 * bar_width, s_mean[concept_idx],
            bar_width, yerr=np.std(scores[:,concept_idx],axis=0))
    ax.set_title('Top-{0} concept scores, separability: {1:.5f}'.format(top_k, separa))
    ax.set_ylabel('Concept score')
    ax.set_xticks(index + num_types * bar_width / 2)
    ax.set_xticklabels(['concept {}'.format(c) for c in concept_idx], rotation=45)
    fig.tight_layout()
    plt.savefig(figname)
    plt.close()


def explain_relative(scores, labels, separa, figname, figname_dist, top_k=6):
    """
    scores: dictionary of concept scores of groundtruth ID, groundtruth OOD, ID -> ID, ID -> OOD, OOD -> ID, OOD -> OOD
    labels: labels for different types of scores
    separa: separability scores, dim=(num_concepts,)
    """
    # concepts with top-k separability scores
    #concept_idx = np.argsort(separa)[::-1][:top_k] # top K: from largest to smallest value
    concept_idx = np.arange(top_k)
    num_types = len(labels)
    num_concepts = top_k
    bar_width = 0.35
    # create location for each bar. scale by an appropriate factor to ensure 
    # the final plot doesn't have any parts overlapping
    index = np.arange(num_concepts) * bar_width * (num_types + 1)

    fig, ax = plt.subplots(figsize=(3*top_k/2,3))
    for i in range(num_types):
        bar = ax.bar(index + i * bar_width, np.mean(scores[labels[i]][:,concept_idx],axis=0),
                bar_width, yerr=np.std(scores[labels[i]][:,concept_idx],axis=0), label=labels[i])
    ax.set_title('Concept scores for each concept and ID/OOD data')
    ax.set_ylabel('Concept score')
    ax.set_xticks(index + num_types * bar_width / 2)
    ax.set_xticklabels(['concept {}'.format(c) for c in concept_idx], rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig(figname)
    plt.close()
    """ 
    # plot score distribution
    fig, axes = plt.subplots(2, 3, sharex=False, sharey=True)
    fig.suptitle('Distribution of concept scores')
    for i, c in enumerate(concept_idx):
        axes[i//3, i%3].set_title('concept {0}: {1:.5f}'.format(c, separa[c]), fontsize=8)
        sns.kdeplot(scores['ID'][:,c], color='blue', ax=axes[i//3, i%3])
        sns.kdeplot(scores['OOD'][:,c], color='red', ax=axes[i//3, i%3])
    fig.legend(['ID', 'OOD'])
    fig.savefig(figname_dist)
    plt.close()
    """


def save_images(filepaths, figname, k=5):
    #if not len(filepaths):
    #    return

    k = min(k, len(filepaths))
    fig, axes = plt.subplots(1,k)
    count = 0
    np.random.shuffle(filepaths)
    for f in filepaths:
        img = Image.open(f).resize((100,100), Image.ANTIALIAS)
        axes[count].imshow(img)
        #ax2.set_title("ID image", size=10, color='b')
        axes[count].axis('off')
    
        count += 1
        if count >= k:
            break

    fig.savefig(figname)
    plt.close()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger = log.setup_logger(args, filename="eval_{}.log".format(args.score))
    in_loader, out_loader = prepare_data(args, logger)
    LOAD_DIR = 'data/Animals_with_Attributes2'
    TOPIC_PATH = os.path.join(args.result_dir,'topic_vec_inceptionv3.npy')
    INPUT_SHAPE = (args.out_data_dim, args.out_data_dim)
    TRAIN_DIR = "/nobackup/jihye/data/Animals_with_Attributes2/train"
    OOD_DATASET = args.out_data
    N_CLASSES = 50
    N_CONCEPTS_ORIG = 100 #np.shape(topic_vec_orig)[-1]
    _ = 0

    if args.score == 'ODIN':
        args.batch_size = 200

    if not os.path.exists(os.path.join(args.result_dir, 'plots')):
        os.makedirs(os.path.join(args.result_dir, 'plots'))
    if not os.path.exists(os.path.join(args.result_dir, 'explanations')):
        os.makedirs(os.path.join(args.result_dir, 'explanations'))
    if not os.path.exists(os.path.join(args.result_dir, 'explanations', args.out_data+'_'+args.score)):
        os.makedirs(os.path.join(args.result_dir, 'explanations', args.out_data+'_'+args.score))
    explain_dir = os.path.join(args.result_dir, 'explanations', args.out_data+'_'+args.score)
    
    ## load trained_model
    logger.info(f"Loading model from {args.model_path}")
    feature_model, predict_model = helper.load_model_inception_new(_, _, input_size=INPUT_SHAPE, pretrain=True, n_gpus=1, modelname=args.model_path)

    in_test_features = feature_model.predict(in_loader)
    out_test_features = feature_model.predict(out_loader)
    N_IN, N_OUT = in_test_features.shape[0], out_test_features.shape[0]

    ## load topic model
    topic_model = concept_model.TopicModel(in_test_features, N_CONCEPTS_ORIG, thres=0.2, predict=predict_model)
    topic_model(in_test_features)
    #topic_model.load_weights(args.result_dir+'/topic_epoch8.h5', by_name=True)
    topic_model.load_weights(args.result_dir+'/topic_latest.h5', by_name=True)
    ## load topic_vec
    topic_vec_orig = topic_model.layers[0].get_weights()[0]
    np.save(args.result_dir+'/topic_vec_orig.npy', topic_vec_orig)
    #topic_vec_orig = np.load(TOPIC_PATH)  # (512, 25) 
    logger.info(f'Number of concepts before removing duplicate ones: {str(N_CONCEPTS_ORIG)}')

    topic_vec, dict_dupl_topic = remove_duplicate_concepts(topic_vec_orig)
    N_CONCEPTS = np.shape(topic_vec)[-1] # 25
    logger.info(f'Number of concepts after removing duplicate ones: {str(N_CONCEPTS)}')
    
    in_test_concepts, in_test_logits = compute_concept_scores(topic_vec, in_test_features, predict_model)
    out_test_concepts, out_test_logits = compute_concept_scores(topic_vec, out_test_features, predict_model)
    in_test_yhat, out_test_yhat = np.argmax(in_test_logits, axis=1), np.argmax(out_test_logits, axis=1)

    ######################################
    ## Visualize the nearest neighbors
    if args.visualize:
        visualize_nn(in_loader, topic_vec, in_test_features, args.result_dir, logger)
    if args.visualize_with_ood:
        visualize_nn(out_loader, topic_vec, out_test_features, args.result_dir, logger, args.out_data)

    # target OOD detector
    logger.info("[ID TEST] performance of target OOD detector with test set...")
    in_test_scores, out_test_scores, thres95, auroc = run_eval(feature_model, predict_model, in_loader, out_loader, logger, args, N_CLASSES)
    #in_test_scores, out_test_scores, thres95, auroc = np.random.rand(N_IN), np.random.rand(N_OUT), 0.5419758558273315, 0.955332290562036

    # Plot ID vs OOD scores by the target detector
    savefig = os.path.join(args.result_dir, 'plots', '{}_AwA2_test_{}_test.jpg'.format(args.score, args.out_data))
    plot_stats(in_test_scores, out_test_scores, savename=savefig)
    
    ######################################
    ## Evaluating coherency......
    coherency = compute_coherency(in_test_features, topic_vec)
    logger.info(f'[ID TEST] coherency: {coherency}')

    ######################################
    ## Evaluating redundancy.......
    redundancy = compute_redundancy(topic_vec)
    logger.info(f'[CONCEPTS] redundancy: {redundancy}')

    #######################################
    ## Evaluating separability of concepts......
    #separa_path = os.path.join(args.result_dir, 'separability_AwA2_{}_raw.npy'.format(args.out_data))
    separa_path = os.path.join(args.result_dir, 'separability_{}_AwA2_{}_multiv.npy'.format(args.score, args.out_data))
    if args.separate:
        idx_IN_IN = in_test_scores >= thres95
        idx_IN_OUT = ~idx_IN_IN
        idx_OUT_IN = out_test_scores >= thres95
        idx_OUT_OUT = ~idx_OUT_IN
        #separa = compute_separability(in_test_concepts, out_test_concepts, in_test_yhat, out_test_yhat, logger) # using groundtruth ID/OOD labels
        in_detect_concepts = np.r_[in_test_concepts[idx_IN_IN], out_test_concepts[idx_OUT_IN]]
        out_detect_concepts = np.r_[in_test_concepts[idx_IN_OUT], out_test_concepts[idx_OUT_OUT]]
        in_detect_yhat = np.r_[in_test_yhat[idx_IN_IN], out_test_yhat[idx_OUT_IN]]
        out_detect_yhat = np.r_[in_test_yhat[idx_IN_OUT], out_test_yhat[idx_OUT_OUT]]
        separa = compute_separability(in_detect_concepts, out_detect_concepts, in_detect_yhat, out_detect_yhat, logger) # using detector's ID/OOD results in canonical world
        np.save(separa_path, separa)


    #######################################
    ## Evaluating the difference between two worlds......
    y_test = np.argmax(np.load('/nobackup/jihye/data/Animals_with_Attributes2/y_test.npy'), axis=1) # true labels

    logger.info("[ID TEST RECOVERED] performance of target OOD detector with test set...")
    in_test_scores_recov, out_test_scores_recov, _, auroc_recov = run_eval(feature_model, topic_model, in_loader, out_loader, logger, args, N_CLASSES)
    savefig = os.path.join(args.result_dir, 'plots', '{}_recov_AwA2_test_{}_test.jpg'.format(args.score, args.out_data))
    plot_stats(in_test_scores_recov, out_test_scores_recov, savename=savefig)

    # compute completeness scores
    _, logits_recov, _ = topic_model(in_test_features)
    in_test_yhat_recov = tf.math.argmax(logits_recov, axis=1).numpy()
    compute_completeness(y_test, in_test_yhat, in_test_yhat_recov, N_CLASSES, logger)
    compute_detection_completeness(auroc, auroc_recov, logger)
    
    ######################################
    ## Compute Hellinger distance between original vs reconstructed classifier outputs
    in_test_logits_recov = softmax(logits_recov).numpy()
    H = [hellinger(in_test_logits[i,:], in_test_logits_recov[i,:]) for i in range(in_test_logits.shape[0])]
    fig = plt.figure()
    ax = plt.subplot(111)
    sns.kdeplot(H, color='blue')
    ax.legend(['in-distribution (test)'])
    fig.savefig(os.path.join(args.result_dir, 'plots', 'classification_hellinger.jpg'))
    plt.close()

    #######################################
    ## Save results....
    results = {'in_yhat':in_test_yhat, 'out_yhat':out_test_yhat, 
            'in_yhat_recov':in_test_yhat_recov, 
            #'out_yhat_recov':out_test_yhat_recov,
            'in_logits':in_test_logits, 'in_logits_recov':in_test_logits_recov,
            'in_concepts':in_test_concepts, 'out_concepts':out_test_concepts,
            'in_scores':in_test_scores, 'out_scores':out_test_scores,
            #'thres':thres95,
            'in_scores_recov':in_test_scores_recov, 'out_scores_recov':out_test_scores_recov}

    result_path = os.path.join(args.result_dir,'results_{}_{}.pkl'.format(args.score,args.out_data))
    with open(result_path,'wb') as f:
        pickle.dump(results, f)

    #######################################
    ## Generating explanations.....

    separa = np.load(separa_path, allow_pickle=True).item()
    separa_global = separa['global']
    logger.info(f'[GLOBAL SEPARABILITY] multivariate separability: {separa_global}')
    separa_class = np.array([separa['class'+str(i)] for i in range(N_CLASSES)], dtype=np.float64)
    logger.info(f'[PER-CLASS SEPARABILIRY] averaged separability: {np.nanmean(separa_class)}')
    classes = np.argsort(separa_class)[::-1]
    #classes = np.append(classes[:5],[c for c in classes[-9:] if separa_class[c]]) # omitting classes with separability==0
    classes = np.delete(classes, np.where(np.isnan(separa_class[classes]))[0])

    if args.explain:
        logger.info(f'classes with highest and lowest separabilities...: {classes}')
        for i in classes:

            idx_in = np.where(in_test_yhat == i)[0]
            idx_out = np.where(out_test_yhat == i)[0]
            in_concepts_ith = in_test_concepts[idx_in,:] # concept scores of ID data classified as class i
            out_concepts_ith = out_test_concepts[idx_out,:] # concept scores of OOD data classified as class i
            
            if len(idx_in) < N_CONCEPTS or len(idx_out) < N_CONCEPTS:
                continue

            # indices for OOD detection results
            idx_IN_IN = in_test_scores[idx_in] >= thres95   # ID detected as ID
            idx_IN_OUT = ~idx_IN_IN                 # ID detected as OOD
            idx_OUT_OUT = out_test_scores[idx_out] < thres95 # OOD detected as OOD
            idx_OUT_IN = ~idx_OUT_OUT               # OOD detected as ID

            print(np.sum(idx_IN_IN))
            print(np.sum(idx_IN_OUT))
            print(np.sum(idx_OUT_OUT))
            print(np.sum(idx_OUT_IN))

            k = N_CONCEPTS #10
            explain_topK(in_concepts_ith, top_k=k, separa=separa_class[i], 
                        figname=os.path.join(explain_dir,'class{}_AwA2_top{}.jpg'.format(i, k)))
            explain_topK(out_concepts_ith, top_k=k, separa=separa_class[i],
                        figname=os.path.join(explain_dir,'class{}_{}_top{}.jpg'.format(i, args.out_data, k)))


            # most prominent concepts for ID/OOD images
            explain_topK(np.r_[in_concepts_ith[idx_IN_IN], out_concepts_ith[idx_OUT_IN]], top_k=k, separa=separa_class[i],
                        figname=os.path.join(explain_dir,'class{}_AwA2_top{}_detected_{}.jpg'.format(i, k, args.score)))
            explain_topK(np.r_[in_concepts_ith[idx_IN_OUT], out_concepts_ith[idx_OUT_OUT]], top_k=k, separa=separa_class[i],
                        figname=os.path.join(explain_dir,'class{}_{}_top{}_detected_{}.jpg'.format(i, args.out_data, k, args.score)))

            # relative comparison
            k = N_CONCEPTS #6
            scores = {}
            labels = ['ID', 'OOD', 'ID->OOD', 'OOD->ID']
            scores[labels[0]] = in_concepts_ith
            scores[labels[1]] = out_concepts_ith
            scores[labels[2]] = in_concepts_ith[idx_IN_OUT]
            scores[labels[3]] = out_concepts_ith[idx_OUT_IN]
            explain_relative(scores, labels, separa['class'+str(i)], 
                figname=os.path.join(explain_dir,'class{}_AwA2_{}_top{}_separability_{}.jpg'.format(i,args.out_data,k,args.score)),
                figname_dist=os.path.join(explain_dir,'class{}_AwA2_{}_distribution.jpg'.format(i,args.out_data)),
                top_k=k)

            
            if np.sum(idx_IN_IN)<5 or np.sum(idx_IN_OUT)<5 or np.sum(idx_OUT_OUT)<5 or np.sum(idx_OUT_IN)<5:
                continue
            # visualize example ID/OOD images
            in_files_ith = np.array(in_loader.filepaths)[idx_in]
            out_files_ith = np.array(out_loader.filepaths)[idx_out]
            save_images(in_files_ith[idx_IN_IN], figname=os.path.join(explain_dir,'class{}_{}_AwA2_IN.jpg'.format(i, args.score)))
            save_images(in_files_ith[idx_IN_OUT], figname=os.path.join(explain_dir,'class{}_{}_AwA2_OUT.jpg'.format(i, args.score)))
            save_images(out_files_ith[idx_OUT_OUT], figname=os.path.join(explain_dir,'class{}_{}_{}_OUT.jpg'.format(i, args.score, args.out_data)))
            save_images(out_files_ith[idx_OUT_IN], figname=os.path.join(explain_dir,'class{}_{}_{}_IN.jpg'.format(i, args.score, args.out_data)))

    logger.flush()


    ###########################################
    ## Computing ConceptSHAP.............
            
    if args.shap:
        nc = N_CONCEPTS_ORIG # number of concepts before duplicate removal
        #inputs = list(itertools.product([0, 1], repeat=N_CONCEPTS_ORIG)) #NOTE: computationally very expensive
        inputs = np.ones((len(dict_dupl_topic),nc))
        for d in dict_dupl_topic:
            idx = [d] + dict_dupl_topic[d]
            inputs[d,idx] = 0
        inputs = np.unique([tuple(row) for row in inputs], axis=0)
        #inputs = inputs[:2]

        #classes = [1, 4]
        outputs_class = np.array([])
        outputs_detect = np.array([])
        kernel = np.array([])
        for concept_mask in inputs:
            logger.info('======================================================')
            compl_class, compl_detect = compute_conceptSHAP(concept_mask, topic_vec_orig,
                                        in_test_features, out_test_features, y_test, in_test_yhat, out_test_yhat, auroc,
                                        in_loader, out_loader,
                                        topic_model, feature_model, args, logger,
                                        finetune=True, labels=classes)
            #compl_class, compl_detect: dim=(len(classes),)
            outputs_class = np.append(outputs_class, compl_class)
            outputs_detect = np.append(outputs_detect, compl_detect)
            k = np.sum(concept_mask)
            kernel = np.append(kernel, (nc-1)*1.0/((nc-k)*k*comb(nc, k)))

        outputs_class = outputs_class.reshape(-1,len(classes))
        outputs_detect = outputs_detect.reshape(-1,len(classes))
        kernel[kernel == np.inf] = 1e+4
        x = np.array(inputs)
        xkx = np.matmul(np.matmul(x.transpose(), np.diag(kernel)), x)
        shap_expl = {'mask': inputs}
        for i in range(len(classes)):
            xky_class = np.matmul(np.matmul(x.transpose(), np.diag(kernel)), outputs_class[:,i])
            shap_class = np.matmul(np.linalg.pinv(xkx), xky_class)
            shap_expl[f'shap_class_class{classes[i]}'] = shap_class

            idx = ~np.isnan(outputs_detect[:,i].astype(np.float))
            xkx_detect = np.matmul(np.matmul(x[idx,:].T, np.diag(kernel[idx])), x[idx,:])
            xky_detect = np.matmul(np.matmul(x[idx,:].T, np.diag(kernel[idx])), outputs_detect[idx,i])
            shap_detect = np.matmul(np.linalg.pinv(xkx), xky_detect)
            shap_expl[f'shap_detect_class{classes[i]}'] = shap_detect
            shap_expl[f'mask_detect_class{classes[i]}'] = x[idx,:]
    
        shap_path = os.path.join(explain_dir,'{}_SHAP.pkl'.format(args.score))
        with open(shap_path,'wb') as f:
            pickle.dump(shap_expl, f)


    """
    yhat_IN = np.argmax(pred_IN, axis=1)
    yhat_OUT = np.argmax(pred_OUT, axis=1)

    y_test = np.load("{}/y_test.npy".format(LOAD_DIR))
    y_test = np.argmax(y_test, axis=1)
    acc_test = np.sum(y_test == yhat_IN)/len(yhat_IN)
    logger.info(f'accuracy on test set: '+str(acc_test))
    """

    """
    def plot_softmax_distribution(idx_in, idx_out):
        print(pred_IN[idx_in])
        print(pred_OUT[idx_out])
        x = ['C'+str(i) for i in range(50)]
        plt.close() # if anything is open
        plt.bar(x, pred_IN[idx_in], color='b')
        plt.bar(x, pred_OUT[idx_out], color='r')
        plt.ylabel('Probability')
        plt.xlabel('Classes')
        plt.xticks(rotation=45)
        plt.legend(['In-distribution', 'Out-of-distribution'])
        plt.grid()
        plt.savefig(args.result_dir+'/softmax_AwA2_{}_{}_{}.jpg'.format(idx_in, args.out_data, idx_out))
        plt.close()
    if use_recovered:
        plot_softmax_distribution(1, 1)
        plot_softmax_distribution(100, 100)
        plot_softmax_distribution(1000, 1000)
        plot_softmax_distribution(np.argmax(np.max(pred_IN, axis=1)), np.argmax(np.max(pred_OUT, axis=1)))
        plot_softmax_distribution(np.argmin(np.max(pred_IN, axis=1)), np.argmin(np.max(pred_OUT, axis=1)))
    """
    

    if args.plot:
        log_path = '{}/{}'.format(args.logdir, args.name)
        plot_score_distr(profiles, in_test_concepts, in_test_yhat, out_test_concepts, out_test_yhat, save_plot=log_path)
        #plot_score_distr(concept_dict, scores_IN, yhat_IN, scores_OUT, yhat_OUT, CONFIDENCE, thresh=10, save_plot=log_path)

    logger.flush()
    



if __name__ == '__main__':
    parser = arg_parser()
    parser.add_argument('--gpu', required=True, type=str)
    parser.add_argument('--result_dir', type=str, help='path to directory where results from concept learning are stored', default='results/AwA2_baseline')
    parser.add_argument('--visualize', '-visualize', action='store_true', help='whether to visualize nearest neighbors with ID test data')
    parser.add_argument('--visualize_with_ood', '-visualize_with_ood', action='store_true', help='whether to visualize nearest neighbors with OOD test data')
    parser.add_argument('--shap', '-shap', action='store_true', help='whether to compute conceptSHAP using detection completeness score')
    parser.add_argument('--separate', '-separate', action='store_true', help='whether to evaluate separability')
    parser.add_argument('--explain', '-explain', action='store_true', help='whether to generate explanations')
    parser.add_argument('--plot', '-plot', action='store_true', help='default False - whether to plot concept score distributions - in vs out')
    #parser.add_argument('--pval', '-pval', action='store_true', help='default False - whether to transform concept scores to empirical p-vals')
    # arguments for OOD detection
    parser.add_argument('--out_data_dim', type=int, default=224, help='dimension of ood data')
    parser.add_argument('--score', choices=['Energy'], default='Energy')
    parser.add_argument('--temperature_energy', default=1, type=int,
                        help='temperature scaling for energy')


    main(parser.parse_args())

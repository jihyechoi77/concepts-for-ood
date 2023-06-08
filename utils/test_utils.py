import argparse
import unittest
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist, cifar10, cifar100
import tensorflow.keras.utils as utils
from tensorflow.keras.models import Model

import numpy as np
import scipy.io as sio
import sklearn.metrics as sk

import AwA2_helper
import ipca_v2
from utils.models import prepare_InceptionV3


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_data", type=str, default="/nobackup/jihye/data/Animals_with_Attributes2/test", help="Path to the in-distribution data folder.")
    parser.add_argument("--out_data", type=str, default="MSCOCO", help="Name of the out-of-distribution dataset.")

    parser.add_argument("--workers", type=int, default=8,
                        help="Number of background threads used to load data.")

    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to log test info (small).")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size.")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring and checkpointing.")

    parser.add_argument("--model", default="InceptionV3", help="Which variant to use")
    parser.add_argument("--model_path", type=str, default="results/Animals_with_Attributes2/inceptionv3_AwA2.h5", help="Path to the finetuned model you want to test")

    return parser


def scores_dim_reduced(topic_prob_n):
    # topic_prob_n: np array of normalized concept scores
    topic_prob_max = np.max(topic_prob_n, axis=(1,2))  # NOTE: matched with the processing in test_ours.py!!!
    topic_prob_min = np.min(topic_prob_n, axis=(1,2))
    concept_scores = np.where(topic_prob_max > np.abs(topic_prob_min), topic_prob_max, topic_prob_min)
    return concept_scores


def remove_duplicate_concepts(topic_vec, n_concept):
    # Remove one concept vector if there are two vectors where the dot product is over 0.95
    # topic_vec: dim=(dim_features, n_concepts) (2048, 70)
    # print(np.shape(topic_vec))

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

    return topic_vec



def prepare_data(args, logger):
    # Load In-distribution dataset
    datagen = ImageDataGenerator(rescale=1.0 / 255.)
    in_gen = datagen.flow_from_directory(args.in_data,
                                        batch_size=args.batch_size,
                                        target_size=(224,224),
                                        class_mode='categorical',
                                        shuffle=False)
    # Load OOD dataset
    OOD_DATASET = args.out_data
    
    if OOD_DATASET == 'MSCOCO':
        OOD_DIR = '/nobackup/jihye/data/MSCOCO_test'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                            batch_size=args.batch_size,
                                            target_size=(224,224),
                                            class_mode=None, shuffle=False)

    elif OOD_DATASET == 'mnist':
        (_, _), (x_out, y_out) = mnist.load_data() # test set from CIFAR-10
        x_out = x_out.reshape(-1,28,28,1)
        x_out = np.concatenate((x_out, x_out, x_out), axis=3)
        y_out = utils.to_categorical(y_out, 10)
        
        x_out = tf.cast(x_out, tf.float32)
        x_out = tf.image.resize(x_out, (224, 224))
        out_gen = datagen.flow(x_out.numpy(), y_out, batch_size=args.batch_size, shuffle=False)

    elif OOD_DATASET == 'cifar10':
        #THRESHOLD_energy = np.linspace(7.0, 9.5, 30)
        #THRESHOLD_smx = np.linspace(0.84, 0.95, 20)
        (_, _), (x_out, y_out) = cifar10.load_data() # test set from CIFAR-10
        y_out = utils.to_categorical(y_out, 10)
        #x_cifar10 = tf.keras.applications.inception_v3.preprocess_input(x_cifar10)

        out_gen = datagen.flow(x_out, y_out, batch_size=args.batch_size, shuffle=False)

    elif OOD_DATASET == 'cifar100':
        #THRESHOLD_energy = np.linspace(7.0, 9.5, 30)
        #THRESHOLD_smx = np.linspace(0.84, 0.95, 20)
        (_, _), (x_out, y_out) = cifar100.load_data() # test set from CIFAR-100
        y_out = utils.to_categorical(y_out, 100)

        out_gen = datagen.flow(x_out, y_out, batch_size=args.batch_size, shuffle=False)

    elif OOD_DATASET == 'svhn':
        #THRESHOLD_energy = np.linspace(5.5, 7.0, 30)
        #THRESHOLD_smx = np.linspace(0.4, 0.55, 20)
        svhn_data = sio.loadmat('/nobackup/jihye/data/SVHN_test_32x32.mat')
        x_out = svhn_data['X'].astype('float32') #/ 255.0
        x_out = np.transpose(x_out, (3,0,1,2)) # dim = (N_imgs, 32, 32, 3)
        y_out = svhn_data['y'].astype('float32')-1 # label from 0 to 9
        y_out = utils.to_categorical(y_out, 10)

        out_gen = datagen.flow(x_out, y_out, batch_size=args.batch_size, shuffle=False)

    elif OOD_DATASET == 'AwA2-test-pgd':
        OOD_DIR = '/nobackup/jihye/data/AwA2-test-pgd'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                            batch_size=args.batch_size,                                                                                         target_size=(224,224), #(32,32)
                                            class_mode=None, shuffle=False)

    elif OOD_DATASET == 'AwA2-test-fractals':
        OOD_DIR = '/nobackup/jihye/data/AwA2-test-fractals'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                            batch_size=args.batch_size,                                                                                         target_size=(224,224), #(32,32)
                                            class_mode=None, shuffle=False)
    elif OOD_DATASET == 'Animals':
        OOD_DIR = '/nobackup/jihye/data/Animals'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                            batch_size=args.batch_size,                                                                                         target_size=(224,224), #(32,32)
                                            class_mode=None, shuffle=False)
    elif OOD_DATASET == 'iNaturalist':
        OOD_DIR = '/nobackup/jihye/data/iNaturalist'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                              batch_size=args.batch_size,
                                              target_size=(224,224), #(32,32)
                                              class_mode=None, shuffle=False)

    elif OOD_DATASET == 'Textures':
        OOD_DIR = '/nobackup/jihye/data/Textures'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                              batch_size=args.batch_size,
                                              target_size=(224,224),
                                              class_mode='categorical', shuffle=False)
    elif OOD_DATASET == 'Places':
        OOD_DIR = '/nobackup/jihye/data/Places'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                                batch_size=args.batch_size,
                                                target_size=(224,224), #(32,32)
                                                class_mode=None, shuffle=False)

    elif OOD_DATASET == 'SUN':
        OOD_DIR = '/nobackup/jihye/data/SUN'
        out_gen = datagen.flow_from_directory(OOD_DIR,
                                            batch_size=args.batch_size,
                                            target_size=(224,224), #(32,32)
                                            class_mode=None, shuffle=False)

    logger.info(f"Using an in-distribution set.") #with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set.") #with {len(out_set)} images.")

    return in_gen, out_gen


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    print(cutoff)
    #return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])
    return thresholds[cutoff], fps[cutoff] / (np.sum(np.logical_not(y_true)))

def get_measures(in_examples, out_examples, recall_level=0.95):
    # in_examples: pos, out_examples: neg
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    # logger.info("# in example is: {}".format(num_in))
    # logger.info("# out example is: {}".format(num_out))

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    threshold, fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr, threshold


def show_performance(pos, neg, method_name='Ours', recall_level=0.95):
    '''
    :param pos: 1's class, class to detect, outliers, or wrongly predicted
    example scores
    :param neg: 0's class scores
    '''

    auroc, aupr_in, aupr_out, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR IN:\t\t\t{:.2f}'.format(100 * aupr_in))
    print('AUPR OUT:\t\t\t{:.2f}'.format(100 * aupr_out))
    # print('FDR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fdr))


def run_concept(model, data_loader, \
                model_path='results/Animals_with_Attributes2/inceptionv3_AwA2.h5', \
                topic_model_path='results/Animals_with_Attributes2_COCO_mahal_coeff_10.0/latest_topic.h5',
                label=None):
    _=None
    feature_model, predict_model = AwA2_helper.load_model_inception_new(_, _, input_size=(32,32), pretrain=True, n_gpus=1, modelname=model_path)

    feature = feature_model.predict(data_loader)
    #logits = predict_model(feature)

    model = prepare_InceptionV3(modelpath=model_path, input_size=(32,32), pretrain=True, return_model=True)
    logits = model.predict(data_loader)
    model.evaluate(data_loader)

    topic_model = ipca_v2.TopicModel(feature[:2], n_concept=70, thres=0.2, predict=predict_model)
    _ = topic_model(feature[:2]) # call the subclassed model first
    topic_model.load_weights(topic_model_path, by_name=True)
    logits_ = topic_model(feature, training=False)

    label = np.argmax(np.load('data/Animals_with_Attributes2/y_test.npy'), axis=1)
    print(np.argmax(logits,axis=1))
    print(tf.math.reduce_max(logits, axis=1))
    if len(label)==logits.shape[0]:
        acc = np.sum(label == np.argmax(logits, axis=1)) / len(label)
        acc_ = np.sum(label == np.argmax(logits_, axis=1)) / len(label)
        print('accuracy with original features: {}'.format(acc))
        print('accuracy with recovered features: {}'.format(acc_))
    
    return logits, logits_


def get_recovered_features(feat_in, feat_out, predict_model, \
        topic_model_path='results/Animals_with_Attributes2/latest_topic.h5',
        n_concept=70, eval=False):
    
    """
    Compute intermediate representations which are recovered from concept scores
    :param feat_in: [ID data] intermediate representations from the first half of the classifier
    :param feat_out: [OOD data] intermediate representations from the first half of the classifier
    :param predict_model: second half of the classifier -- input: intermediate representations, output: class label
    :param n_concept: number of concepts used during concept learning
    """
    
    #_=None
    thres = 0.2

    #topic_model, _, _, _, _, _ = ipca_v2.topic_model_new(predict_model, feat_in, _, _, _, n_concept, verbose=0, thres=thres)
    #topic_model = Model(input=topic_model.inputs, output=topic_model.layers[-2])
    topic_model = ipca_v2.TopicModel_V2(feat_in, n_concept, thres) 
    topic_model(feat_in)
    topic_model.load_weights(topic_model_path, by_name=True)
    
    feat_recov_in = topic_model(feat_in) # features recovered from concept scores
    feat_recov_out = topic_model(feat_out)
    
    if eval:
        logits = predict_model(feat_in)
        yhat = tf.math.argmax(logits, axis=1).numpy()
        logits_recov = predict_model(feat_recov_in)
        yhat_recov = tf.math.argmax(logits_recov, axis=1).numpy()

        # check the recovered accuracy
        y = np.load('data/Animals_with_Attributes2/y_test.npy') # true labels
        n_class = y.shape[1]
        y = np.argmax(y, axis=1)
        acc = np.sum(y == yhat) / len(y)
        acc_recov = np.sum(y == yhat_recov) / len(y)
        acc_random = 1/n_class
        print(f'[ID TEST] accuracy with original features: {acc}')
        print(f'[ID TEST] accuracy with recovered features: {acc_recov}')
        print(f'[ID TEST] completeness score: {(acc_recov-acc_random)/(acc-acc_random)}')

    return feat_recov_in.numpy(), feat_recov_out.numpy()


def compute_concept_scores(topic_vec, feature, verbose=False):
    # topic_vec: concept vectors (dim= (feature_dim, n_concepts))
    # feature: features extracted from an intermediate layer of trained model

    feature_n = feature/(np.linalg.norm(feature,axis=3,keepdims=True)+1e-9)
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)

    topic_prob = np.matmul(feature_n,topic_vec_n)

    topic_prob_max = np.max(topic_prob, axis=(1,2))  # TODO: confirm this part!!!
    topic_prob_min = np.min(topic_prob, axis=(1,2))
    concept_scores = np.where(topic_prob_max > np.abs(topic_prob_min), topic_prob_max, topic_prob_min) # dim = (n_features, n_concepts)
    """
    n_data, h, w, n_concept = topic_prob.shape
    concept_scores = topic_prob.reshape(n_data, h*w*n_concept)
    """

    """
    ##for debugging
    n_concept = np.shape(concept_scores)[1]
    print(tf.reduce_mean(input_tensor=tf.nn.top_k(K.transpose(K.reshape(topic_prob,(-1,n_concept))),k=10,sorted=True).values))
    print(tf.reduce_mean(input_tensor=K.dot(K.transpose(K.variable(value=topic_vec_n)), K.variable(value=topic_vec_n)) - np.eye(n_concept)))
    """

    if verbose:
        topic_prob_mean = np.mean(topic_prob, axis=(1,2))
        topic_prob_std = np.std(topic_prob, axis=(1,2))
        return concept_scores, topic_prob_mean, topic_prob_std

    else:
        return concept_scores



class ConceptProfiles(unittest.TestCase):
    # return per-class concept profiles

    def setUp(self, N_CLASSES, dataLoader):
        self.concept_dict = [{'scores':np.array([])} for i in range(N_CLASSES)]
        self.count = 0
        self.loader = dataLoader
        self.numUpdates = len(self.loader.filenames) // self.loader.batch_size

    def prepare_concept_dict(self, feature_model, topic_vec):
        """
        Prepare concept dictionary with in-domain data
        :param feature_model: first half of the classifier -- input: image, output: intermediate representation
        :param topic_vec: concept vectors
        """

        for step, (x_, y_) in enumerate(self.loader):
            if step > self.numUpdates:
                break

            f_ = feature_model(x_)
            y_ = np.argmax(y_, axis=1)
            scores_ = compute_concept_scores(topic_vec, f_)

            y_unique = np.unique(y_)
            for label in y_unique:
                print('profiling class {}.....'.format(label))
                idx = np.where(y_ == label)[0]
                self.concept_dict[label]['scores'] = np.r_[self.concept_dict[label]['scores'], scores_[idx,:]] if self.concept_dict[label]['scores'].any() else scores_[idx, :]

                self.count += len(idx)


                #for i in range(N_CLASSES):
                #    concept_dict[i]['scores'] = np.array(concept_dict[i]['scores']).reshape(-1,N_CONCEPTS)
        self.assertEqual(self.count, len(self.loader.filenames))



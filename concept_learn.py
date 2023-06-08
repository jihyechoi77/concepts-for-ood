#   lint as: python3`
"""Main file to run concept learning with AwA dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concept_model
import helper
from utils.log import setup_logger
from utils.ood_utils import run_ood_over_batch
from utils.test_utils import get_measures
from utils.stat_utils import multivar_separa 
from test_baselines import run_eval

from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
import tensorflow.keras.utils as utils
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

import os
import argparse
import logging
import numpy as np
import sys
import time


def get_data(bs, ood=True):
    """
    prepare data loaders for ID and OOD data (train/test)
    :param bs: batch size
    :ood: whether to load OOD data as well (False for baseline concept learning by Yeh et al.)
    """

    TRAIN_DIR = "/nobackup/jihye/data/Animals_with_Attributes2/train"
    VAL_DIR = "/nobackup/jihye/data/Animals_with_Attributes2/val"
    TEST_DIR = "/nobackup/jihye/data/Animals_with_Attributes2/test"
    if args.out_data == 'MSCOCO':
        OOD_DIR = "/nobackup/jihye/data/MSCOCO"
    elif args.out_data == 'augAwA':
        OOD_DIR = "/nobackup/jihye/data/AwA2-train-fractals"

    TARGET_SIZE = (224, 224)
    BATCH_SIZE = bs
    BATCH_SIZE_OOD = bs

    print('Loading images through generators ...')
    train_datagen = ImageDataGenerator(rescale=1. / 255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    train_loader = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    batch_size=BATCH_SIZE,
                                                    target_size=TARGET_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

    #print(train_generator.class_indices.items())

    datagen = ImageDataGenerator(rescale=1.0 / 255.)
    val_loader = datagen.flow_from_directory(VAL_DIR,
                                            batch_size=BATCH_SIZE,
                                            target_size=TARGET_SIZE,
                                            class_mode='categorical',
                                            shuffle=False)
    test_loader = datagen.flow_from_directory(TEST_DIR,
                                            batch_size=BATCH_SIZE,
                                            target_size=TARGET_SIZE,
                                            class_mode='categorical',
                                            shuffle=False)
    if ood:
        #numUpdates = int(NUM_TRAIN / BATCH_SIZE) # int(f_train.shape[0] / BATCH_SIZE)
        #NUM_OOD = 31706
        #BATCH_SIZE_OOD = int(NUM_OOD / numUpdates)
        OOD_loader = train_datagen.flow_from_directory(OOD_DIR, #datagen
                                                batch_size=BATCH_SIZE_OOD,
                                                target_size=TARGET_SIZE,
                                                class_mode=None, shuffle=True)
    else:
        OOD_loader = None

    return train_loader, val_loader, test_loader, OOD_loader


def get_class_labels(loader, savepath):
    """
    extract groundtruth class labels from data loader
    :param loader: data loader
    :param savepath: path to the numpy file
    """

    if os.path.exists(savepath):
        y = np.load(savepath)
    else:
        num_data = len(loader.filenames)
        y = []
        for (_, y_batch), _ in zip(loader, range(len(loader))):
            y.extend(y_batch)
       
        np.save(savepath, y)
    return y

def get_args():
    parser = argparse.ArgumentParser(description='concept learning (both baseline and OOD)')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='optimizer')
    parser.add_argument('--thres', type=float, default=0.2, help='threshold for concept scores')
    parser.add_argument('--val_step', type=int, default=2, help='how often to test with validation set during training')
    parser.add_argument('--save_step', type=int, default=2, help='how often to save the topic model during training')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--trained', '-trained', action='store_true', help='default False - whether topic model is trained')
    parser.add_argument('--num_concepts', type=int, default=70, help='number of concepts; parameter for concept learning')
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--name', type=str, required=True, help='directory to save trained topic model and concepts')
    # different options for concept learning objective
    parser.add_argument('--feat_l2', '-feat_l2', action='store_true', help='whether to use ||feat - recovered feat||_2 regularizer') 
    parser.add_argument('--feat_cosine', '-feat_cosine', action='store_true', help='whether to use cosine distance regularizer between feat and recovered feat')
    parser.add_argument('--separability', '-separability', action='store_true', help='whether to use separability regularization')
    parser.add_argument('--coeff_feat', type=float, default=0.1, help='coefficient for loss_l2')
    parser.add_argument('--coeff_cosine', type=float, default=1., help='coefficient for loss_cos')
    parser.add_argument('--coeff_score', type=float, default=0., help='coefficient for loss_score')
    parser.add_argument('--coeff_concept', type=float, default=10., help='coefficient for loss_coherency and loss_similarity')
    parser.add_argument('--coeff_separa', type=float, default=10., help='coefficient for loss_separa')
    parser.add_argument('--num_hidden', type=int, default=2, help='number of hidden layers for mapping g')
    #parameters for OOD detection
    parser.add_argument('--out_data', type=str, choices=['MSCOCO', 'augAwA'], default='MSCOCO', help='Auxiliary OOD Dataset during concept learning')
    parser.add_argument('--ood', '-ood', action='store_true', help='whether to outsource OOD data during concept learning')
    parser.add_argument('--score', type=str, choices=['energy'], default=None, help='OOD detector type')
    parser.add_argument('--temperature_odin', default=1000, type=int, help='temperature scaling for odin')
    parser.add_argument('--epsilon_odin', default=0.0, type=float, help='perturbation magnitude for odin')
    parser.add_argument('--temperature_energy', default=1, type=int, help='temperature scaling for energy')


    return parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #if not os.path.exists(args.output_dir):
    #    os.makedirs(args.output_dir)

    if args.separability:
        args.ood = True
    USE_OOD = args.ood
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    THRESHOLD = args.thres
    trained = args.trained
    N_CONCEPT = args.num_concepts
    offset = args.offset
    topic_modelpath = os.path.join(args.logdir, args.name,'topic_epoch{}.h5'.format(offset))
    #topic_modelpath = os.path.join(args.logdir, args.name,'topic_latest.h5')
    topic_savepath = os.path.join(args.logdir, args.name,'topic_vec_inceptionv3.npy')

    logger = setup_logger(args)

    train_loader, val_loader, test_loader, ood_loader =  get_data(BATCH_SIZE, ood=USE_OOD)

    ## splitting AwA2 data into train/val/test sets
    # helper.prepare_data()
    
    #print(train_generator.class_indices.items())
    #assert ('_OOD', 0) in val_generator.class_indices.items()
    #y_train = get_class_labels(train_loader, savepath='data/Animals_with_Attributes2/y_train.npy')
    y_val = get_class_labels(val_loader, savepath='/nobackup/jihye/data/Animals_with_Attributes2/y_val.npy')
    y_test = get_class_labels(test_loader, savepath='/nobackup/jihye/data/Animals_with_Attributes2/y_test.npy')
    
    # preds_cls_idx = y_test.argmax(axis=-1)
    # idx_to_cls = {v: k for k, v in test_generator.class_indices.items()}
    # preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
    # filenames_to_cls = list(zip(test_generator.filenames, preds_cls))


    # Loads model
    feature_model, predict_model = helper.load_model_inception_new(train_loader, val_loader, \
               batch_size=BATCH_SIZE, input_size=(224,224), pretrain=True, \
               modelname='./results/Animals_with_Attributes2/inceptionv3_AwA2.h5', split_idx=-5)
    """
    #### check accuracy of feature_model -> predict_model
    logits_test = predict_model(feature_model.predict(test_loader))
    yhat_test = tf.math.argmax(logits_test, axis=1).numpy()
    acc_test = np.sum(yhat_test == np.argmax(y_test, axis=1))/len(yhat_test)
    logger.info(f'[ID TEST] accuracy after splitting the original classifier: {acc_test}') # 0.9212619300106044
    """

    """
    num_val_OOD = np.sum(np.char.find(val_generator.filenames, '_OOD')==0)
    num_val_ID = y_val.shape[0]
    y_val = np.r_[np.zeros((num_val_OOD, y_val.shape[1])), y_val]
    y_val = np.c_[np.r_[np.ones((num_val_OOD,1)), np.zeros((num_val_ID,1))],y_val]
    """
    
    ## Concept Learning
    x, _ = test_loader.next()
    f = feature_model(x[:10])
    # topic model: intermediate feature --> concept score --> recovered feature --> prediction (50 classes)
    topic_model_pr = concept_model.TopicModel(f, N_CONCEPT, THRESHOLD, predict_model, args.num_hidden)
    _ = topic_model_pr(f)
    print(topic_model_pr.summary())

    if args.opt =='sgd':
        """
        optimizer = SGD(lr=0.1)
        optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.momentum, optimizer.decay]
        optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
        """
        optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    elif args.opt =='adam':
        optimizer = Adam(lr=0.01)
        optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1, optimizer.beta_2, optimizer.decay]
        optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()
    test_acc_metric = keras.metrics.CategoricalAccuracy()
    softmax = layers.Activation('softmax')

    @tf.function
    def train_step(x_in, y_in, x_out=None, thres=None):
        #tf.keras.applications.inception_v3.preprocess_input(x_in)
        f_in = feature_model(x_in)
        f_in_n = K.l2_normalize(f_in,axis=(3))


        obj_terms = {} # terms in the objective function
        COEFF_CONCEPT = args.coeff_concept #10 -> 5 -> 1 
        with tf.GradientTape() as tape:
            f_in_recov, logits_in, topic_vec_n = topic_model_pr(f_in, training=True)
            pred_in = softmax(logits_in) # class prediction using concept scores
            topic_prob_in_n = K.dot(f_in_n, topic_vec_n) # normalized concept scores

            # total loss
            CE_IN = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_in, pred_in))
            loss_coherency = tf.reduce_mean(tf.nn.top_k(K.transpose(K.reshape(topic_prob_in_n,(-1,N_CONCEPT))),k=10,sorted=True).values)
            loss_similarity = tf.reduce_mean(K.dot(K.transpose(topic_vec_n), topic_vec_n) - tf.eye(N_CONCEPT))
            loss = CE_IN - COEFF_CONCEPT*loss_coherency + COEFF_CONCEPT*loss_similarity  # baseline: Yeh et al.
            obj_terms['[ID] CE'] = CE_IN
            obj_terms['[ID] concept coherency'] = loss_coherency
            obj_terms['[ID] concept similarity'] = loss_similarity
            #print('y_in: '+type(y_in).__name__)
            #print('pred_in: '+type(pred_in).__name__)
            #print('CE_IN: '+type(CE_IN).__name__)
            #print('loss coher: '+type(loss_coherency).__name__)
            #print('loss_sim: '+type(loss_similarity).__name__)
            #print('loss: '+type(loss).__name__)
            
            if args.feat_l2:
                loss_l2 = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(f_in-f_in_recov,2), axis=(1,2,3))))
                #loss_l2 = tf.reduce_mean(tf.reduce_sum(tf.pow(f_in-f_in_recov,2), axis=(1,2,3)))
                loss += args.coeff_feat*loss_l2 #0.07, 0.02
                obj_terms['feature L2'] = loss_l2

            if args.feat_cosine:
                loss_cosine = tf.reduce_mean(tf.keras.losses.cosine_similarity(f_in, f_in_recov)) # equivalent to: tf.reduce_mean(tf.reduce_sum(tf.math.multiply(f_in, f_in_recov),axis=(1,2,3))/(tf.sqrt(tf.reduce_sum(tf.pow(f_in,2),axis=(1,2,3)))*tf.sqrt(tf.reduce_sum(tf.pow(f_in_recov,2),axis=(1,2,3)))))
                loss_cosine = 1 - loss_cosine # cosine distance, range=[0, 2]
                loss += args.coeff_cosine*loss_cosine
                obj_terms['feature cosine distance'] = loss_cosine

            
            if args.score:
                s_in = run_ood_over_batch(x_in, feature_model, predict_model, args, num_classes=50)
                s_out = run_ood_over_batch(x_out, feature_model, predict_model, args, num_classes=50)

                if args.coeff_score > 0.0:
                    # scores from OOD detector when using recovered features
                    s_in_recov = run_ood_over_batch(x_in, feature_model, topic_model_pr, args, num_classes=50)
                    s_out_recov = run_ood_over_batch(x_out, feature_model, topic_model_pr, args, num_classes=50)

                    s_original = tf.concat((s_in, s_out), axis=0)
                    s_recovered = tf.concat((s_in_recov, s_out_recov), axis=0)
                    loss_score = tf.reduce_mean(tf.pow(s_original - s_recovered, 2))
                    loss += args.coeff_score*loss_score
                    obj_terms['score difference'] = loss_score

                    """
                    # Debugging
                    auroc, aupr_in, aupr_out, fpr95, thres95 = get_measures(s_in.numpy()[:,None], s_out.numpy()[:,None])
                    print(f'auroc: {auroc}, aupr in: {aupr_in}, aupr out: {aupr_out}, fpr95: {fpr95}')
                    auroc, aupr_in, aupr_out, fpr95, thres95 = get_measures(s_in_recov.numpy()[:,None], s_out_rec
ov.numpy()[:,None])
                    print(f'auroc: {auroc}, aupr in: {aupr_in}, aupr out: {aupr_out}, fpr95: {fpr95}')
                    input()
                    """
            
            if args.separability:
                f_out = feature_model(x_out)
                f_out_n = K.l2_normalize(f_out,axis=(3))
                _, logits_out, _ = topic_model_pr(f_out, training=True)
                #tf.debugging.assert_equal(topic_vec_n, topic_vec_n_out) 
                topic_prob_out_n = K.dot(f_out_n, topic_vec_n)
                

                # max --> smoothly approximated by logsumexp
                #T = tf.Variable(1e+3, dtype=tf.float32)
                T = 1e+3
                prob_max_in = 1/T*tf.math.reduce_logsumexp(T*topic_prob_in_n,axis=(1,2))
                prob_min_in = -1/T*tf.math.reduce_logsumexp(-T*topic_prob_in_n,axis=(1,2))

                ## concept scores of "true" ID set and "true" OOD set
                concept_in_true = tf.where(tf.abs(prob_max_in) > tf.abs(prob_min_in), prob_max_in, prob_min_in)
                prob_max_out = 1/T*tf.math.reduce_logsumexp(T*topic_prob_out_n,axis=(1,2))
                prob_min_out = -1/T*tf.math.reduce_logsumexp(-T*topic_prob_out_n,axis=(1,2))
                concept_out_true = tf.where(tf.abs(prob_max_out) > tf.abs(prob_min_out), prob_max_out, prob_min_out)
                
                ## concept scores of "detected" ID set and "detected" OOD set
                concept_in = tf.concat([concept_in_true[s_in>=thres], concept_out_true[s_out>=thres]], axis=0) 
                concept_out = tf.concat([concept_in_true[s_in<thres], concept_out_true[s_out<thres]], axis=0)

                # global separability
                loss_separa = multivar_separa(concept_in, concept_out)
                loss -= args.coeff_separa*loss_separa
                obj_terms['separability'] = loss_separa

        obj_terms['total loss.......'] = loss
        train_acc_metric.update_state(y_in, logits_in)
        #print(obj_terms)

        # calculate the gradients using our tape and then update the model weights
        grads = tape.gradient(loss, topic_model_pr.trainable_variables)
        optimizer.apply_gradients(zip(grads, topic_model_pr.trainable_variables))
        #print(type(loss).__name__, ":", grads)
        #input()
        return obj_terms


    if os.path.exists(topic_modelpath):
        topic_model_pr.load_weights(topic_modelpath, by_name=True)
        logger.info(f'topic model loaded from {topic_modelpath}')
    if not trained:
        for layer in topic_model_pr.layers[:-1]:
            #print(layer.trainable)
            layer.trainable = True

        # check all weights are included in trainable_variables
        # for i, var in enumerate(topic_model_pr.trainable_variables):
        #     print(topic_model_pr.trainable_variables[i].name)


        if args.score and args.separability: # identify threshold from held-out set
            datagen = ImageDataGenerator(rescale=1.0 / 255.)
            if args.out_data == 'MSCOCO':
                out_gen = datagen.flow_from_directory('/nobackup/jihye/data/MSCOCO_test',batch_size=150,target_size=(224,224),class_mode=None,shuffle=False)
            elif args.out_data == 'augAwA':
                out_gen = datagen.flow_from_directory('/nobackup/jihye/data/AwA2-test-fractals',batch_size=150,target_size=(224,224),class_mode=None,shuffle=False)
            _, _, thres, _ = run_eval(feature_model, predict_model, val_loader, out_gen, logger, args, 50)
        else:
            thres = None

        for epoch in range(offset+1, offset+EPOCH+1):
            logger.info(f"\n[INFO] starting epoch {epoch}/{offset+EPOCH} ---------------------------------")
            sys.stdout.flush()
            epochStart = time.time()
            
            for step, (x_in, y_in) in enumerate(train_loader):
                step += 1 # starts from 1

                if step > len(train_loader):
                    break

                if USE_OOD:
                    x_out = ood_loader.next()
                    obj_terms = train_step(x_in, y_in, x_out, thres)
                else:
                    obj_terms = train_step(x_in, y_in)

                # Log every 50 batches
                if step % 20 == 0:
                    #print(topic_model_pr.layers[0].get_weights()[0])
                    for term in obj_terms:
                        logger.info(f'[STEP{step}] {term}: {obj_terms[term]}')
            
            train_acc = train_acc_metric.result()
            logger.info("Training acc over epoch: %.4f" % (float(train_acc),))
            
            # show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart) / 60.0
            logger.info("Time taken: %.2f minutes" % (elapsed))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
            if epoch % args.save_step == 0:
                topic_model_pr.save_weights(os.path.join(args.logdir, args.name,'topic_epoch{}.h5'.format(epoch)))

            if epoch % args.val_step == 0:
                _, logits_val, _ = topic_model_pr(feature_model.predict(val_loader), training=False)
                pred_val = softmax(logits_val)
                val_acc_metric.update_state(y_val, logits_val)
                val_acc = val_acc_metric.result()
                logger.info("[EPOCH %d] Validation acc: %.4f" % (epoch, float(val_acc)))
                val_acc_metric.reset_states()
                del logits_val
            
            logger.flush()


    topic_vec = topic_model_pr.layers[0].get_weights()[0]   # 1, (2048, num_concepts)
    # recov_vec = topic_model_pr.layers[-3].get_weights()[0]
    topic_vec_n = topic_vec/(np.linalg.norm(topic_vec,axis=0,keepdims=True)+1e-9)
    np.save(topic_savepath,topic_vec)
    # np.save('results/Animals_with_Attributes2_energy_COCO/recov_vec_inceptionv3.npy',recov_vec)

    assert np.shape(topic_vec)[1] == N_CONCEPT
    # topic_model_pr.evaluate(f_test, y_test)
    # f_val_recovered = topic_model_pr.predict(f_val)
    

    f_test = feature_model.predict(test_loader)
    _, logits_test, _ = topic_model_pr(f_test, training=False)
    pred_test = softmax(logits_test)
    test_acc_metric.update_state(y_test, logits_test)
    test_acc = test_acc_metric.result()
    logger.info('[ID TEST] Accuracy of topic model on test set: %f' %test_acc)
   
    logger.flush()


if __name__ == '__main__':
    global args
    args = get_args()
    main()

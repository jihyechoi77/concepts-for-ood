"""this is where you plug in your OOD detector"""

import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as layers



def get_prediction(predict_model, features):
    logits = predict_model(features)
    if len(logits) == 3: # when TopicModel is used as predict_model
        logits = logits[1]
    return logits


def iterate_data_energy(x, feature_model, predict_model, temper, features=None):
    logits = get_prediction(predict_model, feature_model(x) if features is None else features)
    Ec = -temper * tf.reduce_logsumexp(logits / temper, axis=1)
    
    return -Ec #.numpy()


def run_ood_over_batch(x, feature_model, predict_model, args, num_classes, features=None):

    if np.char.lower(args.score) == 'energy':
        scores = iterate_data_energy(x, feature_model, predict_model, args.temperature_energy, features)

    return scores #.reshape((0,1))

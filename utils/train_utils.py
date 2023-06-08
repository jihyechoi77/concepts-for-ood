import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_datagen(data_dir, train=False, shuffle=False,
                class_mode='categorical', # None when loading unlabaled data
                batch_size=128, target_size=(224,224)):

    print('Loading images through generators ...')
    if train:
        datagen = ImageDataGenerator(rescale=1. / 255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255.)

    generator = datagen.flow_from_directory(data_dir, \
                                        batch_size=batch_size, target_size=target_size, \
                                        class_mode=class_mode, shuffle=shuffle)
    num_samples = len(generator.filenames)
    return generator, num_samples

"""
from tensorflow.keras.utils import Sequence
class MySequence(Sequence):
    def __init__(self, seq1, seq2):
        self.seq1, self.seq2 = seq1, seq2
    def __len__(self):
        return len(self.seq1)

    def __getitem__(self, idx):
        data1 = self.seq1[idx]
        data2 = self.seq2[idx]
        return [data1[0], data2[0]], [data1[1], data2[1]]
"""
def load_combined_datagen(datagen1, datagen2):
    # label is only from datagen1
    while True:
        x1 = datagen1.next()
        x2 = datagen2.next()
        yield [x1[0], x2], x1[1]


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return cov_xx


def tf_mahalanobis(x, data):
    x_minus_mu = x - tf.math.reduce_mean(data,axis=0)
    cov = tf_cov(data)
    inv_covmat = tf.linalg.inv(cov)
    left_term = K.dot(x_minus_mu, inv_covmat)
    mahal = K.dot(left_term, tf.transpose(x_minus_mu))
    return mahal

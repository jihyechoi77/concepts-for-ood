# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD

import numpy as np
from numpy.random import seed
#seed(0)
#tf.random.set_seed(0)
from sklearn.ensemble import RandomForestRegressor

import ipca_v2

def logistic_regression():
    model = Sequential()
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def simple_NN(num_layer=2, activation='sigmoid'):
    # build simple neural network
    # param num_layer = 1 or 2

    model = Sequential()
    if num_layer == 1:
        model.add(layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
    elif num_layer == 2:
        model.add(layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2()))
    
    model.add(layers.Dropout(0.5))

    if activation == 'softmax':
        model.add(layers.Dense(2, activation='softmax'))
    else:
        model.add(layers.Dense(1, activation='sigmoid' if activation=='sigmoid' else None))
    
    return model


def random_forest():
    model = RandomForestRegressor(n_estimators=500, 
                                #max_depth=100,
                                n_jobs=5, random_state=0,
                                verbose=1)

    return model


def topic_model(features, result_dir='resuts/Animals_with_Attributes2/', activation='softmax'):
    if args.result_dir == 'results/Animals_with_Attributes2/':
        topic_model, _, _, \
             _, _, _ = ipca_v2.topic_model_new(predict_model, in_test_features,_,_,_,
                                                70, thres=0.2, load=False)
    else:
        topic_model = ipca_v2.TopicModel_V2(in_test_features, 70, 0.2) #70: N_CONCEPTS before removing the duplicates
        topic_model(in_test_features)
    topic_model.load_weights(args.result_dir+'/latest_topic.h5', by_name=True)

    model = Sequential()
    for layer in topic_model.layers[:-1]:
        model.add(layer)
        model.layers[-1].trainable = False

    if activation == 'softmax':
        model.add(layers.Dense(2, activation='softmax'))
    else:
        model.add(layers.Dense(1, activation='sigmoid' if activation=='sigmoid' else None))

    return model

class InceptionV3Energy(keras.Model):
    def compile(self, optimizer, my_loss):
        super().compile(optimizer)
        self.my_loss = my_loss
        self.optimizer=optimizer
        self.compiled_metrics = keras.metrics.CategoricalAccuracy()

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        # print(data)
        x, y_IN = data
        x_IN = x[0]
        x_OUT = x[1]

        with tf.GradientTape() as tape:
            logits_IN = self(x_IN, training=True)  # Forward pass
            logits_OUT = self(x_OUT, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.my_loss(y_IN, logits_IN, logits_OUT)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_IN, logits_IN)

        # return {m.name: m.result() for m in self.metrics}
        return {"loss": loss, "acc": self.compiled_metrics.result()}

    """
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.compiled_metrics]
    """

def prepare_InceptionV3(modelpath, input_size=(224,224), pretrain=False, return_model=False):

    # tf.random.set_seed(1)
    input_tensor = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
    resized_images = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
    base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                                                   include_top=False,
                                                   input_tensor=resized_images,
                                                   pooling='max')
    for layer in base_model.layers:
        layer.trainable = False
    output_from_model = base_model.layers[-2].output #mixed10
    global_pool = base_model.layers[-1]
    global_pool_out = global_pool(output_from_model)

    flatten_out = layers.Flatten()(global_pool_out)
    fc1 = layers.Dense(units=256, activation='relu',
                        # kernel_initializer=tf.keras.initializer.he_normal(),
                        kernel_regularizer=tf.keras.regularizers.l2())
    fc1_out = fc1(flatten_out)
    dropout = layers.Dropout(0.5)
    dropout_out = dropout(fc1_out)
    fc2 = layers.Dense(units=50, # 50 classes
                      activation=None,
                      kernel_regularizer=tf.keras.regularizers.l2())
    output_tensor = fc2(dropout_out)  # NOTE: logits NOT softmax output!!!!!

    if return_model:
        model = tf.keras.models.Model(inputs=input_tensor, outputs=output_tensor)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy', metrics=['accuracy'])
        if pretrain:
            model.load_weights(modelpath, by_name=True)

        return model

    else:
        return input_tensor, output_tensor

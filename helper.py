"""Helper file to run the discover concept algorithm"""
# lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import splitfolders
from absl import app
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
# from keras.layers import Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import matplotlib
# matplotlib.use('GTK3Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from PIL import Image
from matplotlib import cm
seed(0)
# tf.compat.v1.set_random_seed(0)
tf.random.set_seed(0)


def copy_save_image(x_filename,f1,f2,a,b):
  # open the image
  # Image1 = Image.fromarray(x.astype('uint8'))
  Image1 = Image.open(x_filename)
  Image1.show()
  if f1:
    Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((224,224), Image.ANTIALIAS)
  """
  left = 32*b 
  right = left+116
  top = 32*a
  bottom = top+116
  """
  receptive_size = 40*2 #37*2
  jump = 32
  left = jump*b
  right = left+receptive_size
  top = jump*a
  bottom = top+receptive_size
  # print(left)
  # print(right)
  # print(top)
  # print(bottom)

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  new_size = (receptive_size,receptive_size) #(100,100)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  new_im.show()
  new_im.save(f2)


def copy_save_image_all(x,f1,f2,a,b):

  # open the image
  Image1 = Image.fromarray(x.astype('uint8'))
  old_size = (240,240)
  new_size = (244,244)
  new_im = Image.new("RGB", new_size)
  new_im.paste(Image1, (2,2))
  new_im.save(f2)
  '''
  Image1.save(f1)
  # make a copy the image so that
  # the original image does not get affected
  Image1copy = Image1.copy()
  Image1copy = Image1copy.resize((240,240), Image.ANTIALIAS)
  left = 32*b
  right = left+116
  top = 32*a
  bottom = top+116

  region = Image1copy.crop((left,top,right,bottom))
  #im.paste(region, (50, 50, 100, 100))
  old_size = (116,116)
  new_size = (118,118)
  new_im = Image.new("RGB", new_size)
  new_im.paste(region, (1,1))
  '''
  #Image1.save(f2)


def target_category_loss(x, category_index, nb_classes):
  return x * K.one_hot([category_index], nb_classes)


def split_keras_model(model, input_size, index):
      '''
      Input: 
        model: A pre-trained Keras Sequential model
        index: The index of the layer where we want to split the model
      Output:
        model1: From layer 0 to index
        model2: From index+1 layer to the output of the original model 
      The index layer will be the last layer of the model_1 and the same shape of that layer will be the input layer of the model_2
      '''
      # Creating the first part...
      # Get the input layer shape
      # print(model.layers[0].input_shape) # [(None, 224, 224, 3)]
      # print(model.layers[1].input_shape) # (None, 224, 224, 3)
      # input('wait')
      # layer_input_1 = tf.keras.Input(shape=model.layers[0].input_shape[0][1:])
      layer_input_1 = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
      # Initialize the model with the input layer
      x = layer_input_1
      # x = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(layer_input_1)

      # Foreach layer: connect it to the new model
      for layer in model.layers[1:index]:
            print(x)
            x = layer(x)
      # Create the model instance
      model1 = Model(inputs=layer_input_1, outputs=x)


      # Creating the second part...
      # Get the input shape of desired layer
      input_shape_2 = model.layers[index].get_input_shape_at(0)[1:] 
      print("Input shape of model 2: "+str(input_shape_2))
      # A new input tensor to be able to feed the desired layer
      layer_input_2 = Input(shape=input_shape_2) 

      # Create the new nodes for each layer in the path
      x = layer_input_2
      # Foreach layer connect it to the new model
      for layer in model.layers[index:]:
          x = layer(x)

      # create the model
      model2 = Model(inputs=layer_input_2, outputs=x)

      return (model1, model2)


def split_by_name(model, input_tensor, layer_name):
    bottom_input = input_tensor #Input(model.input_shape[1:])
    bottom_output = bottom_input
    layer = model.get_layer(layer_name)
    top_input = Input(layer.output_shape[1:])
    top_output = top_input
    bottom = True
    for layer in model.layers:
        print(layer)
        if bottom:
            bottom_output = layer(bottom_output)
        else:
            top_output = layer(top_output)
        if layer.name == layer_name:
            bottom = False

    bottom_model = Model(bottom_input, bottom_output)
    top_model = Model(top_input, top_output)

    return bottom_model, top_model


def prepare_data(datadir="./data/Animals_with_Attributes2/JPEGImages", \
                savedir="./data/Animals_with_Attributes2"):
  # install split-folders with progress visualization option: pip install split-folders tqdm
  splitfolders.ratio(datadir, output=savedir, seed=1337, ratio=(.8, .1, .1), group_prefix=None) # default values


def prepare_inceptionV3(input_size=(224,224),
                        modelname='results/Animals_with_Attributes2/inceptionv3_AwA2.h5'
                       ):
  tf.random.set_seed(1)
  input_tensor = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
  resized_images = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
  base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                         include_top=False,
                         input_tensor=resized_images,
                         # input_shape=(input_size[0], input_size[1], 3),
                         pooling='max')
  for layer in base_model.layers:
      layer.trainable = False
  output_from_model = base_model.layers[-2].output
  global_pool = base_model.layers[-1]
  global_pool_out = global_pool(output_from_model)
  flatten_out = layers.Flatten()(global_pool_out)
  fc1 = layers.Dense(units=256,
                    activation='relu',
                    # kernel_initializer=tf.keras.initializer.he_normal(),
                    kernel_regularizer=tf.keras.regularizers.l2())
  fc1_out = fc1(flatten_out)
  dropout = layers.Dropout(0.5)
  dropout_out = dropout(fc1_out)

  fc2 = layers.Dense(units=50, # 50 classes
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2())
  output_tensor = fc2(dropout_out)  # NOTE: logits NOT softmax output!!!!!
  softmax = layers.Activation('softmax')
  softmax_out = softmax(output_tensor)

  """
  mixed10_2 = Input(shape=(5,5,2048), name='input_2')
  # output_from_model_2 = base_model.layers[-1](mixed10_2)
  # flatten_out_2 = layers.Flatten()(output_from_model_2)
  flatten_out_2 = layers.Flatten()(mixed10_2)
  fc1_out_2 = fc1(flatten_out_2)
  dropout_out_2 = dropout(fc1_out_2)
  output_tensor_2 = softmax(dropout_out_2)
  """

  # model = tf.keras.models.Model(inputs=base_model.input, outputs=output_tensor)
  model = tf.keras.models.Model(inputs=input_tensor, outputs=softmax_out)
  #return model, input_tensor, output_from_model, mixed10_2, output_tensor_2
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  model.load_weights(modelname, by_name=True)

  return model


def load_model_inception_new(train_generator, val_generator, pretrain=True, n_gpus=0,\
               modelname='results/Animals_with_Attributes2/inceptionv3_AwA2.h5', \
               batch_size=256, input_size=(224,224), split_idx=-5):

  # tf.set_random_seed(1)
  tf.random.set_seed(1)
  input_tensor = tf.keras.Input(shape=(input_size[0], input_size[1], 3))
  if not input_size[0] == 224:
      #input_tensor = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
      resized_image = layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
      #resized_images = tf.keras.applications.inception_v3.preprocess_input(resized_images)
  base_model = tf.keras.applications.InceptionV3(weights='imagenet',
                                         include_top=False,
                                         input_tensor=input_tensor if input_size[0] == 224 else resized_image,
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
  

  """
  softmax = layers.Dense(units=50, # 50 classes
          activation='softmax',
          # kernel_initializer=tf.keras.initializer.he_normal(),
          kernel_regularizer=tf.keras.regularizers.l2())
  output_tensor = softmax(dropout_out)
  """
    
  fc2 = layers.Dense(units=50, # 50 classes
                    activation=None,
                    kernel_regularizer=tf.keras.regularizers.l2())
  output_tensor = fc2(dropout_out)  # NOTE: logits NOT softmax output!!!!!
  softmax = layers.Activation('softmax')
  softmax_out = softmax(output_tensor)

  mixed10_2 = Input(shape=(5,5,2048), name='input_2')
  global_pool_out_2 = global_pool(mixed10_2)
  flatten_out_2 = layers.Flatten()(global_pool_out_2)
  fc1_out_2 = fc1(flatten_out_2)
  dropout_out_2 = dropout(fc1_out_2) 
  # output_tensor_2 = softmax(dropout_out_2)
  output_tensor_2 = fc2(dropout_out_2)

  model = tf.keras.models.Model(inputs=input_tensor, outputs=softmax_out)
  # print('\n\noriginal model to be trained')
  # print(model.summary())

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='categorical_crossentropy', metrics=['accuracy'])
  

  if pretrain:
    model.load_weights(modelname, by_name=True)
    
    #### check accuracy of the trained model
    loss_val, acc_val = model.evaluate(val_generator)
    print('Loss of the trained original model: '+str(loss_val))
    print('Accuracy of the trained original model: '+str(acc_val))

  else:
    # model.load_weights(modelname, by_name=True)
    _ = model.fit(
        train_generator if not n_gpus else train_dataset,
        validation_data=val_generator if not n_gpus else val_dataset,
        epochs=20,
        verbose=1,
        shuffle=True,
        steps_per_epoch=len(train_generator.filenames)//batch_size)
    model.save_weights(modelname)

  for layer in model.layers:
    layer.trainable = False
  
  #feature_model, predict_model = split_by_name(model, input_tensor, 'mixed2') 
  #print(feature_model.summary())
  #input()
  #print(predict_model.summary())
  #input()
  # feature_model, predict_model = split_keras_model(model, input_size, index=split_idx)
  # feature_model = Model(inputs=model.input, outputs=model.get_layer('mixed10').output)
  feature_model = Model(inputs=input_tensor, outputs=output_from_model)
  # predict_model = Model(inputs=[model.input, model.get_layer('global_max_pooling2d').input], outputs=model.output)
  predict_model = Model(inputs=mixed10_2, outputs=output_tensor_2)
  predict_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='categorical_crossentropy', metrics=['accuracy'])
  #print(feature_model.summary())
  print(predict_model.summary())

  
  return feature_model, predict_model



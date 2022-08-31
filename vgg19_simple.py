# Based on https://github.com/narcomey/mri-superresolution/blob/master/model_1/model.py
import os
import tensorflow as tf
import keras
import model_builder
import metrics

def _get_spatial_ndim(x):
    return keras.backend.ndim(x) - 2

def _conv(x, num_filters, kernel_size, padding='same', **kwargs):
    n = _get_spatial_ndim(x)
    if n not in (2, 3):
        raise NotImplementedError(f'{n}D convolution is not supported')

    return (keras.layers.Conv2D if n == 2 else
            keras.layers.Conv3D)(
                num_filters, kernel_size, padding=padding, **kwargs)(x)

def vgg19_model(input_shape, reuse):
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        print("Building Vgg19 model for grayscale")
        inputs = keras.layers.Input(input_shape)
        n = _get_spatial_ndim(inputs)
        #conv1
        x = _conv(inputs,64,3,activation = 'relu',name='conv1_1')
        x = _conv(x,64,3,activation = 'relu',name='conv1_2')
        x = (keras.layers.MaxPool2d if n == 2 else keras.layers.MaxPool3d)(pool_size=2,stides=2,padding='same',name='pool1')(x) # (batch_size, 128, 128, 64)
        #conv2
        x = _conv(x,128,3,activation = 'relu',name='conv2_1')
        x = _conv(x,128,3,activation = 'relu',name='conv2_2')
        x = (keras.layers.MaxPool2d if n == 2 else keras.layers.MaxPool3d)(pool_size=2,stides=2,padding='same',name='pool2')(x) # (batch_size, 64, 64, 128)
        #conv3
        x = _conv(x,256,3,activation = 'relu',name='conv3_1')
        x = _conv(x,256,3,activation = 'relu',name='conv3_2')
        x = _conv(x,256,3,activation = 'relu',name='conv3_3')
        x = _conv(x,256,3,activation = 'relu',name='conv3_4')
        x = (keras.layers.MaxPool2d if n == 2 else keras.layers.MaxPool3d)(pool_size=2,stides=2,padding='same',name='pool3')(x) # (batch_size, 32, 32, 256)
        #conv4
        x = _conv(x,512,3,activation = 'relu',name='conv4_1')
        x = _conv(x,512,3,activation = 'relu',name='conv4_2')
        x = _conv(x,512,3,activation = 'relu',name='conv4_3')
        x = _conv(x,512,3,activation = 'relu',name='conv4_4')
        x = (keras.layers.MaxPool2d if n == 2 else keras.layers.MaxPool3d)(pool_size=2,stides=2,padding='same',name='pool4')(x) # (batch_size, 16, 16, 512)
        #conv5
        x = _conv(x,512,3,activation = 'relu',name='conv5_1')
        x = _conv(x,512,3,activation = 'relu',name='conv5_2')
        x = _conv(x,512,3,activation = 'relu',name='conv5_3')
        x = _conv(x,512,3,activation = 'relu',name='conv5_4')
        x = (keras.layers.MaxPool2d if n == 2 else keras.layers.MaxPool3d)(pool_size=2,stides=2,padding='same',name='pool4')(x) # (batch_size, 8, 8, 512)
        conv = x
        #Fully Connected layers
        x = keras.layers.Flatten(name='flatten')(x)
        x = keras.layers.Dense(units=4096,activation='relu',name='fc6')(x)
        x = keras.layers.Dense(units=4096,activation='relu',name='fc7')(x)
        x = keras.layers.Dense(units=1000,activation=tf.identity,name='fc8')(x)
        model = keras.Model(inputs,x,name='VGG19_Simple')
        return model, conv
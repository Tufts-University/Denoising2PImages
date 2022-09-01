# Based on: https://medium.com/analytics-vidhya/implementing-srresnet-srgan-super-resolution-with-tensorflow-89900d2ec9b2
# Paper: https://arxiv.org/pdf/1609.04802.pdf (SRGAN)
# Pytorch Impl: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py

import os
import tensorflow as tf
import keras
import model_builder
import metrics
import vgg19_simple
import train
import shutil
import pathlib
import numpy as np

# === SRResNet ===
def _get_spatial_ndim(x):
    return keras.backend.ndim(x) - 2

def _get_num_channels(x):
    return keras.backend.int_shape(x)[-1]

def _conv(x, num_filters, kernel_size, padding='same', **kwargs):
    n = _get_spatial_ndim(x)
    if n not in (2, 3):
        raise NotImplementedError(f'{n}D convolution is not supported')

    return (keras.layers.Conv2D if n == 2 else
            keras.layers.Conv3D)(
                num_filters, kernel_size, padding=padding, **kwargs)(x)

def _residual_blocks(x, repeat):
  num_channels = _get_num_channels(x)

  for _ in range(repeat):
    short_skip = x
    x = _conv(x,num_channels,3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.PReLU()(x)
    x = _conv(x,num_channels,3)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([x, short_skip])
  return x

def _residual_disc_blocks(x):
  num_channels = _get_num_channels(x)
  channels = [num_channels * n for n in range(1,5)]
  print(channels)

  x = _conv(x,num_channels,3,strides = 2)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.LeakyReLU()(x)
  
  for i in range(len(channels)):
    x = _conv(x,channels[i],3,strides = 1)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = _conv(x,channels[i],3,strides = 2)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
  return x

# Build a generator model
def build_generator_model(input_shape = (50,256,256,1),
                          *,
                          num_channels,
                          num_residual_blocks,
                          num_channel_out =1):
  print('=== Building Generator Model --------------------------------------------')
  inputs = keras.layers.Input(input_shape)
  x = _conv(inputs, num_channels, 3)
  x = keras.layers.PReLU()(x)
  long_skip = x

  x = _residual_blocks(x,num_residual_blocks)
  x = _conv(x,num_channels,3)
  x = keras.layers.BatchNormalization(axis=-1)(x)
  x = keras.layers.Add()([x,long_skip])

  x = _conv(x,num_channel_out,3)
  # TODO(nvora01): Not a fan of this, but VGG needs 3 channels
  #outputs = keras.layers.Concatenate()([x,x,x])
  model = keras.Model(inputs,x, name='Generator')
  print('--------------------------------------------------------------------')

  return model

# Build a discriminator model
def build_discriminator_model(input_shape = (50,256,256,1),
                          *,
                          num_channels,
                          num_residual_blocks,
                          num_channel_out =1):
  print('=== Building Discriminator Model --------------------------------------------')
  inputs = keras.layers.Input(input_shape)
  x = _conv(inputs, num_channels, 3)
  x = keras.layers.LeakyReLU()(x)
  
  
  x = _residual_disc_blocks(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(1024)(x)
  x = keras.layers.LeakyReLU()(x)
  outputs = keras.layers.Dense(1,activation='sigmoid')(x)

  model = keras.Model(inputs,outputs,name='Discriminator')
  print('--------------------------------------------------------------------')

  return model

# Losses

# def build_vgg19(input_shape = (50,256,256,1),reuse=False):
#     model = vgg19_simple.vgg19_model(input_shape,reuse)
#     return model

def build_and_compile_srgan(config):
    learning_rate = config['initial_learning_rate']
    generator = build_generator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out = 1)
    
    generator = model_builder.compile_model(generator, learning_rate, 'mse', config['metrics'])
    
    discriminator = build_discriminator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out =1)
    discriminator.summary()

    return generator, discriminator

learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

@tf.function
def train_step(images,srgan_checkpoint,CARE):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr = images[0]
        hr = images[1]
        sr = srgan_checkpoint.generator(lr, training=True)
        hr_output = srgan_checkpoint.discriminator(hr, training=True)
        sr_output = srgan_checkpoint.discriminator(sr, training=True)

        con_loss = metrics.calculate_content_loss(hr, sr, CARE)
        gen_loss = metrics.calculate_generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = metrics.calculate_discriminator_loss(hr_output, sr_output)

    gradients_of_generator = gen_tape.gradient(perc_loss, srgan_checkpoint.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, srgan_checkpoint.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, srgan_checkpoint.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, srgan_checkpoint.discriminator.trainable_variables))

    return perc_loss, disc_loss

def generator_train(generator, model_name, config, output_dir, training_data, validation_data, initial_path):
    generator = train.fit_model(generator, model_name, config, output_dir,training_data, validation_data)
    model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
    assert len(model_paths) != 0, f'No models found under {output_dir}'
    latest = max(model_paths, key=os.path.getmtime)
    final_weights_path = str(initial_path /pathlib.Path(output_dir) / 'Pretrained.hdf5')
    source = str(initial_path / pathlib.Path(output_dir) / latest)
    print(f'Location of source file: "{source}"')
    print(f'Location of Final Weights file: "{final_weights_path}"')
    shutil.copy(source, final_weights_path)
    print(f'Pretrained Weights are saved to: "{final_weights_path}"')

    return generator, final_weights_path
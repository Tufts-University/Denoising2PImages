# Based on: https://medium.com/analytics-vidhya/implementing-srresnet-srgan-super-resolution-with-tensorflow-89900d2ec9b2
# Paper: https://arxiv.org/pdf/1609.04802.pdf (SRGAN)
# Pytorch Impl: https://github.com/Lornatang/SRGAN-PyTorch/blob/main/model.py

import os
import tensorflow as tf
import keras
import model_builder
import metrics

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
  outputs = keras.layers.Concatenate()([x,x,x])
  model = keras.Model(inputs,outputs, name='Generator')
  print('--------------------------------------------------------------------')

  return model

# Build a discriminator model
def build_discriminator_model(input_shape = (50,256,256,3),
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

def build_vgg19(GT_shape=(256, 256, 3)):
    vgg = tf.keras.applications.VGG19(
        weights="imagenet", include_top=False, input_shape=GT_shape
    )
    block3_conv4 = 10
    block5_conv4 = 20
    
    model = keras.Model(
        inputs=vgg.inputs, outputs=vgg.layers[block5_conv4].output,
        name="vgg19"
    )
    model.trainable = False

    return model


def build_gan(generator, discriminator, vgg, raw, gt):
    gen_img = generator(raw)
    gen_features = vgg(gen_img)

    discriminator.trainable = False
    validity = discriminator(gen_img)
    
    return tf.keras.Model(
        inputs=[raw, gt], outputs=[validity, gen_features],
        name="gan")

def build_and_compile_srgan(config):
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])
    generator = build_generator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out = 1)
    
    generator.summary()
    generator = model_builder.compile_model(generator, learning_rate, 'mse', config['metrics'])
    
    discriminator = build_discriminator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out =1)
    discriminator.summary()
    discriminator = model_builder.compile_model(discriminator, learning_rate, 'mse', config['metrics'])
    
    vgg = build_vgg19(GT_shape=(256, 256, 3))
    return generator, discriminator, vgg 
    
@tf.function()
def train_step(data, generator,discriminator):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr, hr = data
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        sr = generator(lr, training=True)

        hr_output = discriminator([hr,hr,hr], training=True)
        sr_output = discriminator(sr, training=True)

        con_loss = metrics.calculate_content_loss(hr, sr)
        gen_loss = metrics.calculate_generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = metrics.calculate_discriminator_loss(hr_output, sr_output)

    gradients_of_generator = gen_tape.gradient(perc_loss, srgan_checkpoint.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, srgan_checkpoint.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, srgan_checkpoint.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, srgan_checkpoint.discriminator.trainable_variables))

    return perc_loss, disc_loss


# TODO(nvora01): DELETE?
# def build_and_compile_gan(config):
#     # Models
#     generator = build_generator_model((*config['input_shape'], 1),
#                 num_channels=config['num_channels'],
#                 num_residual_blocks=config['num_residual_blocks'],
#                 num_channel_out = 1)
    
#     generator.summary()

#     discriminator = build_discriminator_model((*config['input_shape'], 3),
#                 num_channels=config['num_channels'],
#                 num_residual_blocks=config['num_residual_blocks'],
#                 num_channel_out =1)
#     discriminator.summary()
#     vgg = build_vgg19((256, 256, 3))
#     vgg.summary()

#     # Build the GAN
#     raw = tf.keras.layers.Input(shape=(256, 256, 1))
#     gt = tf.keras.layers.Input(shape=(256, 256, 1))
#     gan = build_gan(
#         generator, discriminator, vgg, raw, gt
#         )
#     gan.summary()

#     # Compile
#     gan_opt = tf.keras.optimizers.Adam(beta_1=0.5, beta_2=0.99)
#     gan.compile(
#         loss=["binary_crossentropy", "mse"], 
#         loss_weights=[1e-3, 1],
#         optimizer=gan_opt,
#         )
#     return gan

# Training

# TODOs:
#
# - Add model in main.py
#
# - Compare to my training function
# - Modify relevant functions very prototypically (prolly fit_model)
# - Find data generator sizes & discriminator sizes
#   - Print final output from generator 
#   - Prolly same as input but check conv to see if it makes sense
# - Run to debug on cluster
#
@tf.function()
def train_step(data,config,loss_func=metrics.mse,adv_learning=True,evaluate=['PSNR'],adv_ratio=0.001):
    logs={}
    gen_loss,disc_loss=0,0
    low_resolution,high_resolution=data
    generator_optimizer=tf.keras.optimizers.Adam(config['initial_learning_rate'])
    discriminator_optimizer=tf.keras.optimizers.Adam(config['initial_learning_rate'])
    SRResnet = build_generator_model((*config['input_shape'], 1),
                    num_channels=config['num_channels'],
                    num_residual_blocks=config['num_residual_blocks'],
                    num_channel_out = 1)
    DisNet = build_discriminator_model((*config['input_shape'], 1),
                        num_channels=config['num_channels'],
                        num_residual_blocks=config['num_residual_blocks'],
                        num_channel_out = 1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        super_resolution = SRResnet(low_resolution, training=True)
        gen_loss=loss_func(high_resolution,super_resolution)
        logs['reconstruction']=gen_loss
        if adv_learning:
            real_output = DisNet(high_resolution, training=True)
            fake_output = DisNet(super_resolution, training=True)
            
            adv_loss_g = metrics.generator_loss(fake_output) * adv_ratio
            gen_loss += adv_loss_g
            
            disc_loss = metrics.discriminator_loss(real_output, fake_output)
            logs['adv_g']=adv_loss_g
            logs['adv_d']=disc_loss
    gradients_of_generator = gen_tape.gradient(gen_loss, SRResnet.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, SRResnet.trainable_variables))
    
    if adv_learning:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, DisNet.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, DisNet.trainable_variables))
    
    for x in evaluate:
        if x=='psnr':
            logs[x]=metrics.psnr(high_resolution,super_resolution)
        elif x == 'ssim':
            logs[x]=metrics.ssim(high_resolution,super_resolution)

    return logs
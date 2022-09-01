# Custom RESNET based on RCAN and SRGAN Code
import tensorflow as tf
import keras
import model_builder

# === ResNet ===
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

  outputs = _conv(x,num_channel_out,3)
  model = keras.Model(inputs, outputs, name='Generator')
  print('--------------------------------------------------------------------')

  return model

def build_and_compile_RESNET(config):
    generator = build_generator_model((*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_channel_out = 1)

    generator = model_builder.compile_model(generator, config['initial_learning_rate'], config['loss'], config['metrics'])
    return generator
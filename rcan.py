import numpy as np
import keras
from keras import backend as kb
import tensorflow as tf

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


def _global_average_pooling(x):
    n = _get_spatial_ndim(x)
    if n == 2:
        return keras.layers.GlobalAveragePooling2D()(x)
    elif n == 3:
        return keras.layers.GlobalAveragePooling3D()(x)
    else:
        raise NotImplementedError(
            f'{n}D global average pooling is not supported')


def _channel_attention_block(x, reduction):
    '''
    Channel attention block.
    References
    ----------
    - Squeeze-and-Excitation Networks
      https://arxiv.org/abs/1709.01507
    - Image Super-Resolution Using Very Deep Residual Channel Attention
      Networks
      https://arxiv.org/abs/1807.02758
    '''

    num_channels = _get_num_channels(x)

    y = _global_average_pooling(x)
    y = keras.layers.Reshape((*(1,) * _get_spatial_ndim(x), num_channels))(y)
    y = _conv(y, num_channels // reduction, 1, activation='relu')
    y = _conv(y, num_channels, 1, activation='sigmoid')

    return keras.layers.Multiply()([x, y])


def _residual_channel_attention_blocks(x,
                                       repeat=1,
                                       channel_reduction=8,
                                       residual_scaling=1.0):
    num_channels = _get_num_channels(x)

    for _ in range(repeat):
        skip = x

        x = _conv(x, num_channels, 3, activation='relu')
        x = _conv(x, num_channels, 3)

        x = _channel_attention_block(x, channel_reduction)

        if residual_scaling != 1.0:
            x = keras.layers.Lambda(lambda x: residual_scaling * x)(x)

        x = keras.layers.Add()([x, skip])

    return x


def normalize(image, p_min=2, p_max=99.9, dtype='float32'):
    '''
    Normalizes the image intensity so that the `p_min`-th and the `p_max`-th
    percentiles are converted to 0 and 1 respectively.
    References
    ----------
    Content-Aware Image Restoration: Pushing the Limits of Fluorescence
    Microscopy
    https://doi.org/10.1038/s41592-018-0216-7
    '''
    low, high = np.percentile(image, (p_min, p_max))
    return tf.cast((image - low) / (high - low + 1e-6), dtype)


def _standardize(x):
    '''
    Standardize the signal so that the range becomes [-1, 1] (assuming the
    original range is [0, 1]).
    '''
    prefix = 'lambda_standardize'
    name = prefix + '_' + str(keras.backend.get_uid(prefix))
    return keras.layers.Lambda(lambda x: 2 * x - 1, name=name)(x)


def _destandardize(x):
    '''Undo standardization'''
    prefix = 'lambda_destandardize'
    name = prefix + '_' + str(keras.backend.get_uid(prefix))
    return keras.layers.Lambda(lambda x: 0.5 * x + 0.5, name=name)(x)


def build_rcan(input_shape=(16, 256, 256, 1),
               *,
               num_channels,
               num_residual_blocks,
               num_residual_groups,
               channel_reduction,
               residual_scaling=1.0,
               num_output_channels=-1):
    '''
    Builds a residual channel attention network. Note that the upscale module
    at the end of the network is omitted so that the input and output of the
    model have the same size.
    Parameters
    ----------
    input_shape: tuple of int
        Input shape of the model.
    num_channels: int
        Number of feature channels.
    num_residual_blocks: int
        Number of residual channel attention blocks in each residual group.
    num_residual_groups: int
        Number of residual groups.
    channel_reduction: int
        Channel reduction ratio for channel attention.
    residual_scaling: float
        Scaling factor applied to the residual component in the residual
        channel attention block.
    num_output_channels: int
        Number of channels in the output image. if negative, it is set to the
        same number as the input.
    Returns
    -------
    keras.Model
        Keras model instance.
    References
    ----------
    Image Super-Resolution Using Very Deep Residual Channel Attention Networks
    https://arxiv.org/abs/1807.02758
    '''

    print('=== Building RCAN Model --------------------------------------------')

    config = {
        'input_shape': input_shape,
        'num_channels': num_channels,
        'num_residual_blocks': num_residual_blocks,
        'num_residual_groups': num_residual_groups,
        'channel_reduction': channel_reduction,
        'residual_scaling': residual_scaling,
        'num_output_channels': num_output_channels,
    }
    print(f'Using config: {config}\n')

    if num_output_channels < 0:
        num_output_channels = input_shape[-1]

    inputs = keras.layers.Input(input_shape)
    x = _standardize(inputs)
    x = _conv(x, num_channels, 3)

    long_skip = x

    for _ in range(num_residual_groups):
        short_skip = x

        x = _residual_channel_attention_blocks(
            x,
            num_residual_blocks,
            channel_reduction,
            residual_scaling)

        if num_residual_groups == 1:
            break

        x = _conv(x, num_channels, 3)
        x = keras.layers.Add()([x, short_skip])

    x = _conv(x, num_channels, 3)
    x = keras.layers.Add()([x, long_skip])

    x = _conv(x, num_output_channels, 3)
    outputs = _destandardize(x)

    model = keras.Model(inputs, outputs)

    print('--------------------------------------------------------------------')

    return model


def backend_channels_last():
    assert kb.image_data_format() in ('channels_first', 'channels_last')
    return kb.image_data_format() == 'channels_last'


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)

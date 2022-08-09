import os
import numpy as np
from matplotlib import pyplot as plt
import tifffile as tiff
import re
import zipfile
import scipy.io
import random
from tifffile import imread
import argparse
import json
import pickle as pkl
import imageio
import itertools
import jsonschema
import keras
import pathlib
from tensorflow.image import ssim as ssim2
from tqdm.utils import IS_WIN
from tqdm.keras import TqdmCallback as _TqdmCallback
import functools
import warnings
import tqdm
import keras.backend as K
import tensorflow as tf
import collections
from importlib import import_module
from tensorflow import __version__ as _tf_version
from tensorflow.python.client.device_lib import list_local_devices
from packaging import version
from keras.utils.conv_utils import normalize_tuple
import tensorflow_probability as tfp
import fractions
import numexpr
import pywt

###################### Modify to for data/model changes ########################

# !!!: Make sure to change batch file too.

# Points to the model folder.
model_name = 'FAD_model_0629_cervix_SSIM' 

# Points to the path of the data for NADH.
data_path_nadh = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_NADH_healthy.npz' 

# Points to the path of the data for FAD
data_path_fad = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_FAD_healthy.npz'

# Whether this model works on wavelet-transformed data.
wavelet_model = False

################################################################################
        
def build_rcan(input_shape=(16, 256, 256, 1),
               *,
               num_channels=32,
               num_residual_blocks=3,
               num_residual_groups=5,
               channel_reduction=8,
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

    if num_output_channels < 0:
        num_output_channels = input_shape[-1]

    inputs = keras.layers.Input(input_shape)
    #x = _normalize(inputs)
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

    return keras.Model(inputs, outputs)
    
def load_training_data(file, validation_split=0, axes=None, n_images=None, verbose=False):
    """Load training data from file in ``.npz`` format.
    The data file is expected to have the keys:
    - ``X``    : Array of training input images.
    - ``Y``    : Array of corresponding target images.
    - ``axes`` : Axes of the training images.
    Parameters
    ----------
    file : str
        File name
    validation_split : float
        Fraction of images to use as validation set during training.
    axes: str, optional
        Must be provided in case the loaded data does not contain ``axes`` information.
    n_images : int, optional
        Can be used to limit the number of images loaded from data.
    verbose : bool, optional
        Can be used to display information about the loaded images.
    Returns
    -------
    tuple( tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`), str )
        Returns two tuples (`X_train`, `Y_train`), (`X_val`, `Y_val`) of training and validation sets
        and the axes of the input images.
        The tuple of validation data will be ``None`` if ``validation_split = 0``.
    """

    f = np.load(file)
    X, Y = f['X'], f['Y']
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)

    assert X.ndim == Y.ndim
    assert len(axes) == X.ndim
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y = X[:n_images], Y[:n_images]

    if validation_split > 0:
        #n_val   = int(round(n_images * validation_split))
        ''' NEED TO SET VALIDATION SPLIT BETTER '''
        n_val = 620
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:],  Y[-n_val:]
        X,   Y   = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val

    data_val = (X_t,Y_t) if validation_split > 0 else None

    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split>0 else 0
        image_size = tuple( X.shape[ax[a]] for a in axes if a in 'TZYX' )
        n_dim = len(image_size)

        print('number of training images:\t', n_train)
        print('number of validation images:\t', n_val)
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)

    return (X,Y), data_val, axes

def apply(model, data, overlap_shape=None, verbose=False):
    '''
    Applies a model to an input image. The input image stack is split into
    sub-blocks with model's input size, then the model is applied block by
    block. The sizes of input and output images are assumed to be the same
    while they can have different numbers of channels.
    Parameters
    ----------
    model: keras.Model
        Keras model.
    data: array_like or list of array_like
        Input data. Either an image or a list of images.
    overlap_shape: tuple of int or None
        Overlap size between sub-blocks in each dimension. If not specified,
        a default size ((32, 32) for 2D and (2, 32, 32) for 3D) is used.
        Results at overlapped areas are blended together linearly.
    Returns
    -------
    ndarray
        Result image.
    '''

    model_input_image_shape = tuple(model.input.shape.as_list()[1:-1])
    model_output_image_shape = tuple(model.output.shape.as_list()[1:-1])

    if len(model_input_image_shape) != len(model_output_image_shape):
        raise NotImplementedError

    image_dim = len(model_input_image_shape)
    num_input_channels = model.input.shape[-1]
    num_output_channels = model.output.shape[-1]

    scale_factor = tuple(
        fractions.Fraction(o, i) for i, o in zip(
            model_input_image_shape, model_output_image_shape))

    def _scale_tuple(t):
        t = [v * f for v, f in zip(t, scale_factor)]

        if not all([v.denominator == 1 for v in t]):
            raise NotImplementedError

        return tuple(v.numerator for v in t)

    def _scale_roi(roi):
        roi = [slice(r.start * f, r.stop * f)
               for r, f in zip(roi, scale_factor)]

        if not all([
                r.start.denominator == 1 and
                r.stop.denominator == 1 for r in roi]):
            raise NotImplementedError

        return tuple(slice(r.start.numerator, r.stop.numerator) for r in roi)

    if overlap_shape is None:
        if image_dim == 2:
            overlap_shape = (32, 32)
        elif image_dim == 3:
            overlap_shape = (2, 32, 32)
        else:
            raise NotImplementedError
    elif len(overlap_shape) != image_dim:
        raise ValueError(f'Overlap shape must be {image_dim}D; '
                         f'Received shape: {overlap_shape}')

    step_shape = tuple(
        m - o for m, o in zip(
            model_input_image_shape, overlap_shape))

    block_weight = np.ones(
        [m - 2 * o for m, o
         in zip(model_output_image_shape, _scale_tuple(overlap_shape))],
        dtype=np.float32)

    block_weight = np.pad(
        block_weight,
        [(o + 1, o + 1) for o in _scale_tuple(overlap_shape)],
        'linear_ramp'
    )[(slice(1, -1),) * image_dim]

    batch_size = model.gpus if is_multi_gpu_model(model) else 1
    batch = np.zeros(
        (batch_size, *model_input_image_shape, num_input_channels),
        dtype=np.float32)

    if isinstance(data, (list, tuple)):
        input_is_list = True
    else:
        data = [data]
        input_is_list = False

    result = []
    for image in data:
        # add the channel dimension if necessary
        print(image.shape)
        if len(image.shape) == image_dim:
            image = image[..., np.newaxis]

        if (len(image.shape) != image_dim + 1
                or image.shape[-1] != num_input_channels):
            raise ValueError(f'Input image must be {image_dim}D with '
                             f'{num_input_channels} channels; '
                             f'Received image shape: {image.shape}')

        input_image_shape = image.shape[:-1]
        output_image_shape = _scale_tuple(input_image_shape)

        applied = np.zeros(
            (*output_image_shape, num_output_channels), dtype=np.float32)
        sum_weight = np.zeros(output_image_shape, dtype=np.float32)

        num_steps = tuple(
            i // s + (i % s != 0)
            for i, s in zip(input_image_shape, step_shape))

        # top-left corner of each block
        blocks = list(itertools.product(
            *[np.arange(n) * s for n, s in zip(num_steps, step_shape)]))

        for chunk_index in tqdm.trange(
                0, len(blocks), batch_size, disable=not verbose,
                dynamic_ncols=True, ascii=tqdm.utils.IS_WIN):
            rois = []
            for batch_index, tl in enumerate(
                    blocks[chunk_index:chunk_index + batch_size]):
                br = [min(t + m, i) for t, m, i
                      in zip(tl, model_input_image_shape, input_image_shape)]
                r1, r2 = zip(
                    *[(slice(s, e), slice(0, e - s)) for s, e in zip(tl, br)])

                m = image[r1]
                if model_input_image_shape != m.shape[-1]:
                    pad_width = [(0, b - s) for b, s
                                 in zip(model_input_image_shape, m.shape[:-1])]
                    pad_width.append((0, 0))
                    m = np.pad(m, pad_width, 'reflect')

                batch[batch_index] = m
                rois.append((r1, r2))

            p = model.predict(batch, batch_size=batch_size)

            for batch_index in range(len(rois)):
                for channel in range(num_output_channels):
                    p[batch_index, ..., channel] *= block_weight

                r1, r2 = [_scale_roi(roi) for roi in rois[batch_index]]

                applied[r1] += p[batch_index][r2]
                sum_weight[r1] += block_weight[r2]

        for channel in range(num_output_channels):
            applied[..., channel] /= sum_weight

        if applied.shape[-1] == 1:
            applied = applied[..., 0]

        result.append(applied)

    return result if input_is_list else result[0]
    
#################################################################################

model = build_rcan(
    (*input_shape,1),
    num_channels=config['num_channels'],
    num_residual_blocks=config['num_residual_blocks'],
    num_residual_groups=config['num_residual_groups'],
    channel_reduction=config['channel_reduction'])
    
gpus = get_gpu_count()
model = convert_to_multi_gpu_model(model, gpus)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=config['initial_learning_rate']),
    loss={'mae' : mae,'mse': mse,'SSIM_loss': SSIM_loss,'MSSSIM_loss': MSSSIM_loss}[config['loss']],
    metrics=[{'psnr': psnr, 'ssim': ssim}[m] for m in config['metrics']])
    
###########################################################################
## APPLY Model ##
os.chdir(model_save_path)
if not os.path.exists(os.path.join(model_save_path,'outputs')): os.mkdir(os.path.join(model_save_path,'outputs'))

model_paths = [ model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
assert len(model_paths) != 0, f'No models found under {model_save_path}'
latest = max(model_paths, key=os.path.getmtime)
print(f'Using latest model: {latest}')

def wavelet_transform(mat):
    for i in range(len(mat)):
        C = pywt.dwt2(mat[i,:,:],'bior4.4',mode = 'periodization')
        cA,(cH,cV,cD) = C
        row = np.append(cA,cH,axis=1)
        row2 = np.append(cV,cD,axis=1)
        mat[i,:,:] = np.vstack((row,row2))
    
    return mat
    
def wavelet_inverse_transform(mat):
    for i in range(len(mat)):
        (cA,cH,cV,cD) = (mat[i, :128, :128], mat[i, :128, 128:], mat[i, 128:, :128], mat[i, 128:, 128:])
        C = cA,(cH,cV,cD)
        mat[i, :, :] = pywt.idwt2(C,'bior4.4',mode = 'periodization')
    
    return mat

model.load_weights(latest)
os.chdir(os.path.join(model_save_path,'outputs'))
stack_ranges = [[0,24],[25,74],[75,114],[115,154]]
base_name = model_name[model_name.index('_'):]


def generate_data(datatype, X_test, Y_test):
    global base_name, stack_ranges, wavelet_model
    global model
    
    
    for i in range(np.shape(stack_ranges)[0]):
        image_mat = []
        for n in range(stack_ranges[i][0],stack_ranges[i][1]+1):
            print(n)
            raw = np.reshape(X_test[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
            gt = np.reshape(Y_test[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
            
            if wavelet_model:
                X_test_input = np.copy(wavelet_transform(X_test[4*n:4*n+4]))
                X_test_input = np.reshape(X_test_input,[2, -1, 256, 256]).swapaxes(1,2).reshape(512, 512)
                restored = apply(model, X_test_input, overlap_shape=(0,0), verbose=True)
            else:
                X_test_input = raw
                restored = apply(model, X_test_input, overlap_shape=(32,32), verbose=True)
            
            
            # Inverse transform.
            if wavelet_model:
                restored = np.reshape(restored, [2, 256, 2, 256]).swapaxes(1, 2).reshape(4, 256, 256)
                restored = wavelet_inverse_transform(restored)
                restored = np.reshape(restored,[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
                
            result = [raw, restored, gt]
            result = [normalize_between_zero_and_one(m) for m in result]
            result = np.stack(result)
            result = np.clip(255 * result, 0, 255).astype('uint8')
            
            image_mat.append([result[0],result[1],result[2]]) 
        scipy.io.savemat(datatype + base_name + '_image'+ str(i) +'.mat', {'images': image_mat})
   
# Returns 256x256 X, Y matrices and apply the model.
_, (X_test, Y_test), _ = load_training_data(data_path_nadh, validation_split=0.15, axes='SXY', verbose=True)
generate_data('NADH', X_test, Y_test)

_, (X_test, Y_test), _ = load_training_data(data_path_fad, validation_split=0.15, axes='SXY', verbose=True)
generate_data("FAD", X_test, Y_test)

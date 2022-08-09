from __future__ import print_function, unicode_literals, absolute_import, division
import os
import numpy as np
import scipy.io
import itertools
import pathlib
import tqdm
import collections
from importlib import import_module
from tensorflow import __version__ as _tf_version
from tensorflow.python.client.device_lib import list_local_devices
from packaging import version
import fractions
import tensorflow as tf
from tensorflow.image import ssim as ssim2
import keras
import keras.backend as K
from csbdeep.utils import axes_dict
from csbdeep.internals import nets
from csbdeep.models import Config


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

    model_input_image_shape = (50, 256, 256, 1)[1:-1]
    model_output_image_shape = (50, 256, 256, 1)[1:-1]
    print(data.shape)
    if len(model_input_image_shape) != len(model_output_image_shape):
        raise NotImplementedError

    image_dim = len(model_input_image_shape)
    num_input_channels = model.input.shape[-1]
    num_output_channels = model.input.shape[-1]

    scale_factor = tuple(
        fractions.Fraction(int(o), int(i)) for i, o in zip(
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

    batch_size = 1
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
''' Data was Normalized in Local Prep'''

data_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_FAD_healthy.npz'
model_name = 'NADH_CAREmodel_0713_cervix_SSIML2_BS50_Deep_fs3'
main_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/'
os.chdir(main_path)

if not os.path.exists(os.path.join(main_path,model_name)): os.mkdir(os.path.join(main_path,model_name))
model_save_path = os.path.join(main_path,model_name)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(is_cuda_gpu_available)

(X,Y), (X_val,Y_val), axes = load_training_data(data_path, validation_split=0.15, axes='SXYC', verbose=True)
c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = Config(axes, n_channel_in, n_channel_out, 
                train_steps_per_epoch=100, train_batch_size=50, 
                train_epochs=300, unet_n_depth=6, 
                unet_n_first=32, unet_kern_size=3, 
                train_learning_rate=1e-05)
print(config)
                
# CALLBACKS
# learning rate reducer callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(verbose=True,factor=0.97,min_delta=0,patience=20,)

# create a callback that saves the model's weights every said amount of epochs
epoch_freq = 10
checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    str( pathlib.Path(model_save_path) / checkpoint_filepath),
    monitor='val_loss',
    save_best_only=True, verbose=1, mode='min')

# BUILD
model = build_CARes(config)
model = compile_model(model, config.train_learning_rate)
model.summary()

# EVAL
os.chdir(model_save_path)
if not os.path.exists(os.path.join(model_save_path,'outputs')): os.mkdir(os.path.join(model_save_path,'outputs'))
latest = 'weights_180_0.21952377.hdf5'
#latest = 'weights_253_0.16830541.hdf5'
model.load_weights(latest)
os.chdir(os.path.join(model_save_path,'outputs'))
stack_ranges = [[0,24],[25,74],[75,114],[115,154]]
for i in range(np.shape(stack_ranges)[0]):
    image_mat = []
    for n in range(stack_ranges[i][0],stack_ranges[i][1]+1):
        print(n)
        raw = np.reshape(X_val[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1,2).reshape(512, 512)
        gt = np.reshape(Y_val[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1,2).reshape(512, 512)
        print(raw)
        restored = apply(model, raw, overlap_shape=None, verbose=True)
        result = [raw, restored, gt]
        result = [normalize_between_zero_and_one(m) for m in result]
        result = np.stack(result)
        result = np.clip(255 * result, 0, 255).astype('uint8')
        image_mat.append([result[0],result[1],result[2]]) 
    #scipy.io.savemat(model_name + '_image'+ str(i) +'.mat', {'images': image_mat})
    scipy.io.savemat('FAD' + model_name[4:] + '_image'+ str(i) +'.mat', {'images': image_mat})
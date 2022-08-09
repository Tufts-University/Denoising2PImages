import pathlib
import numpy as np
import scipy 
import fractions
import itertools
import tqdm


# Local dependencies

import model_builder
import data_generator
import basics


def load_weights(model, output_dir):
    print('=== Loading model weights ------------------------------------------')

    model.load_weights(str(pathlib.Path(output_dir) / 'weights_final.hdf5'))

    print('--------------------------------------------------------------------')


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

    batch_size = model.gpus if basics.is_multi_gpu_model(model) else 1
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

stack_ranges = [[0,24],[25,74],[75,114],[115,154]]
def patch_and_apply(model, data_type, trial_name, wavelet_model, X_test, Y_test):
    print('=== Applying model ------------------------------------------------')
    # TODO: Add print

    for i in range(np.shape(stack_ranges)[0]):
        image_mat = []
        for n in range(stack_ranges[i][0],stack_ranges[i][1]+1):
            print(n)
            raw = np.reshape(X_test[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
            gt = np.reshape(Y_test[4*n:4*n+4],[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
            
            if wavelet_model:
                X_test_input = np.copy(data_generator.wavelet_transform(X_test[4*n:4*n+4]))
                X_test_input = np.reshape(X_test_input,[2, -1, 256, 256]).swapaxes(1,2).reshape(512, 512)
                restored = apply(model, X_test_input, overlap_shape=(0,0), verbose=True)
            else:
                X_test_input = raw
                restored = apply(model, X_test_input, overlap_shape=(32,32), verbose=True)
            
            
            # Inverse transform.
            if wavelet_model:
                restored = np.reshape(restored, [2, 256, 2, 256]).swapaxes(1, 2).reshape(4, 256, 256)
                restored = data_generator.wavelet_inverse_transform(restored)
                restored = np.reshape(restored,[2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)
                
            result = [raw, restored, gt]

            # TODO: Check if normalization is needed.
            # result = [normalize_between_zero_and_one(m) for m in result]
            result = np.stack(result)
            result = np.clip(255 * result, 0, 255).astype('uint8')
            
            image_mat.append([result[0],result[1],result[2]]) 
        scipy.io.savemat(data_type + trial_name + '_image'+ str(i) +'.mat', {'images': image_mat})    

    print('--------------------------------------------------------------------')


def _normalize_between_zero_and_one(m):
    max_val, min_val = m.max(), m.min()
    diff = max_val - min_val
    return (m - min_val) / diff if diff > 0 else np.zeros_like(m)


def eval(model_name, trial_name, config, output_dir, data_path):
    print('Evaluating...')

    # TODO: Load training data and add generate_data func.
    # Similar to 'data_generator.py'
    _, (X_val, Y_val), _ = data_generator.default_load_data(
        data_path,
        requires_channel_dim=model_name == 'care')

    if config['wavelet'] == True:
        data_generator.wavelet_transform(validation_data)

    strategy = model_builder.create_strategy()
    model = model_builder.build_and_compile_model(model_name, strategy, config)

    load_weights(model, output_dir=output_dir)

    apply_model(
        model,
        model_name=model_name,
        validation_data=validation_data)

    return model

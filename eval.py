import pathlib
import numpy as np
import scipy.io
import fractions
import itertools
import tqdm
import os


# Local dependencies

import model_builder
import data_generator
import basics


def load_weights(model, output_dir):
    print('=== Loading model weights ------------------------------------------')

    weights_path = str(pathlib.Path(output_dir) / 'weights_final.hdf5')
    print(f'Getting weights at: {weights_path}')
    
    model.load_weights(weights_path)

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

    # TODO: Check if it need to be:
    #   model.gpus if basics.is_multi_model(model) else 1
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


# The start and end indices (inclusive) of where different stacks begin and end.
stack_ranges = [[0, 24], [25, 74], [75, 114], [115, 154]]


def patch_and_apply(model, data_type, trial_name, wavelet_model, X_test, Y_test):
    print('=== Applying model ------------------------------------------------')

    print(f'Using wavelet model: {wavelet_model}')

    # The size of our stacks (how many slices are in each stack) varies
    # based on how much noise is on the top and bottom slices, as they are
    # excluded. Thus, we use the stack ranges array to group the slices from
    # the same stack into one output file.
    for (stack_index, [stack_start, stack_end]) in enumerate(stack_ranges):
        print(f'Accessing stack: {stack_index}')

        image_mat = []
        for n in range(stack_start, stack_end+1):
            print(f'Accessing slice: {n} of stack: {stack_index}')

            raw = data_generator.stitch_patches(X_test[4*n:4*n+4])
            gt = data_generator.stitch_patches(Y_test[4*n:4*n+4])

            # Apply the model to generate the restored image.
            restored = None
            if wavelet_model:
                X_test_input = data_generator.wavelet_transform(
                    np.copy(X_test[4*n:4*n+4]))
                X_test_input = data_generator.stitch_patches(X_test_input)
                restored = apply(model, X_test_input,
                                 overlap_shape=(0, 0), verbose=False)
            else:
                X_test_input = raw
                restored = apply(model, X_test_input,
                                 overlap_shape=(32, 32), verbose=False)

            # Inverse transform.
            if wavelet_model:
                restored = data_generator.patch_slice(restored)
                restored = data_generator.wavelet_inverse_transform(restored)
                restored = data_generator.stitch_patches(restored)

            result = [raw, restored, gt]

            # TODO: Check if normalization is needed.
            # result = [normalize_between_zero_and_one(m) for m in result]
            result = np.stack(result)
            result = np.clip(255 * result, 0, 255).astype('uint8')

            image_mat.append([result[0], result[1], result[2]])
        scipy.io.savemat(f'{data_type}_{trial_name}_image{stack_index}.mat', {
                         'images': image_mat})

    print('--------------------------------------------------------------------')


def _normalize_between_zero_and_one(m):
    max_val, min_val = m.max(), m.min()
    diff = max_val - min_val
    return (m - min_val) / diff if diff > 0 else np.zeros_like(m)


def eval(model_name, trial_name, config, output_dir, nadh_path, fad_path):
    print('Evaluating...')

    strategy = model_builder.create_strategy()
    model = model_builder.build_and_compile_model(model_name, strategy, config)

    load_weights(model, output_dir=output_dir)

    # Go to the results directory to generate and store evaluated images.
    results_dir = os.path.join(output_dir, 'results')
    if os.path.exists(results_dir):
        subpaths = os.listdir(results_dir)
        for subpath in subpaths:
            if nadh_path != None and subpath.lower().find('nadh') != -1:
                raise Exception(
                    f'Found existing NADH results at: {results_dir}, file: {subpath}')
            elif fad_path != None and subpath.lower().find('fad') != -1:
                raise Exception(
                    f'Found existing FAD results at: {results_dir}, file: {subpath}')
    else:
        os.mkdir(results_dir)

    if nadh_path != None:
        print('=== Evaluating NADH -----------------------------------------------')
        # Similar to 'data_generator.py'
        _, (X_val, Y_val) = data_generator.default_load_data(
            nadh_path,
            requires_channel_dim=model_name == 'care')

        print(f'Changing to directory: {results_dir}')
        os.chdir(results_dir)

        patch_and_apply(
            model, data_type='NADH', trial_name=trial_name,
            wavelet_model=config['wavelet'],
            X_test=X_val, Y_test=Y_val)
        print('--------------------------------------------------------------------')

    if fad_path != None:
        print('=== Evaluating FAD -------------------------------------------------')
        # Similar to 'data_generator.py'
        _, (X_val, Y_val) = data_generator.default_load_data(
            fad_path,
            requires_channel_dim=model_name == 'care')
            
        print(f'Changing to directory: {results_dir}')
        os.chdir(results_dir)

        patch_and_apply(
            model, data_type='FAD', trial_name=trial_name,
            wavelet_model=config['wavelet'],
            X_test=X_val, Y_test=Y_val)
        print('--------------------------------------------------------------------')

    return model

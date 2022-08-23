import wave
import numpy as np
import tensorflow as tf
import keras as keras
from keras import backend as kb
from keras.utils.conv_utils import normalize_tuple
import warnings
import collections
from csbdeep.utils import axes_dict
import pywt
import random
# Contains code for data loading and generation before
# being passed into the network.


class DataGenerator:
    '''
    Generates batches of image pairs with real-time data augmentation.
    Parameters
    ----------
    shape: tuple of int
        Shape of batch images (excluding the channel dimension).
    batch_size: int
        Batch size.
    transform_function: str or callable or None
        Function used for data augmentation. Typically you will set
        ``transform_function='rotate_and_flip'`` to apply combination of
        randomly selected image rotation and flipping.  Alternatively, you can
        specify an arbitrary transformation function which takes two input
        images (source and target) and returns transformed images. If
        ``transform_function=None``, no augmentation will be performed.
    intensity_threshold: float
        If ``intensity_threshold > 0``, pixels whose intensities are greater
        than this threshold will be considered as foreground.
    area_ratio_threshold: float between 0 and 1
        If ``intensity_threshold > 0``, the generator calculates the ratio of
        foreground pixels in a target patch, and rejects the patch if the ratio
        is smaller than this threshold.
    scale_factor: int != 0
        Scale factor for the target patch size. Positive and negative values
        mean up- and down-scaling respectively.
    '''

    def __init__(self,
                 shape,
                 batch_size,
                 transform_function='rotate_and_flip',
                 intensity_threshold=0.0,
                 area_ratio_threshold=0.0,
                 scale_factor=1):
        def rotate_and_flip(x, y, dim):
            if dim == 2:
                k = np.random.randint(0, 4)
                x, y = [np.rot90(v, k=k) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.fliplr(v) for v in (x, y)]
                return x, y
            elif dim == 3:
                k = np.random.randint(0, 4)
                x, y = [np.rot90(v, k=k, axes=(1, 2)) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.flip(v, axis=1) for v in (x, y)]
                if np.random.random() < 0.5:
                    x, y = [np.flip(v, axis=0) for v in (x, y)]
                return x, y
            else:
                raise ValueError('Unsupported dimension')

        self._shape = tuple(shape)
        self._batch_size = batch_size

        dim = len(self._shape)

        if transform_function == 'rotate_and_flip':
            if shape[-2] != shape[-1]:
                raise ValueError(
                    'Patch shape must be square when using `rotate_and_flip`; '
                    f'Received shape: {shape}')
            self._transform_function = lambda x, y: rotate_and_flip(x, y, dim)
        elif callable(transform_function):
            self._transform_function = transform_function
        elif transform_function is None:
            self._transform_function = lambda x, y: (x, y)
        else:
            raise ValueError('Invalid transform function')

        self._intensity_threshold = intensity_threshold

        if not 0 <= area_ratio_threshold <= 1:
            raise ValueError('"area_ratio_threshold" must be between 0 and 1')
        self._area_threshold = area_ratio_threshold * np.prod(shape)

        self._scale_factor = normalize_tuple(scale_factor, dim, 'scale_factor')
        if any(not isinstance(f, int) or f == 0 for f in self._scale_factor):
            raise ValueError('"scale_factor" must be nonzero integer')

    class _Sequence(keras.utils.Sequence):
        def _scale(self, shape):
            return tuple(
                s * f if f > 0 else s // -f
                for s, f in zip(shape, self._scale_factor))

        def __init__(self,
                     x,
                     y,
                     batch_size,
                     shape,
                     transform_function,
                     intensity_threshold,
                     area_threshold,
                     scale_factor):
            self._batch_size = batch_size
            self._transform_function = transform_function
            self._intensity_threshold = intensity_threshold
            self._area_threshold = area_threshold
            self._scale_factor = scale_factor

            for s, f, in zip(shape, self._scale_factor):
                if f < 0 and s % -f != 0:
                    raise ValueError(
                        'When downsampling, all elements in `shape` must be '
                        'divisible by the scale factor; '
                        f'Received shape: {shape}, '
                        f'scale factor: {self._scale_factor}')

            self._x, self._y = [
                list(m) if isinstance(m, (list, tuple)) else [m]
                for m in [x, y]]
            self._x = np.moveaxis(self._x, 0, -1)
            self._y = np.moveaxis(self._y, 0, -1)
            if len(self._x) != len(self._y):
                raise ValueError(
                    'Different number of images are given: '
                    f'{len(self._x)} vs. {len(self._y)}')

            if len({m.dtype for m in self._x}) != 1:
                raise ValueError('All source images must be the same type')
            if len({m.dtype for m in self._y}) != 1:
                raise ValueError('All target images must be the same type')
            print(len(self._x))
            for i in range(len(self._x)):
                if len(self._x[i].shape) == len(shape):
                    self._x[i] = self._x[i][..., np.newaxis]

                if len(self._y[i].shape) == len(shape):
                    self._y[i] = self._y[i][..., np.newaxis]

                if len(self._x[i].shape) != len(shape) + 1:
                    raise ValueError(f'Source image must be {len(shape)}D')

                if len(self._y[i].shape) != len(shape) + 1:
                    raise ValueError(f'Target image must be {len(shape)}D')
                if self._x[i].shape[:-1] < shape:
                    raise ValueError(
                        'Source image must be larger than the patch size')

                expected_y_image_size = self._scale(self._x[i].shape[:-1])
                if self._y[i].shape[:-1] != expected_y_image_size:
                    raise ValueError('Invalid target image size: '
                                     f'expected {expected_y_image_size}, '
                                     f'but received {self._y[i].shape[:-1]}')

            if len({m.shape[-1] for m in self._x}) != 1:
                raise ValueError(
                    'All source images must have the same number of channels')
            if len({m.shape[-1] for m in self._y}) != 1:
                raise ValueError(
                    'All target images must have the same number of channels')
            print()
            self._batch_x = np.zeros(
                (batch_size, *shape, self._x[0].shape[-1]),
                dtype=self._x[0].dtype)
            self._batch_y = np.zeros(
                (batch_size, *self._scale(shape), self._y[0].shape[-1]),
                dtype=self._y[0].dtype)

        def __len__(self):
            return len(self._x) // self._batch_size  # return a dummy value

        def __next__(self):
            return self.__getitem__(0)

        def __getitem__(self, _):
            for i in range(self._batch_size):
                for _ in range(139):
                    j = np.random.randint(0, len(self._x))

                    tl = [np.random.randint(0, a - b + 1)
                          for a, b in zip(
                              self._x[j].shape, self._batch_x.shape[1:])]

                    x = np.copy(self._x[j][tuple(
                        [slice(a, a + b) for a, b in zip(
                            tl, self._batch_x.shape[1:])])])
                    y = np.copy(self._y[j][tuple(
                        [slice(a, a + b) for a, b in zip(
                            self._scale(tl), self._batch_y.shape[1:])])])

                    if (self._intensity_threshold <= 0.0 or
                            np.count_nonzero(y > self._intensity_threshold)
                            >= self._area_threshold):
                        break
                else:
                    warnings.warn(
                        'Failed to sample a valid patch',
                        RuntimeWarning,
                        stacklevel=3)

                self._batch_x[i], self._batch_y[i] = \
                    self._transform_function(x, y)
            return self._batch_x, self._batch_y

    def flow(self, x, y):
        '''
        Returns a `keras.utils.Sequence` object which generates batches
        infinitely. It can be used as an input generator for
        `keras.models.Model.fit_generator()`.
        Parameters
        ----------
        x: array_like or list of array_like
            Source image(s).
        y: array_like or list of array_like
            Target image(s).
        Returns
        -------
        keras.utils.Sequence
            `keras.utils.Sequence` object which generates tuples of source and
            target image patches.
        '''
        return self._Sequence(x,
                              y,
                              self._batch_size,
                              self._shape,
                              self._transform_function,
                              self._intensity_threshold,
                              self._area_threshold,
                              self._scale_factor)


# MARK: Load training data helpers.
#
# TODO: Look whether they're provided in CSBDeep.
def backend_channels_last():
    assert kb.image_data_format() in ('channels_first', 'channels_last')
    return kb.image_data_format() == 'channels_last'


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)


def consume(iterator):
    collections.deque(iterator, maxlen=0)


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """

    def _raise(e):
        if isinstance(e, BaseException):
            raise e
        else:
            raise ValueError(e)

    allowed = 'STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None.'))
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError(
        "invalid axis '%s', must be one of %s." % (a, list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(
        ValueError("disallowed axis '%s'." % a)) for a in axes)
    consume(axes.count(a) == 1 or _raise(ValueError(
        "axis '%s' occurs more than once." % a)) for a in axes)
    length is None or len(axes) == length or _raise(
        ValueError('axes (%s) must be of length %d.' % (axes, length)))
    return (axes, allowed) if return_allowed else axes


def get_wavelet_config(function_name, is_discrete=None):
    if function_name == '':
        return None

    is_discrete = function_name in pywt.wavelist(kind='discrete')
    if (not is_discrete) and (function_name not in pywt.wavelist(kind='continuous')):
        raise Exception(
            f'Wavelet "{function_name}" is neither in the discrete not continuous list of pywt.wavelist().'
        )
    print(
        f'Wavelet using {"discrete" if is_discrete else "continuous"} function "{function_name}".')

    return [function_name, is_discrete]


def wavelet_transform(mat, wavelet_config, verbose=False):
    '''Applies a wavelet transform on a matrix of shape nx256x256 or nx256x256x1.'''

    [function_name, is_discrete] = wavelet_config

    if verbose:
        print(
            f'Wavelet transforming matrix of shape {mat.shape}; length: {len(mat)}')

    assert np.shape(mat)[1:] == (256, 256) or np.shape(mat)[1:] == (256, 256, 1),\
        f'Expected matrix of shape nx256x256 or nx256x256x1 but got: {np.shape(mat)}'
    requires_extra_dim = np.shape(mat)[-1] == 1

    for i in range(len(mat)):
        if is_discrete:
            C = pywt.dwt2(
                mat[i, :, :] if not requires_extra_dim else np.squeeze(
                    mat[i, :, :, :]),
                wavelet=function_name,
                mode='periodization')
        else:
            C = pywt.cwt2(
                mat[i, :, :] if not requires_extra_dim else np.squeeze(
                    mat[i, :, :, :]),
                wavelet=function_name,
                mode='periodization')

        cA, (cH, cV, cD) = C
        if verbose:
            print(
                f'Got cA shaped {cA.shape}, cH shaped {cH.shape}, cV shaped {cV.shape}, cD shaped {cD.shape}')

        row = np.append(cA, cH, axis=1)
        row2 = np.append(cV, cD, axis=1)
        stack = np.vstack((row, row2))

        if verbose:
            print(f'Got stack shaped {np.shape(stack)}')

        if not requires_extra_dim:
            mat[i, :, :] = stack
        else:
            mat[i, :, :, :] = np.expand_dims(stack, -1)

    return mat


def wavelet_inverse_transform(mat, wavelet_config, verbose=False):
    '''Reverses the wavelet transform on a matrix of shape nx256x256 or nx256x256x1.'''

    [function_name, is_discrete] = wavelet_config

    if verbose:
        print(
            f'Wavelet inverse transforming matrix of shape {mat.shape}; length: {len(mat)}')

    assert np.shape(mat)[1:] == (256, 256) or np.shape(mat)[1:] == (256, 256, 1),\
        f'Expected matrix of shape nx256x256 or nx256x256x1 but got: {np.shape(mat)}'
    requires_extra_dim = np.shape(mat)[-1] == 1

    for i in range(len(mat)):
        if not requires_extra_dim:
            cA, cH, cV, cD = mat[i, :128, :128], mat[i, :128, 128:],\
                mat[i, 128:, :128], mat[i, 128:, 128:]
        else:
            cA, cH, cV, cD = np.squeeze(mat[i, :128, :128, :]),\
                np.squeeze(mat[i, :128, 128:, :]),\
                np.squeeze(mat[i, 128:, :128, :]),\
                np.squeeze(mat[i, 128:, 128:, :])

        if verbose:
            print(
                f'Got cA shaped {cA.shape}, cH shaped {cH.shape}, cV shaped {cV.shape}, cD shaped {cD.shape}')
        C = cA, (cH, cV, cD)

        if is_discrete:
            restored = pywt.idwt2(
                C, wavelet=function_name, mode='periodization')
        else:
            restored = pywt.icwt2(
                C, wavelet=function_name, mode='periodization')
        if verbose:
            print(f'Got restored shape: {restored.shape}')

        if not requires_extra_dim:
            mat[i, :, :] = restored
        else:
            mat[i, :, :, :] = np.expand_dims(restored, -1)

    return mat


def load_training_data(file, validation_split=0, axes=None, n_images=None,
                       verbose=False):
    """Load training data from file in ``.npz`` format.
    The data file is expected to have the keys:
    - ``X``    : Array of training input images.
    - ``Y``    : Array of corresponding target images.
    - ``axes`` : Axes of the training images.
    Parameters
    ----------
    file : str
        File name
    validation_split : int
        Seed used for splitting of images to use as validation set during training.
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

    if verbose:
        print('Loading data...')

    f = np.load(file)
    X, Y = f['X'], f['Y']
    SB, SE = f['SB'], f['SE']

    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)

    if verbose:
        print(f'Found axes: {axes}')
        print(f'Raw X shape is {tf.shape(X)}; Raw Y shape is {tf.shape(Y)}')

    # The inputted data has 3 channels; add one dimension if a channel
    # dimension is requested.
    if len(axes) == 4:
        X = tf.expand_dims(X, axis=-1)
        Y = tf.expand_dims(Y, axis=-1)

        print(f'New X shape is {tf.shape(X)}; New Y shape is {tf.shape(Y)}')

    assert X.shape == Y.shape
    assert X.ndim == Y.ndim
    assert len(axes) == X.ndim
    assert len(axes) == 3 or 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    # assert 0 <= validation_split < 1

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes).get('C', None)

    # assert type(validation_split) is int,"validation_split must be an integer"
    # # TODO (nvora01): See if there is a more efficient way of doing this... 
    # ROIs = len(SB)
    # random.seed(validation_split)
    # print('Splitting Data using seed ' + str(validation_split))

    # train_idx = sorted(random.sample(range(0,ROIs), ROIs-8))
    # validation_idx = sorted([x for x in list(range(0,ROIs)) if x not in train_idx])

    # # Generation of Validation Set
    # X_t,Y_t = np.empty((1,256,256)), np.empty((1,256,256))
    # for i in range(len(validation_idx)):
    #     X_t, Y_t = np.concatenate((X_t,X[int(SB[validation_idx[i]]):int(SE[validation_idx[i]])+1]),axis=0),
    #     np.concatenate((Y_t,Y[int(SB[validation_idx[i]]):int(SE[validation_idx[i]])+1]),axis=0)

    # # Remove the empty initialization image from stack
    # X_t, Y_t = np.delete(X_t,0,axis=0), np.delete(Y_t,0,axis=0)

    # # Generating stack ranges for validation set
    # num_stacks = [int(SE[x]-SB[x]+1) for x in validation_idx]
    # prev = 0
    # stack_ranges=np.empty((len(num_stacks),2), dtype=int)
    # for i in range(len(num_stacks)):
    #     stack_ranges[i] = [prev, prev + num_stacks[i]-1]
    #     prev += num_stacks[i]

    # # Generation of Training Set
    # validation_idx = np.flip(validation_idx)
    # for i in range(len(validation_idx)):
    #     X, Y = np.delete(X,list(range(int(SB[validation_idx[i]]),int(SE[validation_idx[i]])+1)),axis=0), 
    #     np.delete(Y,list(range(int(SB[validation_idx[i]]),int(SE[validation_idx[i]])+1)),axis=0)
    
    # # TODO (nvora01): Remove extra code
    
    # # n_val = 620
    # # n_train = n_images - n_val
    # # assert 0 < n_val and 0 < n_train
    # # X_t, Y_t = X[-n_val:],  Y[-n_val:]
    # # X,   Y = X[:n_train], Y[:n_train]

    # # assert X.shape[0] == n_train and X_t.shape[0] == n_val

    # if channel != None:
    #     X_t = move_channel_for_backend(X_t, channel=channel)
    #     Y_t = move_channel_for_backend(Y_t, channel=channel)

    # TODO (nvora01): Remove this section
    ################################################################################
    ######################Testing on older models!##################################
    new_test = range(0,int(SE[6])+1)
    ranges = np.hstack((new_test,range(int(SE[-5]),len(X))))
    X_t, Y_t = X[ranges],  Y[ranges]

    validation_idx = np.hstack((range(0,7),range(-5,0)))
    num_stacks = [int(SE[x]-SB[x]+1) for x in validation_idx]
    prev = 0
    stack_ranges=np.empty((len(num_stacks),2),dtype=int)
    for i in range(len(num_stacks)):
        stack_ranges[i] = [prev, prev + num_stacks[i]-1]
        prev += num_stacks[i]

    X, Y = X[range(int(SB[7]), int(SB[-5]))], Y[range(int(SB[7]), int(SB[-5]))]
    if channel != None:
        X_t = move_channel_for_backend(X_t, channel=channel)
        Y_t = move_channel_for_backend(Y_t, channel=channel)
    ################################################################################
    if channel != None:
        X = move_channel_for_backend(X, channel=channel)
        Y = move_channel_for_backend(Y, channel=channel)

    if channel != None:
        axes = axes.replace('C', '')  # remove channel
        if backend_channels_last():
            axes = axes+'C'
        else:
            axes = axes[:1]+'C'+axes[1:]

    data_val = (X_t, Y_t)

    if verbose:
        ax = axes_dict(axes)
        n_train, n_val = len(X), len(X_t) if validation_split > 0 else 0
        image_size = tuple(X.shape[ax[a]] for a in axes if a in 'TZYX')
        n_dim = len(image_size)
        if channel != None:
            n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]

        print('number of training images:\t', n_train)
        print('number of validation images:\t', n_val)
        print('image size (%dD):\t\t' % n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        if channel != None:
            print('channels in / out:\t\t', n_channel_in, '/', n_channel_out)

    return (X, Y), data_val, axes, stack_ranges


def patch_slice(slice):
    '''Splits up the 512x512 slice into 4 256x256 patches.'''
    slice = np.squeeze(slice)
    assert np.shape(slice) == (
        512, 512), f'Slice must be 512x512 but instead found shape: {np.shape(slice)}'

    # The axes are swapped to maintain the correct order since our patches are square 256x256
    # and not 512x128 rectangles.
    return np.reshape(slice, [2, 256, 2, 256]).swapaxes(1, 2).reshape(4, 256, 256)


def stitch_patches(patches):
    '''Stitches the 4 256x256 patches back together into a 512x512 slice.'''
    patches = np.squeeze(patches)
    assert np.shape(patches) == (
        4, 256, 256), f'Patches must be 4x256x256 but instead found shape: {np.shape(patches)}'

    # The axes are swapped to maintain the correct order since our patches are square 256x256
    # and not 512x128 rectangles.
    return np.reshape(patches, [2, -1, 256, 256]).swapaxes(1, 2).reshape(512, 512)


def default_load_data(data_path, requires_channel_dim, config):
    (X, Y), (X_val, Y_val), _, val_ranges = load_training_data(
        data_path,
        validation_split=config['val_seed'],
        axes='SXY' if not requires_channel_dim else 'SXYC',
        verbose=True)

    return (X, Y), (X_val, Y_val), val_ranges


def gather_data(config, data_path, requires_channel_dim):
    '''Gathers the data that is already normalized in local prep.'''
    print('=== Gathering data ---------------------------------------------------')

    # Similar to 'data_generator.py'
    (X, Y), (X_val, Y_val), _ = default_load_data(data_path, requires_channel_dim, config)

    wavelet_config = get_wavelet_config(
        function_name=config['wavelet_function'])

    if wavelet_config != None:
        X = wavelet_transform(np.array(X), wavelet_config=wavelet_config)
        Y = wavelet_transform(np.array(Y), wavelet_config=wavelet_config)
        X_val = wavelet_transform(
            np.array(X_val), wavelet_config=wavelet_config)
        Y_val = wavelet_transform(
            np.array(Y_val), wavelet_config=wavelet_config)

    data_gen = DataGenerator(
        config['input_shape'],
        50,
        transform_function=None)

    if not requires_channel_dim:
        # The data generator only accepts 2D data.
        training_data = data_gen.flow(*list(zip([X, Y])))
        validation_data = data_gen.flow(*list(zip([X_val, Y_val])))
    else:
        # TODO: Streamline RCAN and CARE data generation.
        # Shape: (2, n_train_patches, 256, 256, 1)
        training_data = (X, Y)
        # Shape: (2, n_valid_patches, 256, 256, 1)
        validation_data = (X_val, Y_val)

    print(f'Got training data with shape {np.shape(training_data)}.')
    print(f'Got validation data with shape {np.shape(validation_data)}.')

    print('----------------------------------------------------------------------')

    return (training_data, validation_data)

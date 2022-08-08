import pathlib
import keras as keras
from tqdm.keras import TqdmCallback as _TqdmCallback
import warnings
import functools
from tqdm.utils import IS_WIN
import tqdm

# Local dependencies
from basics import is_multi_gpu_model


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s '
                                  'available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to '
                                  '%0.5f, saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        save_model(
                            filepath, self.model, self.save_weights_only)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f'
                                  % (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s'
                          % (epoch + 1, filepath))
                save_model(filepath, self.model, self.save_weights_only)


class TqdmCallback(_TqdmCallback):
    def __init__(self):
        super().__init__(
            tqdm_class=functools.partial(
                tqdm.tqdm, dynamic_ncols=True, ascii=IS_WIN))
        self.on_batch_end = self.bar2callback(
            self.batch_bar, pop=['batch', 'size'])


def save_model(filename, model, weights_only=False):
    if is_multi_gpu_model(model):
        m = model.layers[-(len(model.outputs) + 1)]
    else:
        m = model

    if weights_only:
        m.save_weights(filename, overwrite=True)
    else:
        m.save(filename, overwrite=True)


def staircase_exponential_decay(n):
    '''
    Returns a scheduler function to drop the learning rate by half
    every `n` epochs.
    '''
    return lambda epoch, lr: lr / 2 if epoch != 0 and epoch % n == 0 else lr


def get_callbacks(epochs, output_dir, checkpoint_filepath, validation_data):
    return [
        keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(epochs // 4)),
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=True),
        ModelCheckpoint(
            str(pathlib.Path(output_dir) / checkpoint_filepath),
            monitor='loss' if validation_data is None else 'val_loss',
            save_best_only=True, verbose=1, mode='min'),
        TqdmCallback()
    ]

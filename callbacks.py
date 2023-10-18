import tensorflow as tf
import pathlib
import keras as keras
from tqdm.keras import TqdmCallback as _TqdmCallback
import warnings
import functools
from tqdm.utils import IS_WIN
import tqdm
import os
from matplotlib import pyplot as plt
from IPython.display import clear_output

# Local dependencies
from basics import is_multi_gpu_model

# str(pathlib.Path(output_dir) / checkpoint_filepath),
#             monitor='val_loss' if validation_data is not None else 'loss',
#             save_best_only=True, verbose=1, mode='min')

class PlotLearning(keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """
    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []
            

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]
        
        # Plotting
        metrics = [x for x in logs if 'val' not in x]
        
        f, axs = plt.subplots(1, len(metrics), figsize=(15,5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2), 
                        self.metrics[metric], 
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2), 
                            self.metrics['val_' + metric], 
                            label='val_' + metric)
                
            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor, verbose=1, save_best_only=True, 
        save_weights_only=True, mode='min') -> None:
    
        super(ModelCheckpoint,self).__init__(filepath=filepath,monitor=monitor,
                verbose=verbose,save_best_only=save_best_only,
                save_weights_only=save_weights_only, mode=mode)
        
        self.ckpt = tf.train.Checkpoint(completed_epochs=tf.Variable(0,trainable=False,dtype='int32'))
        ckpt_dir = f'{os.path.dirname(filepath)}/tf_ckpts'
        self.manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=3)

    def on_epoch_begin(self,epoch,logs=None):        
        self.ckpt.completed_epochs.assign(epoch)
        self.manager.save()
        print( f"Epoch checkpoint {self.ckpt.completed_epochs.numpy()}  saved to: {self.manager.latest_checkpoint}" ) 

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

def get_callbacks(model_name, epochs, output_dir, checkpoint_filepath, validation_data):
    # Learning-rate callback.
    if model_name == 'rcan':
        learning_rate_callback = keras.callbacks.LearningRateScheduler(
            staircase_exponential_decay(epochs // 4))
    elif model_name == 'care':
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            verbose=True, factor=0.97, min_delta=0, patience=20)
    elif model_name == 'srgan':
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            verbose=True, factor=0.97, min_delta=0, patience=20)
    elif model_name == 'resnet':
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            verbose=True, factor=0.97, min_delta=0, patience=20)
    elif model_name == 'wunet':
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            verbose=True, factor=0.97, min_delta=0, patience=20)        
    elif model_name == 'UnetRCAN':
        learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            verbose=True, factor=0.97, min_delta=0, patience=20)        
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    csv_logger = tf.keras.callbacks.CSVLogger('training.log')
    return [
        learning_rate_callback,
        keras.callbacks.TensorBoard(
            log_dir=str(output_dir),
            write_graph=True),
        ModelCheckpoint(
            str(pathlib.Path(output_dir) / checkpoint_filepath),
            monitor='val_loss' if validation_data is not None else 'loss',
            save_best_only=True, verbose=1, mode='min'),
        PlotLearning(),
        csv_logger,
        TqdmCallback()
    ]
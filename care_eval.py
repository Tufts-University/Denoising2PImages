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
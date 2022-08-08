from __future__ import print_function, unicode_literals, absolute_import, division
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
import pathlib
from tqdm.utils import IS_WIN
from tqdm.keras import TqdmCallback as _TqdmCallback
import functools
import warnings
import tqdm
import collections
from importlib import import_module
from tensorflow import __version__ as _tf_version
from tensorflow.python.client.device_lib import list_local_devices
from packaging import version
import tensorflow_probability as tfp
import fractions
import numexpr
import tensorflow as tf
from tensorflow.image import ssim as ssim2
from tensorflow.image import ssim_multiscale as msssim
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
import keras
from keras import backend as kb
from keras.utils.conv_utils import normalize_tuple
from csbdeep.utils import axes_dict, plot_some, plot_history
from csbdeep.internals import nets
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.models import Config, CARE
    


#################################################################################
''' Data was Normalized in Local Prep'''

c = axes_dict(axes)['C']
n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

config = Config(axes, n_channel_in, n_channel_out, 
                train_steps_per_epoch=100, train_batch_size=50, 
                train_epochs=300, unet_n_depth=6, 
                unet_n_first=32, unet_kern_size=3, 
                train_learning_rate=1e-05)
print(config)
                
# create a callback that saves the model's weights every said amount of epochs
epoch_freq = 10

# BUILD
model = build_CARes(config)
model = compile_model(model, config.train_learning_rate)
model.summary()
# TRAIN

history = model.fit(X, Y,
                    batch_size=config.train_batch_size,
                    epochs=config.train_epochs,
                    shuffle=True,
                    validation_data=(X_val, Y_val), 
                    callbacks=[reduce_lr, cp_callback])
                    
model.save_weights(model_save_path+'CAREtrained.h5')

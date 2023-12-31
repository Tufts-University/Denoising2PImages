import tensorflow as tf
from tensorflow.python.client.device_lib import list_local_devices
import os

def is_multi_gpu_model(model):
    '''Checks if the model supports multi-GPU data parallelism.'''
    return hasattr(model, 'is_multi_gpu_model') and model.is_multi_gpu_model


def get_gpu_count():
    '''Returns the number of available GPUs.'''
    return len([x for x in list_local_devices() if x.device_type == 'GPU'])


def print_device_info():
    print('=== Device info -----------------------------------------------------')

    print(
        f'The number of available GPUs is: {len(tf.config.list_physical_devices("GPU"))}')
    print(f'Cuda gpu is available: {tf.test.is_gpu_available(cuda_only=True)}')

    print('--------------------------------------------------------------------')


def final_weights_name():
    return 'weights_final.hdf5'

def SRGAN_Weight_search(data_path):
    os.chdir(data_path)
    print(f'Searching for Pretrained Weights in the folder: {data_path}')
    Gen_flag = 0
    CARE_flag = 0
    if os.path.exists('Pretrained.hdf5'):
            print(f'Pretrained RESNET found in {data_path}')
            Gen_flag = 1
    if os.path.exists('CARE_Pretrained.hdf5'):
            print(f'Pretrained CARE found in {data_path}')
            CARE_flag = 1
    return Gen_flag, CARE_flag
            
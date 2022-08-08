from tensorflow.python.client.device_lib import list_local_devices


def is_multi_gpu_model(model):
    '''Checks if the model supports multi-GPU data parallelism.'''
    return hasattr(model, 'is_multi_gpu_model') and model.is_multi_gpu_model

def get_gpu_count():
    '''Returns the number of available GPUs.'''
    return len([x for x in list_local_devices() if x.device_type == 'GPU'])
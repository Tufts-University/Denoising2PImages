import os as os
import tensorflow as tf
import sys

# Local dependencies
import data_generator
import train

#################################################################################

data_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_FAD_healthy.npz'
model_name = 'FAD_model_0713_cervix_SSIML1'
main_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/'
os.chdir(main_path)
if not os.path.exists(os.path.join(main_path, model_name)):
    os.mkdir(os.path.join(main_path, model_name))
model_save_path = os.path.join(main_path, model_name)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
print(is_cuda_gpu_available)

config = {
    "epochs": 300,
    "steps_per_epoch": 4,
    "num_residual_groups": 5,
    "num_residual_blocks": 5,
    "input_shape": [256, 256],
    "LW_weight_seed": 0,
    "LW_weight_num": 4
}

config.setdefault('epochs', 300)
config.setdefault('steps_per_epoch', 256)
config.setdefault('initial_learning_rate', 1e-5)
config.setdefault('loss', 'SSIML1_loss')
config.setdefault('metrics', ['psnr', 'ssim'])
config.setdefault('num_residual_blocks', 3)
config.setdefault('num_residual_groups', 5)
config.setdefault('channel_reduction', 4)
config.setdefault('num_channels', 32)
config.setdefault('LW_weight_seed', 0)
config.setdefault('LW_weight_num', 4)


def gather_data(config, data_path):
    '''Gathers the data that is already normalized in local prep.'''
    print('Gathering data...')

    (X, Y), (X_val, Y_val), axes = data_generator.load_training_data(
        data_path, validation_split=0.15, axes='SXY', verbose=True)

    data_gen = data_generator.DataGenerator(
        config['input_shape'],
        50,
        transform_function=None)

    training_data = data_gen.flow(*list(zip([X, Y])))
    validation_data = data_gen.flow(*list(zip([X_val, Y_val])))

    return (training_data, validation_data)


def main():
    if len(sys.argv) >= 1 and sys.argv[0] == 'train':
        print('Running in "train" mode.')

        (training_data, validation_data) = gather_data(config, data_path)

        train.train(config,
                    output_dir=model_save_path,
                    training_data=training_data,
                    validation_data=validation_data)

        print('Successfully completed training.')


try:
    main()
except Exception as e:
    sys.stderr.write(f'Failed with error: {e}')
    raise e

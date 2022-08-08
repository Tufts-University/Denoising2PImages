from importlib.metadata import requires
import os as os
import tensorflow as tf
import sys

# Local dependencies
import data_generator
import train
import basics

#################################################################################

data_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/NV_713_FAD_healthy.npz'
model_name = 'monorepo test'
main_path = '/cluster/tufts/georgakoudi_lab01/nvora01/NV_052622_Denoising/'
os.chdir(main_path)

if not os.path.exists(os.path.join(main_path, model_name)):
    os.mkdir(os.path.join(main_path, model_name))
model_save_path = os.path.join(main_path, model_name)


def make_config(model_name):
    return {
        'epochs': 300,
        'steps_per_epoch': {'rcan': None, 'care': 100}[model_name],
        'num_residual_groups': 5,
        'num_residual_blocks': 5,
        'input_shape': [256, 256],
        'initial_learning_rate': 1e-5,
        'loss': 'ssiml1_loss',
        'metrics': ['psnr', 'ssim'],
        'num_residual_blocks': 3,
        'num_residual_groups': 5,
        'channel_reduction': 4,
        'num_channels': 32,
    }


def gather_data(config, data_path, requires_channel_dim):
    '''Gathers the data that is already normalized in local prep.'''
    print('=== Gathering data ---------------------------------------------------')

    (X, Y), (X_val, Y_val), axes = data_generator.load_training_data(
        data_path,
        validation_split=0.15,
        axes='SXY' if not requires_channel_dim else 'SXYC',
        verbose=True)

    data_gen = data_generator.DataGenerator(
        config['input_shape'],
        50,
        transform_function=None)

    if not requires_channel_dim:
        # The data generator only accepts 2D data.
        training_data = data_gen.flow(*list(zip([X, Y])))
        validation_data = data_gen.flow(*list(zip([X_val, Y_val])))
    else:
        # TODO: Streamline RCAN and CARE data generation.
        training_data = (X, Y)
        validation_data = (X_val, Y_val)

    print('----------------------------------------------------------------------')

    return (training_data, validation_data)


def apply_config_flags(config_flags, config):
    for config_flag in config_flags:
        components = config_flag.split('=')
        if len(components) != 2:
            raise ValueError(
                f'Invalid config flag: "{config_flag}"; expected key="string" or key=number')

        key, raw_value = components
        if key not in config:
            raise ValueError(
                f'Invalid config flag: "{config_flag}"; key "{key}" not found in config.')

        try:
            value = int(raw_value)
        except:
            try:
                value = float(raw_value)
            except:
                value = raw_value

        config[key] = value

    return config


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py <mode: train | eval> <name: rcan | care> <config options...>')
        raise ValueError('Invalid arguments.')

    # === Get arguments ===

    # We get the arguments in the form:
    # ['main.py', mode, model_name, config_options...]

    mode = sys.argv[1]
    if mode not in ['train', 'eval']:
        raise ValueError(f'Invalid mode: "{mode}"')

    model_name = sys.argv[2]
    if model_name not in ['rcan', 'care']:
        raise ValueError(f'Invalid model name: "{model_name}"')

    config_flags = sys.argv[3:] if len(sys.argv) > 3 else []
    config = make_config(model_name)
    config = apply_config_flags(config_flags, config)

    print(f'Using config: {config}\n')

    # === Send out jobs ===

    basics.print_device_info()

    if mode == 'train':
        print('Running in "train" mode.\n')

        (training_data, validation_data) = gather_data(
            config, data_path, requires_channel_dim=model_name == 'care')

        train.train(model_name,
                    config,
                    output_dir=model_save_path,
                    training_data=training_data,
                    validation_data=validation_data)

        print('Successfully completed training.')
    elif mode == 'eval':
        print('Not implemented.')


try:
    main()
except Exception as e:
    sys.stderr.write(f'Failed with error: {e}')
    raise e

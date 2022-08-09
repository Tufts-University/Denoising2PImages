from importlib.metadata import requires
import os as os
import tensorflow as tf
import sys

# Local dependencies
import train
import basics
import eval

#################################################################################


def make_config(model_name):
    return {
        'cwd': '',
        'data': '',
        'epochs': 300,
        'wavelet': False,
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
                if raw_value == 'True':
                    value = True
                elif raw_value == 'False':
                    value = False
                else:
                    value = raw_value

        config[key] = value

    return config


def main():
    if len(sys.argv) < 4:
        print('Usage: python main.py <mode: train | eval> <name: rcan | care> <trial_name> <config options...>')
        raise Exception('Invalid arguments.')

    # === Get arguments ===

    # We get the arguments in the form:
    # ['main.py', mode, model_name, config_options...]

    mode = sys.argv[1]
    if mode not in ['train', 'eval']:
        raise Exception(f'Invalid mode: "{mode}"')

    model_name = sys.argv[2]
    if model_name not in ['rcan', 'care']:
        raise Exception(f'Invalid model name: "{model_name}"')

    trial_name = sys.argv[3]
    print(f'Using trial name: "{trial_name}"')

    config_flags = sys.argv[4:] if len(sys.argv) > 4 else []
    config = make_config(model_name)
    config = apply_config_flags(config_flags, config)

    print(f'Using config: {config}\n')

    main_path = config['cwd']
    if main_path == '':
        raise Exception('Please set the "cwd" config flag. To use the current directory use: cwd=.')
    elif not os.path.isdir(main_path):
        raise Exception(f'Could not find current working directory (cwd): "{main_path}"')

    data_path = config['data']
    if data_path == '':
        raise Exception('Please set the "data" config flag to specify where the data is in relation to the current directory.')

    # === Get right paths ===

    os.chdir(main_path)
    print(f'Changed directory: {os.getcwd()}')
    
    if not os.path.isfile(data_path):
        raise Exception(f'Could not find file at data path: "{data_path}"')

    if not os.path.exists(os.path.join(main_path, trial_name)):
        os.mkdir(os.path.join(main_path, trial_name))
    model_save_path = os.path.join(main_path, trial_name)

    # === Send out jobs ===

    basics.print_device_info()

    if mode == 'train':
        print('Running in "train" mode.\n')

        train.train(model_name,
                    config=config,
                    output_dir=model_save_path,
                    data_path=data_path)

        print('Successfully completed training.')
    elif mode == 'eval':
        print('Running in "eval" mode.\n')

        eval.eval(model_name,
                  trial_name=trial_name,
                  config=config,
                  output_dir=model_save_path,
                  data_path=data_path)

        print('Successfully completed evaluation.')


try:
    main()
except Exception as e:
    sys.stderr.write(f'Failed with error: {e}')
    raise e

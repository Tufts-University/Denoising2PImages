from importlib.metadata import requires
import os as os
import tensorflow as tf
import sys

# Local dependencies
import train
import basics
import eval

#################################################################################

# TODO (Nilay): Create a config run option to simplify options and list defaults
def make_config(model_name):
    return {
        'cwd': '',
        'nadh_data': '',
        'fad_data': '',
        'epochs': 300,
        'steps_per_epoch': {'srgan': None,'rcan': None, 'care': 100, 'resnet':None}[model_name],
        'input_shape': [256, 256], # TODO: Remove since this is almost fixed. (Nilay) Need to check what Filip did here since we may want variable size patches
        'initial_learning_rate': 1e-5,
        'val_seed': 0, # Controls the validation split
        'val_split': 4, # Controls how many stacks to include in the validation set
        'test_split': 8, # Controls how many stack to include in the test set
        'test_flag' : True, # Controls if a test set is generated

        # Metrics
        'loss': {'srgan': 'mse', 'care': 'ssiml1_loss', 'rcan': 'ssiml1_loss', 'resnet':'mse'}[model_name],
        'metrics': ['psnr', 'ssim'],

        # Metric hyperparameters 
        'loss_alpha': 0.5, # How much different loss functions are weighted in the compound loss.

        # RCAN and Resnet config
        'num_channels': 32,
        'num_residual_blocks': 5,
        'num_residual_groups': 5,
        'channel_reduction': 4,

        # Unet config
        'unet_n_depth': 6,
        'unet_n_first': 32, 
        'unet_kern_size': 3,

        # Wavelet config
        'wavelet_function': '', # One of pywt.wavelist() or empty for non-wavelet.
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
    if model_name not in ['rcan', 'care', 'srgan', 'resnet']:
        raise Exception(f'Invalid model name: "{model_name}"')

    trial_name = sys.argv[3]
    print(f'Using trial name: "{trial_name}"')

    config_flags = sys.argv[4:] if len(sys.argv) > 4 else []
    config = make_config(model_name)
    config = apply_config_flags(config_flags, config)

    print(f'Using config: {config}\n')

    main_path = config['cwd']
    if main_path == '':
        raise Exception(
            'Please set the "cwd" config flag. To use the current directory use: cwd=.')
    elif not os.path.isdir(main_path):
        raise Exception(
            f'Could not find current working directory (cwd): "{main_path}"')

    nadh_data_path = config['nadh_data']
    fad_data_path = config['fad_data']
    if nadh_data_path == '' and fad_data_path == '':
        raise Exception(
            'Please at least one of the two data-path flags "nadh_data" or "fad_data" config flag to specify where the data is in relation to the current directory.')

    # === Get right paths ===

    print(f'Changing to directory: {main_path}')
    os.chdir(main_path)

    # Check data paths exist.
    if nadh_data_path != "" and not os.path.isfile(nadh_data_path):
        raise Exception(
            f'Could not find file at NADH data path: "{nadh_data_path}"')
    if fad_data_path != "" and not os.path.isfile(fad_data_path):
        raise Exception(
            f'Could not find file at FAD data path: "{fad_data_path}"')

    if not os.path.exists(os.path.join(main_path, trial_name)):
        os.mkdir(os.path.join(main_path, trial_name))
    model_save_path = trial_name

    # === Send out jobs ===

    basics.print_device_info()

    if mode == 'train':
        data_path = None
        if nadh_data_path != '' and fad_data_path == '':
            print(f'Using NADH data at: {nadh_data_path}')
            data_path = nadh_data_path
        elif fad_data_path != '' and nadh_data_path == '':
            print(f'Using FAD data at: {fad_data_path}')
            data_path = fad_data_path
        else:
            raise Exception(
                'Train expects just one data set; either "nadh_data" or "fad_data".')

        print('Running in "train" mode.\n')

        train.train(model_name,
                    config=config,
                    output_dir=model_save_path,
                    data_path=data_path)

        print('Successfully completed training.')
    elif mode == 'eval':
        if nadh_data_path != '':
            print(f'Using NADH data at: {nadh_data_path}')
        if fad_data_path != '':
            print(f'Using FAD data at: {fad_data_path}')

        print('Running in "eval" mode.\n')

        eval.eval(model_name,
                  trial_name=trial_name,
                  config=config,
                  output_dir=model_save_path,
                  # The above code checks that at least one is not empty.
                  nadh_path=nadh_data_path if nadh_data_path != '' else None,
                  fad_path=fad_data_path if fad_data_path != '' else None)

        print('Successfully completed evaluation.')


try:
    main()
except Exception as e:
    sys.stderr.write(f'Failed with error: {e}')
    raise e

import pathlib
import os
import shutil

# Local dependencies
import callbacks
import model_builder
import data_generator
import basics
import srgan

def determine_training_strategy(model, output_dir):
    print('=== Determining Training Strategy -----------------------------------')

    if not os.path.exists(output_dir):
        print(f'Creating output directory: "{output_dir}"')
        os.makedirs(output_dir)

    dir_contents = os.listdir(output_dir)

    checkpoint_files = [filename for filename in dir_contents if 'weights' in filename.lower()]
    finished_training = basics.final_weights_name() in dir_contents

    if finished_training:
        raise Exception(f'Model has already trained and produced final weights: "{basics.final_weights_name()}"')
    elif len(checkpoint_files) > 0:
        print(f'Found {len(checkpoint_files)} checkpoint weight files: {checkpoint_files}.')

        last_modified_file = max(checkpoint_files, key=lambda file: os.path.getmtime(os.path.join(output_dir, file)))
        print(f'Found last modified checkpoint file: "{last_modified_file}"')

        raise Exception(f'Cannot continue training from checkpoints. Terminating...')
        # TODO: Implement continued training from checkpoints. (Load correct lr, epochs, and anything else that changes.)
        # model.load_weights(os.path.join(output_dir, last_modified_file))
        # print("Successfully loaded weights from last checkpoint.")
    else: 
        print('Starting training without any checkpoint weights.')

    print('--------------------------------------------------------------------')

    return model

def fit_model(model, model_name, config, output_dir, training_data, validation_data):
    print('=== Fitting model --------------------------------------------------')
    final_dir = pathlib.Path(output_dir)
    os.chdir(final_dir)
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'
    model.fit(
        x=training_data if model_name != 'care' else training_data[0],
        y=None if model_name != 'care' else training_data[1],
        epochs=config['epochs'],
        shuffle=True,
        validation_data=validation_data,
        verbose=0,
        callbacks=callbacks.get_callbacks(
            model_name,
            config['epochs'],
            final_dir,
            checkpoint_filepath,
            validation_data))

    print('--------------------------------------------------------------------')

    return model


def train(model_name, config, output_dir, data_path):
    print('Training...')

    (training_data, validation_data) = data_generator.gather_data(
        config, 
        data_path, 
        requires_channel_dim=model_name == 'care' or model_name == 'wunet')

    strategy = model_builder.create_strategy()
    if model_name == 'srgan':
        initial_path = os.getcwd()
        if not os.path.exists(pathlib.Path(output_dir)):
            print(f'Creating output directory: "{output_dir}"')
            os.makedirs(output_dir)
        srgan_checkpoint, srgan_checkpoint_manager = srgan.SRGAN_fit_model(model_name, strategy, config, initial_path,output_dir,training_data, validation_data)
        os.chdir(pathlib.Path(output_dir)/ 'ckpt' / 'srgan')
        srgan_checkpoint.restore(srgan_checkpoint_manager.latest_checkpoint)
        final_model = srgan_checkpoint.generator
        final_weights_path = str(pathlib.Path(output_dir) / basics.final_weights_name())
        final_model.save_weights(final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')
    else:
        initial_path = os.getcwd()
        model = model_builder.build_and_compile_model(model_name, strategy, config)
        model = determine_training_strategy(model, output_dir)
        model = fit_model(model, model_name, config, output_dir,
                        training_data, validation_data)

        os.chdir(output_dir)
        model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
        assert len(model_paths) != 0, f'No models found under {output_dir}'
        latest = max(model_paths, key=os.path.getmtime)
        final_weights_path = str(pathlib.Path(output_dir) / basics.final_weights_name())
        source = output_dir + '/' + latest
        print(f'Location of source file: "{source}"')
        print(f'Location of Final Weights file: "{final_weights_path}"')
        shutil.copy(source, final_weights_path)
        print(f'Weights are saved to: "{final_weights_path}"')
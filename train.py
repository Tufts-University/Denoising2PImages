from tabnanny import check
import tensorflow as tf
import pathlib
import os
import shutil

# Local dependencies
import callbacks
import model_builder
import data_generator
import basics


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

    steps_per_epoch = config['steps_per_epoch'] if config['steps_per_epoch'] != None else None
    validation_steps = None if validation_data is None else steps_per_epoch
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

    model.fit(
        x=training_data if model_name == 'rcan' else training_data[0],
        y=None if model_name == 'rcan' else training_data[1],
        epochs=config['epochs'],
        # steps_per_epoch=steps_per_epoch,
        shuffle=True,
        validation_data=validation_data,
        # validation_steps=validation_steps,
        verbose=0,
        callbacks=callbacks.get_callbacks(
            model_name,
            config['epochs'],
            output_dir,
            checkpoint_filepath,
            validation_data))

    print('--------------------------------------------------------------------')

    return model


def train(model_name, config, output_dir, data_path):
    print('Training...')

    (training_data, validation_data) = data_generator.gather_data(
        config, 
        data_path, 
        requires_channel_dim=model_name == 'care')

    strategy = model_builder.create_strategy()
    model = model_builder.build_and_compile_model(model_name, strategy, config)
    model = determine_training_strategy(model, output_dir)
    model = fit_model(model, model_name, config, output_dir,
                      training_data, validation_data)

    # TODO: Confirm that this actually works
    os.chdir(output_dir)
    model_paths = [model_path for model_path in os.listdir() if model_path.endswith(".hdf5") ]
    assert len(model_paths) != 0, f'No models found under {output_dir}'
    latest = max(model_paths, key=os.path.getmtime)
    final_weights_path = str(pathlib.Path(output_dir) / basics.final_weights_name())
    shutil.copy(latest, final_weights_path)

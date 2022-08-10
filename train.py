import tensorflow as tf
import pathlib

# Local dependencies
import callbacks
import model_builder
import data_generator


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
        requires_channel_dim=model_name == 'care', 
        wavelet_model=config['wavelet'])

    strategy = model_builder.create_strategy()
    model = model_builder.build_and_compile_model(model_name, strategy, config)
    model = fit_model(model, model_name, config, output_dir,
                      training_data, validation_data)

    # FIXME: Look if early stopping is in effect (so we don't get overfitted weights).
    final_weights_path = str(pathlib.Path(output_dir) / 'weights_final.hdf5')
    model.save_weights(final_weights_path)

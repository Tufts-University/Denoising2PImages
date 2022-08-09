import pathlib


# Local dependencies

import model_builder
import data_generator


def load_weights(model, output_dir):
    print('=== Loading model weights ------------------------------------------')

    model.load_weights(str(pathlib.Path(output_dir) / 'weights_final.hdf5'))

    print('--------------------------------------------------------------------')


def apply_model(model, model_name, validation_data):
    print('=== Applying model ------------------------------------------------')

    model.evaluate(
        x=validation_data[0],
        y=validation_data[1],
        verbose=1)

    print('--------------------------------------------------------------------')


def eval(model_name, config, output_dir, data_path):
    print('Evaluating...')

    # TODO: Load training data and add generate_data func.
    (_, validation_data) = data_generator.gather_data(
        config, 
        data_path, 
        requires_channel_dim=model_name == 'care', 
        wavelet_transform=config['wavelet'])

    if config['wavelet'] == True:
        data_generator.wavelet_transform(validation_data)

    strategy = model_builder.create_strategy()
    model = model_builder.build_and_compile_model(model_name, strategy, config)

    load_weights(model, output_dir=output_dir)

    apply_model(
        model,
        model_name=model_name,
        validation_data=validation_data)

    return model

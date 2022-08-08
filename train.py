import tensorflow as tf
import pathlib

# Local dependencies
import basics
import metrics
import rcan
import care
import callbacks


def print_config_summary(config):
    print('Building RCAN model')
    print('  - input_shape =', config['input_shape'])
    for s in ['num_channels',
              'num_residual_blocks',
              'num_residual_groups',
              'channel_reduction']:
        print(f'  - {s} =', config[s])


def create_strategy():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {strategy.num_replicas_in_sync}')

    return strategy


def compile_model(model, initial_learning_rate, loss_name, metric_names):
    print('=== Compiling model ------------------------------------------------')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate),
        loss=metrics.lookup_loss(loss_name),
        metrics=metrics.lookup_metrics(metric_names))

    print('Model summary:\n')
    model.summary()

    print('--------------------------------------------------------------------')

    return model


def build_and_compile_model(model_name, strategy, config):
    with strategy.scope():
        if model_name == 'rcan':
            model = rcan.build_rcan(
                (*config['input_shape'], 1),
                num_channels=config['num_channels'],
                num_residual_blocks=config['num_residual_blocks'],
                num_residual_groups=config['num_residual_groups'],
                channel_reduction=config['channel_reduction'])
        elif model_name == 'care':
            model = care.build_care(config, 'SXYC')
        else:
            raise ValueError(f'Non-implemented model: {model_name}')

        #model = convert_to_multi_gpu_model(model, gpus)
        model = compile_model(
            model,
            config['initial_learning_rate'],
            config['loss'],
            config['metrics'])

    return model


def train(model_name, config, output_dir, training_data, validation_data):
    print('Training...')
    print_config_summary(config)

    strategy = create_strategy()
    model = build_and_compile_model(model_name, strategy, config)
    print('Compiled model')

    print('Training RCAN model')

    steps_per_epoch = config['steps_per_epoch'] // basics.get_gpu_count()
    validation_steps = None if validation_data is None else steps_per_epoch
    if validation_data is not None:
        checkpoint_filepath = 'weights_{epoch:03d}_{val_loss:.8f}.hdf5'
    else:
        checkpoint_filepath = 'weights_{epoch:03d}_{loss:.8f}.hdf5'

    model.fit(
        x=training_data,
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

    final_weights_path = str(pathlib.Path(output_dir) / 'weights_final.hdf5')
    model.save_weights(final_weights_path)

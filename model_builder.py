import tensorflow as tf


# Local dependencies
import metrics
import rcan
import care


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
import tensorflow as tf


# Local dependencies
import metrics
import rcan
import care
import srgan
import RESNET
import WUnet
import UnetRCAN

def create_strategy():
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')

    return strategy


def compile_model(model, initial_learning_rate, loss_name, metric_names, loss_alpha = 0, filter_size=11,filter_sigma=1.5):
    print('=== Compiling model ------------------------------------------------')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate),
        loss=metrics.lookup_loss(loss_name, alpha=loss_alpha,filter_size=filter_size,filter_sigma=filter_sigma),
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
        elif model_name == 'srgan':
            # For SRGAN we need an adversarial network for content loss. VGG-19 only accepts RGB images so we use CARE 
            model = care.build_care(config, 'SXYC')
            model = compile_model(
                model,
                config['initial_learning_rate'],
                'ssiml1_loss',
                config['metrics'],
                config['loss_alpha'],
                config['ssim_FSize'],
                config['ssim_FSig'])
            generator, discriminator = srgan.build_and_compile_srgan(config)
            return generator, discriminator, model
        elif model_name == 'resnet':
            model = RESNET.build_and_compile_RESNET(config)
            return model
        elif model_name == 'wunet':
            model = WUnet.build_and_compile_WUnet(config)
            return model
        elif model_name == 'UnetRCAN':
            model = UnetRCAN.build_and_compile_UNetRCAN(config)
            return model
        else:
            raise ValueError(f'Non-implemented model: {model_name}')

        if model_name != 'srgan':
            model = compile_model(
                model,
                config['initial_learning_rate'],
                config['loss'],
                config['metrics'],
                config['loss_alpha'],
                config['ssim_FSize'],
                config['ssim_FSig'])

        return model
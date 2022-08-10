import tensorflow as tf
from csbdeep.internals import nets
from csbdeep.models import Config, CARE


def build_care(shared_config, axes):
    print("=== Building CARE model... -------------------------------------------------")

    # We work with just one channel at a time.
    n_channel_in = 1
    # We are just denoising, so the output is the same shape as the input
    n_channel_out = n_channel_in

    # TODO: Move more of these the shared config.
    config = Config(axes, n_channel_in, n_channel_out,
                    train_steps_per_epoch=shared_config['steps_per_epoch'], train_batch_size=50,
                    train_epochs=shared_config['epochs'], unet_n_depth=config['unet_n_depth'],
                    unet_n_first=config['unet_n_first'], unet_kern_size=config['unet_kern_size'],
                    train_learning_rate=shared_config['initial_learning_rate'])

    print(f'Using config: {config}\n')

    model = nets.common_unet(
        n_dim=config.n_dim,
        n_channel_out=config.n_channel_out,
        prob_out=config.probabilistic,
        residual=config.unet_residual,
        n_depth=config.unet_n_depth,
        kern_size=config.unet_kern_size,
        n_first=config.unet_n_first,
        last_activation=config.unet_last_activation,
    )(config.unet_input_shape)

    print('--------------------------------------------------------------------')

    return model

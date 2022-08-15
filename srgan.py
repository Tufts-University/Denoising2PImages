# Based on: https://medium.com/analytics-vidhya/implementing-srresnet-srgan-super-resolution-with-tensorflow-89900d2ec9b2
# Paper: https://arxiv.org/pdf/1609.04802.pdf (SRGAN)

import tensorflow as tf


# === SRResNet ===

def residual_block_gen(ch=64, k_s=3, st=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),

        tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.PReLU(),
    ])

    return model


def upsample_block(x, ch=256, k_s=3, st=1):
    x = tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same')(x)
    x = tf.nn.depth_to_space(x, 2)  # Subpixel pixelshuffler
    x = tf.keras.layers.PReLU()(x)

    return x


def build_generator():
    '''
    Builds the SRResNet model. Because it relies on convolutions, it can
    operate on any input size.
    '''

    input_lr = tf.keras.layers.Input(shape=(None, None, 3))
    input_conv = tf.keras.layers.Conv2D(64, 9, padding='same')(input_lr)
    input_conv = tf.keras.layers.PReLU()(input_conv)
    SRRes = input_conv  # First skip connection

    # Create 5 residual blocks and establish the skip connections
    # between them.
    for x in range(5):
        res_output = residual_block_gen()(SRRes)
        SRRes = tf.keras.layers.Add()([SRRes, res_output])

    # Add the last convolution, BN, and skip from input.
    SRRes = tf.keras.layers.Conv2D(64, 9, padding='same')(SRRes)
    SRRes = tf.keras.layers.BatchNormalization()(SRRes)
    SRRes = tf.keras.layers.Add()([SRRes, input_conv])

    # Two upsample blocks.
    SRRes = upsample_block(SRRes)
    SRRes = upsample_block(SRRes)
    output_sr = tf.keras.layers.Conv2D(
        3, 9, activation='tanh', padding='same')(SRRes)  # TODO: Check activation.

    SRResnet = tf.keras.models.Model(input_lr, output_sr)
    return SRResnet


# === Discriminator ===

def residual_block_disc(ch=64, k_s=3, st=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
    ])
    return model


def build_discriminator():
    input_lr = tf.keras.layers.Input(shape=(128, 128, 3))  # FIXME: Check size.

    channels_and_strides = [
        (64, 1),
        (64, 2),
        (128, 1), (128, 2),
        (256, 1), (256, 2),
        (512, 1), (512, 2)
    ]

    # This does not use residual_block_disc because we don't do batch normalization.
    input_conv = tf.keras.layers.Conv2D(
        channels_and_strides[0][0], channels_and_strides[0][1], padding='same')(input_lr)
    input_conv = tf.keras.layers.LeakyReLU()(input_conv)

    disc = input_conv

    # Build the seven residual blocks.
    for x in range(1, 8):
        disc = residual_block_disc(
            ch=channels_and_strides[x][0], st=channels_and_strides[x][1])(disc)

    disc = tf.keras.layers.Flatten()(disc)
    disc = tf.keras.layers.Dense(1024)(disc)
    disc = tf.keras.layers.LeakyReLU()(disc)
    disc_output = tf.keras.layers.Dense(1, activation='sigmoid')(disc)

    discriminator = tf.keras.models.Model(input_lr, disc_output)
    return discriminator

# Losses


def PSNR(y_true, y_pred):
    mse = tf.reduce_mean((y_true - y_pred) ** 2)
    return 20 * log10(1 / (mse ** 0.5))


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def pixel_MSE(y_true, y_pred):
    return tf.reduce_mean((y_true - y_pred) ** 2)


VGG19 = tf.keras.applications.VGG19(
    weights='imagenet', include_top=False, input_shape=(128, 128, 3))
VGG_i, VGG_j = 2, 2


def VGG_loss(y_hr, y_sr, i_m=2, j_m=2):
    i, j = 0, 0
    accumulated_loss = 0.0
    for l in VGG19.layers:
        cl_name = l.__class__.__name__
        if cl_name == 'Conv2D':
            j += 1
        if cl_name == 'MaxPooling2D':
            i += 1
            j = 0
        if i == i_m and j == j_m:
            break
        y_hr = l(y_hr)
        y_sr = l(y_sr)
        if cl_name == 'Conv2D':
            accumulated_loss += tf.reduce_mean((y_hr-y_sr)**2) * 0.006
    return accumulated_loss


def VGG_loss_old(y_true, y_pred):
    accumulated_loss = 0.0
    for l in VGG19.layers:
        y_true = l(y_true)
        y_pred = l(y_pred)
        accumulated_loss += tf.reduce_mean((y_true-y_pred)**2) * 0.006
    return accumulated_loss


cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(0.001)
discriminator_optimizer = tf.keras.optimizers.Adam(0.001)
adv_ratio = 0.001
evaluate = ['PSNR']
# MSE
loss_func, adv_learning = pixel_MSE, False
# VGG2.2
loss_func, adv_learning = lambda y_hr, h_sr: VGG_loss(
    y_hr, y_sr, i_m=2, j_m=2), False
# VGG 5.4
loss_func, adv_learning = lambda y_hr, h_sr: VGG_loss(
    y_hr, y_sr, i_m=5, j_m=4), False
# SRGAN-MSE
loss_func, adv_learning = pixel_MSE, True
# SRGAN-VGG 2.2
loss_func, adv_learning = lambda y_hr, h_sr: VGG_loss(
    y_hr, y_sr, i_m=2, j_m=2), True
# SRGAN-VGG 5.4
loss_func, adv_learning = lambda y_hr, h_sr: VGG_loss(
    y_hr, y_sr, i_m=5, j_m=4), True
loss_func, adv_learning = lambda y_hr, h_sr: VGG_loss(
    y_hr, y_sr, i_m=5, j_m=4), True
# Real loss
loss_func, adv_learning = pixel_MSE, False


# Training

# TODOs:
#
# - Add model in main.py
#
# - Compare to my training function
# - Modify relevant functions very prototypically (prolly fit_model)
# - Find data generator sizes & discriminator sizes
#   - Print final output from generator 
#   - Prolly same as input but check conv to see if it makes sense
# - Run to debug on cluster
#

@tf.function()
def train_step(data, loss_func=pixel_MSE, adv_learning=True, evaluate=['PSNR'], adv_ratio=0.001):
    logs = {}
    gen_loss, disc_loss = 0, 0
    low_resolution, high_resolution = data
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        super_resolution = SRResnet(low_resolution, training=True)
        gen_loss = loss_func(high_resolution, super_resolution)
        logs['reconstruction'] = gen_loss
        if adv_learning:
            real_output = discriminator(high_resolution, training=True)
            fake_output = discriminator(super_resolution, training=True)

            adv_loss_g = generator_loss(fake_output) * adv_ratio
            gen_loss += adv_loss_g

            disc_loss = discriminator_loss(real_output, fake_output)
            logs['adv_g'] = adv_loss_g
            logs['adv_d'] = disc_loss
    gradients_of_generator = gen_tape.gradient(
        gen_loss, SRResnet.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, SRResnet.trainable_variables))

    if adv_learning:
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

    for x in evaluate:
        if x == 'PSNR':
            logs[x] = PSNR(high_resolution, super_resolution)
    return logs


def train():
    for x in range(50):
        train_dataset_mapped = train_data.map(build_data,num_parallel_calls=tf.data.AUTOTUNE).batch(128)
        val_dataset_mapped = test_data.map(build_data,num_parallel_calls=tf.data.AUTOTUNE).batch(128)

        for image_batch in tqdm.tqdm(train_dataset_mapped, position=0, leave=True):
            logs=train_step(image_batch,loss_func,adv_learning,evaluate,adv_ratio)
            for k in logs.keys():
                print(k,':',logs[k],end='  ')
                print()
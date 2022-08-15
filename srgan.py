# Based on: https://medium.com/analytics-vidhya/implementing-srresnet-srgan-super-resolution-with-tensorflow-89900d2ec9b2
# Paper: https://arxiv.org/pdf/1609.04802.pdf (SRGAN)

import tensorflow as tf


# Local dependencies

# -

def residual_block_gen(ch=64, k_s=3, st=1):
    # FIXME: Paper uses PReLU instead of LeakyReLU
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
        tf.keras.layers.BatchNormalization(),

        # FIXME: Paper doesn't have leaky RELU here;
        # just an element-wise add with the previous skip connection
        tf.keras.layers.LeakyReLU(),
    ])
    return model


def upsample_block(x, ch=256, k_s=3, st=1):
    x = tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same')(x)

    # FIXME: Paper has BN not depth_to_space
    x = tf.nn.depth_to_space(x, 2)  # Subpixel pixelshuffler

    x = tf.keras.layers.LeakyReLU()(x)
    return x


# === SRResNet ===

def build_generator():
    '''
    Builds the SRResNet model. Because it relies on convolutions, it can
    operate on any input size.
    '''

    input_lr=tf.keras.layers.Input(shape=(None,None,3))
    input_conv=tf.keras.layers.Conv2D(64,9,padding='same')(input_lr)
    input_conv=tf.keras.layers.LeakyReLU()(input_conv)
    SRRes=input_conv

    for x in range(5):
        res_output=residual_block_gen()(SRRes)
        SRRes=tf.keras.layers.Add()([SRRes,res_output])
        SRRes=tf.keras.layers.Conv2D(64,9,padding='same')(SRRes)
        SRRes=tf.keras.layers.BatchNormalization()(SRRes)
        SRRes=tf.keras.layers.Add()([SRRes,input_conv])
        SRRes=Upsample_block(SRRes)
        SRRes=Upsample_block(SRRes)
        output_sr=tf.keras.layers.Conv2D(3,9,activation='tanh',padding='same')(SRRes)
        SRResnet=tf.keras.models.Model(input_lr,output_sr)


# === Discriminator ===

def residual_block_disc(ch=64,k_s=3,st=1):
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(ch,k_s,strides=(st,st),padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
    ])
    return model


def build_discriminator():
    input_lr = tf.keras.layers.Input(shape=(128,128,3))
    input_conv = tf.keras.layers.Conv2D(64,3,padding='same')(input_lr)
    input_conv = tf.keras.layers.LeakyReLU()(input_conv)
    
    channel_nums = [64,128,128,256,256,512,512]
    stride_sizes = [2,1,2,1,2,1,2]

    disc = input_conv

    for x in range(7):
        disc = residual_block_disc(ch = channel_nums[x],st=stride_sizes[x])(disc)
    
    disc = tf.keras.layers.Flatten()(disc)
    disc = tf.keras.layers.Dense(1024)(disc)
    disc = tf.keras.layers.LeakyReLU()(disc)
    disc_output = tf.keras.layers.Dense(1,activation='sigmoid')(disc)
    
    discriminator = tf.keras.models.Model(input_lr,disc_output)
    return discriminator
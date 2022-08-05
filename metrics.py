# Contains the metrics and loss functions used in the models.

import keras
import tensorflow as tf

def ssim(y_true, y_pred):
    '''
    Computes the structural similarity index between two images. Note that the
    maximum signal value is assumed to be 1.
    References
    ----------
    Image Quality Assessment: From Error Visibility to Structural Similarity
    https://doi.org/10.1109/TIP.2003.819861
    '''

    return tf.ssim(y_true, y_pred, 1, k2=0.05)

def SSIM_loss(y_true, y_pred):
    return 1-((ssim(y_true, y_pred)+1)*0.5)
    
def SSIML1_loss(y_true, y_pred):
    SSIM = 1-((ssim(y_true, y_pred)+1)*0.5)
    MAE = keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
    alpha = 0.84

    return alpha * SSIM + (1-alpha) * MAE

def psnr(y_true, y_pred):
    '''
    Computs the peak signal-to-noise ratio between two images. Note that the
    maximum signal value is assumed to be 1.
    '''
    p, q = [K.batch_flatten(y) for y in [y_true, y_pred]]
    psnr2 = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr2
    # return -4.342944819 * K.log(K.mean(K.square(p - q), axis=-1))

def SSIMR2_loss(y_true, y_pred):
    def R_squared(y_true, y_pred):
        residual = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
        mean = tf.reduce_mean(y_true)
        print(f'residual: {residual}; mean: {mean}')
        total = tf.reduce_sum(tf.math.squared_difference(y_true, mean))
        r2 = 1 - residual/total
        print(r2)
        return 1 - r2

    SSIM = 1-((ssim(y_true, y_pred)+1)*0.5)
    R2 = R_squared(y_true, y_pred)
    alpha = 0.2
    return SSIM*alpha + (1-alpha)*R2
    
def MSSSIM_loss(y_true, y_pred):
    w = (0.0448, 0.2856, 0.3001, 0.2363)
    SSIM = 1 - tf.msssim(y_true, y_pred, 1, filter_size=11, power_factors = w, filter_sigma=1.5, k2=0.05)
    MAE = keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
    alpha = 0.84
    return alpha * SSIM + (1-alpha) * MAE

def mae(y_true, y_pred):
    '''
    Computes the mean absolute error between two images.
    '''
    return keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])

def mse(y_true, y_pred):
    '''
    Computes the mean squared error between two images.
    '''
    return keras.losses.mse(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
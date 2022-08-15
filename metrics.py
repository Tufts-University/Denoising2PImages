# Contains the metrics and loss functions used in the models.

import keras
from keras import backend as kb
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

    return tf.image.ssim(y_true, y_pred, 1, k2=0.05)


def ssim_loss(y_true, y_pred):
    return 1-((ssim(y_true, y_pred)+1)*0.5)


def ssiml1_loss(y_true, y_pred):
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
    p, q = [kb.batch_flatten(y) for y in [y_true, y_pred]]
    psnr2 = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr2
    # return -4.342944819 * kb.log(kb.mean(kb.square(p - q), axis=-1))


def ssimr2_loss(y_true, y_pred):
    def R_squared(y_true, y_pred):
        residual = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
        mean = tf.reduce_mean(y_true)
        print(f'residual: {residual}; mean: {mean}')
        total = tf.reduce_sum(tf.math.squared_difference(y_true, mean))
        r2 = 1 - residual/total
        print(r2)
        return 1 - r2

    SSIM = ssim_loss(y_true, y_pred)
    R2 = R_squared(y_true, y_pred)
    alpha = 0.5
    return SSIM*alpha + (1-alpha)*R2


def MSSSIM_loss(y_true, y_pred):
    w = (0.0448, 0.2856, 0.3001, 0.2363)
    # FIXME: Check arguments & call.
    SSIM = 1 - tf.image.ssim_multiscale(y_true, y_pred, 1, filter_size=11,
                                        power_factors=w, filter_sigma=1.5, k2=0.05)
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


def pcc(y_true, y_pred):
    '''
    Computes the Pearson correlation coefficient between two images.
    Returns a value in the range [-1, 1].
    '''
    x = y_true
    y = y_pred

    mx = tf.reduce_mean(x, axis=0)
    my = tf.reduce_mean(y, axis=0)
    # xm = x - x̄, ym = y - ȳ
    xm, ym = x - mx, y - my

    r_num = tf.reduce_sum(xm * ym)

    x_square_sum = tf.reduce_sum(tf.square(xm))
    y_square_sum = tf.reduce_sum(tf.square(ym))
    r_den = tf.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den

    return tf.reduce_mean(r)


def pcc_loss(y_true, y_pred):
    '''
    Computes (1 - pcc)/2, producing a value in the range [0, 1].
    '''
    return (1 - pcc(y_true, y_pred)) / 2


def ssimpcc_loss(y_true, y_pred):
    '''
    A loss combining the ssim and pcc losses. The resulting
    value is in the [0, 2] range.
    '''
    SSIM = ssim_loss(y_true, y_pred)
    PCC = pcc_loss(y_true, y_pred)
    alpha = 0.84

    return alpha*SSIM + (1-alpha)*PCC


def lookup_metrics(metric_names):
    metric_dict = {
        'psnr': psnr,
        'ssim': ssim,
        'pcc': pcc,
    }

    return [metric_dict[metric_name] for metric_name in metric_names]


def lookup_loss(loss_name):
    loss_dict = {
        'mae': mae,
        'mse': mse,
        'ssim_loss': ssim_loss,
        'ssiml1_loss': ssiml1_loss,
        'ssimr2_loss': ssimr2_loss,
        'pcc_loss': pcc_loss,
        'ssimpcc_loss': ssimpcc_loss,
    }

    return loss_dict[loss_name]
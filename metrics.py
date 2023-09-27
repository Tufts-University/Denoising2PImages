# Contains the metrics and loss functions used in the models.

import keras
import tensorflow as tf
from tf_focal_frequency_loss import FocalFrequencyLoss as FFL
import scipy
import cv2 as cv

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

def ssim(y_true, y_pred,filter_size=11,filter_sigma=1.5):
    '''
    Computes the structural similarity index between two images. Note that the
    maximum signal value is assumed to be 1.
    References
    ----------
    Image Quality Assessment: From Error Visibility to Structural Similarity
    https://doi.org/10.1109/TIP.2003.819861
    '''
    print(f'Using a Filter Size {filter_size} and a Filter Sigma of {filter_sigma}')
    return tf.image.ssim(y_true, y_pred, 1, filter_size, filter_sigma, k2=0.05)


def ssim_loss(y_true, y_pred, filter_size=11,filter_sigma=1.5):
    return 1-((ssim(y_true, y_pred, filter_size, filter_sigma)+1)*0.5)

# Sourced from Tensorflow Implementation of Focal Frequency Loss for Image Reconstruction and Synthesis [ICCV 2021]
# https://github.com/ZohebAbai/tf-focal-frequency-loss
ffl = FFL(loss_weight=1.0, alpha=1.0)
def ffloss(y_true,y_pred):
    return ffl(y_pred,y_true)

def SSIMFFL(y_true,y_pred, alpha, filter_size,filter_sigma):
    SSIM = 1-((ssim(y_true, y_pred,filter_size,filter_sigma)+1)*0.5)
    FFL = ffloss(y_true,y_pred)
    return alpha * SSIM + (1-alpha) * FFL

# Default alpha was 0.84
def ssiml1_loss(y_true, y_pred, alpha, filter_size,filter_sigma):
    SSIM = 1-((ssim(y_true, y_pred,filter_size,filter_sigma)+1)*0.5)
    MAE = keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])

    return alpha * SSIM + (1-alpha) * MAE

def ssiml2_loss(y_true, y_pred,alpha,filter_size,filter_sigma):
    SSIM = 1-((ssim(y_true, y_pred,filter_size,filter_sigma)+1)*0.5)
    MSE = keras.losses.mse(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
    return alpha * SSIM + (1-alpha) * MSE

def psnr(y_true, y_pred):
    '''
    Computs the peak signal-to-noise ratio between two images. Note that the
    maximum signal value is assumed to be 1.
    '''
    # p, q = [kb.batch_flatten(y) for y in [y_true, y_pred]]
    psnr2 = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return psnr2
    # return -4.342944819 * kb.log(kb.mean(kb.square(p - q), axis=-1))


def ssimr2_loss(y_true, y_pred, alpha, filter_size,filter_sigma):
    def R_squared(y_true, y_pred):
        residual = tf.reduce_sum(tf.math.squared_difference(y_true, y_pred))
        mean = tf.reduce_mean(y_true)
        print(f'residual: {residual}; mean: {mean}')
        total = tf.reduce_sum(tf.math.squared_difference(y_true, mean))
        r2 = 1 - residual/total
        print(r2)
        return 1 - r2

    SSIM = ssim_loss(y_true, y_pred, filter_size,filter_sigma)
    R2 = R_squared(y_true, y_pred)
    return SSIM*alpha + (1-alpha)*R2


# Default alpha was 0.84
def MSSSIM_loss(y_true, y_pred, alpha):
    w = (0.0448, 0.2856, 0.3001, 0.2363)
    # FIXME: Check arguments & call.
    SSIM = 1 - tf.image.ssim_multiscale(y_true, y_pred, 1, filter_size=11,
                                        power_factors=w, filter_sigma=1.5, k2=0.05)
    MAE = keras.losses.mae(
        *[keras.backend.batch_flatten(y) for y in [y_true, y_pred]])
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


def ssimpcc_loss(y_true, y_pred, alpha, filter_size,filter_sigma):
    '''
    A loss combining the ssim and pcc losses. The resulting
    value is in the [0, 2] range.
    '''
    SSIM = ssim_loss(y_true, y_pred, filter_size,filter_sigma)
    PCC = pcc_loss(y_true, y_pred)

    return alpha*SSIM + (1-alpha)*PCC

@tf.function
def calculate_content_loss(hr, sr, perceptual_model):
    sr_features = perceptual_model(sr) / 12.75
    hr_features = perceptual_model(hr) / 12.75
    return mse(hr_features, sr_features)

def calculate_generator_loss(sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out)

def calculate_discriminator_loss(hr_out, sr_out):
    hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)
    sr_loss = binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
    return hr_loss + sr_loss

def tf_equalize_histogram(image):
    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return eq_hist

@tf.function
def guassian_bpf(image,LFC,HFC):
    f = tf.cast(image,dtype = tf.float64)
    n = f.get_shape()
    nx,ny = n[0],n[1]
    paddings = tf.constant([[nx//2, nx//2+1], [ny//2, ny//2+1]])

    f = tf.pad(f,paddings,"CONSTANT")
    fftI = tf.signal.fft2d(f)
    fftI = tf.signal.fftshift(f)

    # Intialize filters
    filter1, filter2, filter3 = tf.ones_like(f), tf.ones_like(f), tf.ones_like(f)
    for i in range(0:2*nx):
        for j in range(0:2*ny):
            dist = tf.constant(((i-(nx+1))^2 + (j-(ny+1))^2)^0.5,dtype = tf.float64)
            filter1[i,j] = tf.math.exp(-dist^2/(2*HFC^2)) # higher cut off (cuts everything above that)
            filter2[i,j] = tf.math.exp(-dist^2/(2*LFC^2)) # low cut off (cuts everything above that)
            filter3[i,j] = tf.cast(1,dtype= tf.float64)-filter2 # invert lower cut off so we remove all below
            filter3[i,j] = filter3[i,j] * filter1[i,j] # multiple 1st and 3rd filter to get bandpass
    filtered_image = fftI + filter3*fftI
    filtered_image = tf.signal.ifftshift(filtered_image)
    filtered_image = tf.pad(filtered_image,paddings,"CONSTANT")
    filtered_image = tf.signal.ifft2d(filtered_image)
    filtered_image = tf.math.real(filtered_image[0:nx-1,0:ny-1])
    return filtered_image

@tf.function
def butterworth_bpf(image,LFC,HFC,order):
    f = tf.cast(image,dtype = tf.float64)
    n = f.get_shape()
    nx,ny = n[0],n[1]
    paddings = tf.constant([[nx//2, nx//2+1], [ny//2, ny//2+1]])

    f = tf.pad(f,paddings,"CONSTANT")
    fftI = tf.signal.fft2d(f)
    fftI = tf.signal.fftshift(f)

    # Intialize filters
    filter1, filter2, filter3 = tf.ones_like(f), tf.ones_like(f), tf.ones_like(f)
    for i in range(0:2*nx):
        for j in range(0:2*ny):
            dist = tf.constant(((i-(nx+1))^2 + (j-(ny+1))^2)^0.5,dtype = tf.float64)
            filter1[i,j] = tf.constant(1/(1+(dist/HPF)^(2/*order)),dtype = tf.float64) # higher cut off (cuts everything above that)
            filter2[i,j] = tf.constant(1/(1+(dist/LPF)^(2/*order)),dtype = tf.float64) # low cut off (cuts everything above that)
            filter3[i,j] = tf.cast(1,dtype= tf.float64)-filter2 # invert lower cut off so we remove all below
            filter3[i,j] = filter3[i,j] * filter1[i,j] # multiple 1st and 3rd filter to get bandpass
    filtered_image = fftI + filter3*fftI
    filtered_image = tf.signal.ifftshift(filtered_image)
    filtered_image = tf.pad(filtered_image,paddings,"CONSTANT")
    filtered_image = tf.signal.ifft2d(filtered_image)
    filtered_image = tf.math.real(filtered_image[0:nx-1,0:ny-1])
    return filtered_image

@tf.function
def Otsu_filter(image):
    noise_removal_threshold = 25
    image = cv.cvtColor(image.numpy(), cv.COLOR_BGR2GRAY)
    _,thresh = cv.threshold(image,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    mask = np.zeros_like(thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
      area = cv.contourArea(contour)
      if area > noise_removal_threshold:
        cv.fillPoly(mask, [contour], 255)
    
    return tf.convert_to_tensor(mask,dtype=float64)


def Cytoplasm_mask(y_true,y_pred):
    # Adaptive hist equilization
    y_true_hist, y_pred_hist = tf_equalize_histogram(y_true), tf_equalize_histogram(y_pred)
    # Guassian Bandpass Filter
    y_true_bpf, y_pred_bpf = guassian_bpf(y_true_hist,25,256), guassian_bpf(y_pred_hist,25,256) 
    y_true_bpf2, y_pred_bpf2 = guassian_bpf(y_true_bpf,20,120), guassian_bpf(y_pred_bpf,20,120) 
    # Butterworth Bandpass Filter
    y_true_bpf3, y_pred_bpf3 = guassian_bpf(y_true_bpf2,10,120,3), guassian_bpf(y_pred_bpf2,10,120,3) 
    # Normalization
    y_true_bpf3, y_pred_bpf3 = y_true_bpf3 - tf.math.reduce_min(y_true_bpf3), y_pred_bpf3 - tf.math.reduce_min(y_pred_bpf3)
    y_true_norm, y_pred_norm = y_true_bpf3 / tf.math.reduce_max(y_true_bpf3), y_pred_bpf3 / tf.math.reduce_max(y_pred_bpf3)
    # Thresholding
    y_true_cyto, y_pred_cyto = Otsu_filter(y_true_norm), Otsu_filter(y_pred_norm)
    return y_true_cyto,y_pred_cyto

def RR_loss(y_true, y_pred):
    y_true_N, y_true_F = y_true
    y_pred_N, y_pred_F = y_pred
    # Generate Mask
    y_true_cyto, y_pred_cyto = Cytoplasm_mask(y_true_N,y_pred_N)
    y_NADH_true = y_true_N * y_true_cyto
    y_FAD_true = y_true_F * y_true_cyto
    y_NADH_pred = y_pred_N * y_pred_cyto
    y_FAD_pred = y_pred_F * y_pred_cyto
    RR_true = y_FAD_true/(y_FAD_true+y_NADH_true)
    RR_pred = y_FAD_pred/(y_FAD_pred+y_NADH_pred)
    
    MSE = keras.losses.mse(
        *[keras.backend.batch_flatten(y) for y in [RR_true, RR_pred]])

    return MSE
# Lookup --------------------------------------------------------------------


def lookup_metrics(metric_names):
    metric_dict = {
        'psnr': psnr,
        'ssim': ssim,
        'pcc': pcc,
    }

    return [metric_dict[metric_name] for metric_name in metric_names]


def lookup_loss(loss_name, alpha = 0, filter_size=11 , filter_sigma=1.5):
    print(f'Found a loss alpha of {alpha}.')
    
    loss_dict = {
        'mae': mae,
        'mse': mse,
        'ssim_loss': ssim_loss,
        'ssiml1_loss': lambda y_true, y_pred: ssiml1_loss(y_true, y_pred, alpha, filter_size, filter_sigma),
        'ssiml2_loss': lambda y_true, y_pred: ssiml2_loss(y_true, y_pred, alpha, filter_size, filter_sigma),
        'ssimr2_loss': lambda y_true, y_pred: ssimr2_loss(y_true, y_pred, alpha, filter_size, filter_sigma),
        'pcc_loss': pcc_loss,
        'ffloss': ffloss,
        'SSIMFFL': lambda y_true, y_pred: SSIMFFL(y_true, y_pred, alpha, filter_size, filter_sigma),
        'MSSSIM_loss': lambda y_true, y_pred: MSSSIM_loss(y_true, y_pred, alpha),
        'ssimpcc_loss': lambda y_true, y_pred: ssimpcc_loss(y_true, y_pred, alpha, filter_size, filter_sigma),
    }

    return loss_dict[loss_name]

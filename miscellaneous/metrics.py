import numpy as np
import tensorflow as tf


def RE(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))

    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def RMSE(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))

    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def PSNR(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1,))
    y_pred = tf.reshape(y_pred, shape=(-1,))

    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def SSIM(y_true, y_pred):
    if len(y_true.shape) == 2:
        y_true = np.reshape(y_true, y_true.shape + (1,))
    if len(y_pred.shape) == 2:
        y_pred = np.reshape(y_pred, y_pred.shape + (1,))

    y_true = tf.cast(y_true, tf.double)
    y_pred = tf.cast(y_pred, tf.double)

    return tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def SSIM_loss(y_true, y_pred):
    return 1 - SSIM(y_true[:, :, :, :1], y_pred)

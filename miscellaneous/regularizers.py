import tensorflow as tf

def TV(x):
    """
    Implements the total variation regularizer, defined as:
    
    TV(x) = || \sqrt(D_h(x)^2 + D_v(x)^2) ||_1.
    """
    # To compute the TV of x, we first compute the two directional gradients.
    dy, dx = tf.image.image_gradients(x)

    # Compute the magnitude of the gradient
    epsilon = 1e-8
    Dx = tf.math.sqrt(tf.square(dx) + tf.square(dy) + epsilon)

    # Compute the TV
    tv = tf.math.reduce_sum(Dx)
    return tv

def GenTV(x, z):
    """
    Implements the total variation regularizer centered in z, defined as:
    
    TV_z(x) = || \sqrt(D_h(x-z)^2 + D_v(x-z)^2) ||_1.
    """
    # Update x as the difference between x and the true x
    x_new = x - z

    # To compute the TV of x, we first compute the two directional gradients.
    dy, dx = tf.image.image_gradients(x_new)

    # Compute the magnitude of the gradient
    epsilon = 1e-8
    Dx = tf.math.sqrt(tf.square(dx) + tf.square(dy) + epsilon)

    # Compute the TV
    tv = tf.math.reduce_sum(Dx)
    return tv

def Tik(x):
    """
    Implements the Tikhonov regularizer, defined as:
    
    Tik(x) = || x ||_2^2.
    """
    # Compute the two-norm-squared of x

    tik = tf.math.reduce_sum(tf.square(x))
    return tik

def GenL1Tik(x, z):
    """
    Implements the L1-Tikhonov regularizer centered in z, defined as:
    
    Tik(x) = || x - z ||_1.
    """
    # Update x as the difference between x and the true x
    x_new = x - z

    # Compute the two-norm-squared of x
    tik = tf.math.reduce_sum(tf.abs(x_new))
    return tik
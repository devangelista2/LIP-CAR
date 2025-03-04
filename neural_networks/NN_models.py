from tensorflow import keras as ks


# UNet
def get_Unet(input_shape):
    """
    Define the UNet Model, following the paper A Residual Dense U-Net Neural Network for Image Denoising.

    input_shape -> Tuple, input dimension
    n_scales -> Number of downsampling
    conv_per_scale -> Number of convolutions for each scale
    init_conv -> Number of convolutional filters at the first scale
    """
    n_ch = 64
    conv_per_scale = 2
    n_scales = 4
    skips = []

    x = ks.layers.Input(input_shape)
    h = x

    # ANALYSIS
    for scale in range(n_scales):
        for c in range(conv_per_scale):
            h = ks.layers.Conv2D(n_ch, 3, 1, padding="same")(h)
            h = ks.layers.BatchNormalization()(h)
            h = ks.layers.ReLU()(h)

        skips.append(h)
        h = ks.layers.MaxPooling2D()(h)
        n_ch = n_ch * 2

    # FILTERING
    for c in range(conv_per_scale):
        h = ks.layers.Conv2D(n_ch, 3, 1, padding="same")(h)
        h = ks.layers.BatchNormalization()(h)
        h = ks.layers.ReLU()(h)

    n_ch = n_ch // 2
    h = ks.layers.Conv2DTranspose(n_ch, 3, 1, padding="same")(h)
    h = ks.layers.UpSampling2D()(h)

    # SYNTHESIS
    for scale in range(n_scales):
        h = ks.layers.Concatenate()([h, skips.pop(-1)])
        for c in range(conv_per_scale):
            h = ks.layers.Conv2D(n_ch, 3, 1, padding="same")(h)
            h = ks.layers.BatchNormalization()(h)
            h = ks.layers.ReLU()(h)

        if scale < n_scales - 1:
            n_ch = n_ch // 2
            h = ks.layers.Conv2DTranspose(n_ch, 3, 1, padding="same")(h)
            h = ks.layers.UpSampling2D()(h)

    y = ks.layers.Conv2D(1, 1, 1, padding="same", activation="tanh")(h)
    y = ks.layers.ReLU()(y)

    return ks.models.Model(x, y)


def get_Unet_Bone(input_shape):
    """
    input_shape -> Tuple, input dimension
    init_conv -> Number of convolutional filters at the first scale
    """

    n_ch = 12
    skips = []

    x = ks.layers.Input(input_shape)
    h = x

    # First Block ANALYSIS
    h = ks.layers.Conv2D(
        filters=n_ch, kernel_size=(1, 1), strides=(1, 1), padding="same"
    )(h)
    z1 = h
    h = ks.layers.Conv2D(
        filters=n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z1])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    skips.append(h)
    h = ks.layers.Conv2D(
        filters=2 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Second Block ANALYSIS
    z2 = h
    h = ks.layers.Conv2D(
        filters=2 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=2 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z2])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    skips.append(h)
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Third Block ANALYSIS
    z3 = h
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z3])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    skips.append(h)
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Fourth Block ANALYSIS
    z4 = h
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z4])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    skips.append(h)
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)

    # FILTERING
    z5 = h
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z5])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    h = ks.layers.Conv2DTranspose(
        filters=8 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)

    # First Block SYNTHESIS
    h = ks.layers.Concatenate()([h, skips.pop(-1)])
    z6 = h
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=16 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z6])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    h = ks.layers.Conv2DTranspose(
        filters=4 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Second Block SYNTHESIS
    h = ks.layers.Concatenate()([h, skips.pop(-1)])
    z7 = h
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=8 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z7])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    h = ks.layers.Conv2DTranspose(
        filters=2 * n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Third Block SYNTHESIS
    h = ks.layers.Concatenate()([h, skips.pop(-1)])
    z8 = h
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=4 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z8])
    h = ks.layers.LeakyReLU(alpha=0.2)(h)
    h = ks.layers.Conv2DTranspose(
        filters=n_ch, kernel_size=(2, 2), strides=(2, 2), padding="same"
    )(h)
    # Fourth Block SYNTHESIS
    h = ks.layers.Concatenate()([h, skips.pop(-1)])
    z9 = h
    h = ks.layers.Conv2D(
        filters=2 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Conv2D(
        filters=2 * n_ch, kernel_size=(3, 3), strides=(1, 1), padding="same"
    )(h)
    h = ks.layers.Add()([h, z9])
    y = ks.layers.Conv2D(
        filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same", activation="ReLU"
    )(h)

    return ks.models.Model(x, y)

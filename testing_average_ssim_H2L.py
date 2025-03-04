import os

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

from miscellaneous import metrics, regularizers, utilities

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


######## Initialization #############################################################
# Choose the Forward (H2L) or Backward (L2H) operator
# the options are: 'InvH2L'

operator = "InvH2L"

# Choose the architecture
# the options are: 'Unet', 'ConvexUnet', or 'Unet_Bone'

architecture = "Unet_Bone"
architecture_l2h = "Unet_Bone"

# Choose the pre-dose
# the options are: False or True

predose = True
predose_l2h = True

# Choose if apply BM3D to the low-dose images
# the options are: False or True

BM3D_lowdose = True
BM3D_lowdose_l2h = False

# Choose the metric for the loss function
# the options are: 'MSE', 'MAE' or 'SSIM'

loss = "SSIM"
loss_l2h = "SSIM"

# Choose which dataset to test
# the options are: 'train' or 'test'

train_or_test = "test"

# Set the corruption level.

delta = 0
delta_suffix = str(delta).split(".")[-1]

#####################################################################################
# Set the data path
P = ""
if predose:
    P = "P"

P_l2h = ""
if predose_l2h:
    P_l2h = "P"

BM3D = ""
if BM3D_lowdose:
    BM3D = "_BM3D"

BM3D_l2h = ""
if BM3D_lowdose_l2h:
    BM3D_l2h = "_BM3D"

# The main operator is an H2L.
x_train_or_test_path = f"./images/highdose/{train_or_test}.npy"
y_train_or_test_path = f"./images/lowdose{BM3D}/{train_or_test}.npy"

predose_train_or_test_path = f"./images/predose/{train_or_test}.npy"

# Load and preprocess data
x_train_or_test = np.load(x_train_or_test_path)
x_train_or_test = utilities.preprocess_array(x_train_or_test)

y_train_or_test = np.load(y_train_or_test_path)
y_train_or_test = utilities.preprocess_array(y_train_or_test)

if predose:
    predose_train_or_test = np.load(predose_train_or_test_path)
    predose_train_or_test = utilities.preprocess_array(predose_train_or_test)

# Consider just N samples from the set
N = 60  # Number of samples
idx_sample = np.random.choice(np.arange(len(x_train_or_test)), size=(N,))

x_train_or_test = x_train_or_test[idx_sample]
y_train_or_test = y_train_or_test[idx_sample]
if predose:
    predose_train_or_test = predose_train_or_test[idx_sample]

# Print shapes

if predose:
    print(x_train_or_test.shape, predose_train_or_test.shape)
else:
    print(x_train_or_test.shape)

# Load the model

h2l_path = f"./model_weights/{P}H2L{BM3D}_{architecture}_{loss}.h5"
l2h_path = f"./model_weights/{P_l2h}L2H{BM3D_l2h}_{architecture_l2h}_{loss_l2h}.h5"

model = ks.models.load_model(
    h2l_path, custom_objects={"SSIM": metrics.SSIM, "SSIM_loss": metrics.SSIM_loss}
)
l2h_model = ks.models.load_model(
    l2h_path, custom_objects={"SSIM": metrics.SSIM, "SSIM_loss": metrics.SSIM_loss}
)


# Define the forward operator
def A(x, x_pre=None):
    # I assume x to have 4 dimensions, with shape (N, m, n, 1)
    if x_pre is not None:
        y = model(tf.concat([x, x_pre], axis=-1))
    else:
        y = model(x)
    return y


def Res(x, x_pre=None):
    # Compute the residue of x
    return tf.math.reduce_sum(tf.square(A(x, x_pre) - y))


def F(x, x_pre=None, lmbda=0.1):
    # The objective function. Potentially, defined as something like
    # F(x) = 0.5 * || A(x) - y_delta ||_2^2 + lambda * R(x)

    # Fidelity
    J = 0.5 * Res(x, x_pre)

    # Regularization function
    if Reg == "GenTV":
        R = regularizers.GenTV(
            x, x_l2h
        )  # in  {TV(x), GenTV(x, z), Tik(x), GenL1Tik(x, z)}
    elif Reg == "TV":
        R = regularizers.TV(x)

    return J + lmbda * R


# Optimization algorithm with custom optimizer
def Optimize(F, x_pre, optimizer, x_0, kmax, x_true=None, info=True):
    # Initialization
    x_k = tf.Variable(x_0)
    loss_vec = np.zeros((kmax + 1,))

    if x_true is not None:
        ssim_vec = np.zeros((kmax + 1,))
        res_vec = np.zeros((kmax + 1,))

    # Compute the starting iterate and the corresponding loss
    loss_vec[0] = F(x_k, x_pre, lmbda)
    if x_true is not None:
        ssim_vec[0] = metrics.SSIM(x_true, x_k)
        res_vec[0] = Res(x_k, x_pre)

    # Run the Gradient Descent
    for k in range(kmax):
        # Let the Tape collect the gradient in the forward pass
        with tf.GradientTape() as tape:
            tape.watch(x_k)
            L = F(x_k, x_pre, lmbda)

        # Compute the gradient
        gradient = tape.gradient(L, x_k)
        loss_vec[k + 1] = L
        if x_true is not None:
            ssim_vec[k + 1] = metrics.SSIM(x_true, x_k)
            res_vec[k + 1] = Res(x_k, x_pre)

        # Update x_k
        optimizer.apply_gradients(zip([gradient], [x_k]))

        # Visualize info if required
        if info:
            if (k + 1) % 10 == 0:
                print(
                    f"Loss at iteration {k+1}: {L}. SSIM at iteration {k+1}: {ssim_vec[k+1]}. Residue at iteration {k+1}: {res_vec[k+1]}."
                )

    return x_k, loss_vec, ssim_vec, res_vec


# Initialize error array
ssim_vec = np.zeros((N,))
ssim_vec_corr = np.zeros((N,))
for i in range(N):
    # Get the data
    x_true = tf.Variable(x_train_or_test[i : i + 1])
    y = tf.Variable(y_train_or_test[i : i + 1])

    if predose:
        x_pre = tf.Variable(predose_train_or_test[i : i + 1])
    else:
        x_pre = None

    if predose_l2h:
        x_l2h = l2h_model(tf.concat([y, x_pre], axis=-1))
    else:
        x_l2h = l2h_model(y)

    # Set parameters
    Reg = "GenTV"
    lmbda = 6e-3
    x_0 = y  # x_l2h OR y
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # Solve the problem
    x_rec, _, _, _ = Optimize(
        F, x_pre, optimizer, x_0=x_0, x_true=x_true, kmax=150, info=False
    )
    x_rec = x_rec[0, :, :, 0]
    ssim_vec[i] = metrics.SSIM(x_train_or_test[i, :, :, 0], x_rec)

    # Corrupt
    y = y + delta * np.random.normal(0, 1, y.shape)

    if predose_l2h:
        x_l2h = l2h_model(tf.concat([y, x_pre], axis=-1))
    else:
        x_l2h = l2h_model(y)

    # Set parameters
    lmbda = np.logspace(np.log10(6e-3), np.log10(0.1), 10)[3]
    x_0 = y  # x_l2h OR y
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # Solve the problem
    x_rec, _, _, _ = Optimize(
        F, x_pre, optimizer, x_0=x_0, x_true=x_true, kmax=150, info=False
    )
    x_rec = x_rec[0, :, :, 0]
    ssim_vec_corr[i] = metrics.SSIM(x_train_or_test[i, :, :, 0], x_rec)

    print(f"{i+1}/{N}. SSIM: {ssim_vec[i]}. SSIM Corr: {ssim_vec_corr[i]}.")
# np.save(f'./results/testing/stability/'
#         f'boxplot_0_{P}{operator}{BM3D}_{architecture}_{loss}_{Reg}_{train_or_test}.npy', ssim_vec)
np.save(
    f"./results/testing/stability/"
    f"boxplot_{delta_suffix}_{P}{operator}{BM3D}_{architecture}_{loss}_{Reg}_{train_or_test}.npy",
    ssim_vec_corr,
)

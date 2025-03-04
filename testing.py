import numpy as np
from tensorflow import keras as ks

from miscellaneous import metrics, utilities

######## Initialization #############################################################
# Choose the Forward (H2L) or Backward (L2H) operator
# the options are: 'L2H' or 'H2L'

operator = "L2H"

# Choose the architecture
# the options are: 'Unet', 'ConvexUnet', or 'Unet_Bone'

architecture = "Unet_Bone"

# Choose the pre-dose
# the options are: False or True

predose = True

# Choose if apply BM3D to the low-dose images
# the options are: False or True

BM3D_lowdose = False

# Choose the metric for the loss function
# the options are: 'MSE', 'MAE' or 'SSIM'

loss = "SSIM"

# Choose which dataset to test
# the options are: 'train' or 'test'

train_or_test = "test"

#####################################################################################
# Set the data path
P = ""
if predose:
    P = "P"

BM3D = ""
if BM3D_lowdose:
    BM3D = "_BM3D"

if operator == "L2H":
    x_train_or_test_path = f"./images/lowdose{BM3D}/{train_or_test}.npy"
    y_train_or_test_path = f"./images/highdose/{train_or_test}.npy"
if operator == "H2L":
    x_train_or_test_path = f"./images/highdose/{train_or_test}.npy"
    y_train_or_test_path = f"./images/lowdose{BM3D}/{train_or_test}.npy"

predose_train_or_test_path = f"./images/predose/{train_or_test}.npy"


# Load and preprocess data
x_train_or_test = np.load(x_train_or_test_path)
x_train_or_test = utilities.preprocess_array(x_train_or_test)


predose_train_or_test = []
if predose:
    predose_train_or_test = np.load(predose_train_or_test_path)
    predose_train_or_test = utilities.preprocess_array(predose_train_or_test)
    x_train_or_test = ks.layers.concatenate(
        [x_train_or_test, predose_train_or_test], axis=-1
    )

# Print shapes

if predose:
    print(x_train_or_test.shape, predose_train_or_test.shape)
else:
    print(x_train_or_test.shape)

# Load the model

model_path = f"./model_weights/{P}{operator}{BM3D}_{architecture}_{loss}.h5"

model = ks.models.load_model(
    model_path, custom_objects={"SSIM": metrics.SSIM, "SSIM_loss": metrics.SSIM_loss}
)

reconstruction = model.predict(x_train_or_test)[:, :, :, 0]


np.save(
    f"./results/testing/{operator}/{architecture}/"
    f"{P}{operator}{BM3D}_{architecture}_{loss}_{train_or_test}.npy",
    reconstruction,
)

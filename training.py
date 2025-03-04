import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras as ks

from miscellaneous import metrics, utilities
from neural_networks import NN_models

######## Initialization #############################################################
# Choose the Forward (H2L) or Backward (L2H) operator
# the options are: 'L2H' or 'H2L'

operator = 'H2L'

# Choose the architecture
# the options are: 'Unet', 'ConvexUnet', or 'Unet_Bone'

architecture = 'Unet_Bone'

# Choose the pre-dose
# the options are: False or True

predose = False

# Choose if apply BM3D to the low-dose images
# the options are: False or True

BM3D_lowdose = True

# Choose the metric for the loss function
# the options are: 'MSE', 'MAE' or 'SSIM'

loss = 'SSIM'

# Set the training and optimization parameters

n_epochs = 85
n_batch = 16

learning_rate = 1e-4
validation = 0.1



##########################################################################################

######### Compute the weights ############################################################

# Set the data path
P = ''
if predose:
    P = 'P'

BM3D = ''
if BM3D_lowdose:
    BM3D = '_BM3D'

if operator == 'L2H':
    x_train_path = f"./images/lowdose{BM3D}/train.npy"
    y_train_path = f'./images/highdose/train.npy'
if operator == 'H2L':
    x_train_path = f'./images/highdose/train.npy'
    y_train_path = f'./images/lowdose{BM3D}/train.npy'

predose_train_path = f'./images/predose/train.npy'



# Load and preprocess train data
x_train = np.load(x_train_path)
x_train = utilities.preprocess_array(x_train)

y_train = np.load(y_train_path)
y_train = utilities.preprocess_array(y_train)

predose_train = []
if predose:
    predose_train = np.load(predose_train_path)
    predose_train = utilities.preprocess_array(predose_train)
    x_train = ks.layers.concatenate([x_train, predose_train], axis=-1)

# Print shapes
_, m, n, c = x_train.shape
if predose:
    print(x_train.shape, y_train.shape, predose_train.shape)
else:
    print(x_train.shape, y_train.shape)




# Build model and compile it

# Define the architecture dictionary
get_architecture = {'Unet': NN_models.get_Unet,
                    'ConvexUnet': NN_models.get_ConvexUnet,
                    'Unet_Bone': NN_models.get_Unet_Bone}

# Define the loss dictionary
loss_function = {'SSIM': metrics.SSIM_loss,
                 'MSE': 'mse',
                 'MAE': 'mae'}


model = get_architecture[architecture](input_shape=(m, n, c))


model.compile(optimizer=ks.optimizers.Adam(learning_rate=learning_rate),
              loss=loss_function[loss],
              metrics=[metrics.SSIM, 'mse'])

# Train
hist = model.fit(x_train, y_train, batch_size=n_batch, epochs=n_epochs, validation_split=validation)


# Save the weights

model.save(f'./model_weights/{P}{operator}{BM3D}_{architecture}_{loss}.h5')

print(f"Training of NN model -> Finished.")

# Extract data from hist
train_mse = hist.history['loss']
val_mse = hist.history['val_loss']

train_SSIM = hist.history['SSIM']
val_SSIM = hist.history['val_SSIM']

# Plot them
plt.figure()
plt.plot(train_mse)
plt.plot(val_mse)
plt.xlabel('epoch')
plt.grid()
plt.legend(['Train MSE', 'Validation MSE'])
plt.tight_layout()
plt.savefig(f'results/training/'
            f'{operator}/{architecture}/{P}{operator}{BM3D}_{architecture}_{loss}_mse.png')

plt.figure()
plt.plot(train_SSIM)
plt.plot(val_SSIM)
plt.xlabel('epoch')
plt.grid()
plt.legend(['Train SSIM', 'Validation SSIM'])
plt.tight_layout()
plt.savefig(f'results/training/'
            f'{operator}/{architecture}/{P}{operator}{BM3D}_{architecture}_{loss}_ssim.png')


import os

import matplotlib.pyplot as plt
import numpy as np


# Preprocess the array before training, testing or visualising
def preprocess_array(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    if len(arr.shape) <= 3:
        return np.reshape(arr, arr.shape + (1, ))
    else:
        return arr

def preprocess_array_vis(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def generate_image(low, high, rec, ground_truth_is_low):
    # Save the result
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 3, 1)
    plt.title(r'lowdose', fontsize=40)
    plt.axis('off')
    plt.imshow(low, cmap='gray', vmin=0, vmax=1)

    plt.subplot(2, 3, 2)
    plt.title(r'reconstruction', fontsize=40)
    plt.axis('off')
    plt.imshow(rec, cmap='gray', vmin=0, vmax=1)

    plt.subplot(2, 3, 3)
    plt.title(r'$highdose$', fontsize=40)
    plt.axis('off')
    plt.imshow(high, cmap='gray', vmin=0, vmax=1)

    if ground_truth_is_low:
        plt.subplot(2, 3, 5)
        plt.title(r'$\| lowdose - reconstruction \|$', fontsize=40)
        plt.axis('off')
        plt.imshow(np.abs(low - rec), cmap='gray', vmin=0, vmax=1)
    else:
        plt.subplot(2, 3, 5)
        plt.title(r'$\| highdose - reconstruction \|$', fontsize=40)
        plt.axis('off')
        plt.imshow(np.abs(high - rec), cmap='gray', vmin=0, vmax=1)


def generate_image_identity(high, rec):
    # Save the result
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 3, 1)
    plt.title(r'highdose', fontsize=30)
    plt.axis('off')
    plt.imshow(high, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1, 3, 2)
    plt.title(r'reconstruction', fontsize=30)
    plt.axis('off')
    plt.imshow(rec, cmap='gray', vmin=0, vmax=1)

    plt.subplot(1, 3, 3)
    plt.title(r'$\| high - rec \|$', fontsize=30)
    plt.axis('off')
    plt.imshow(np.abs(high - rec), cmap='gray', vmin=0, vmax=1)


def exclude_max_min(arr):
    max_idx = arr.argmax()
    new_arr = np.concatenate((arr[:max_idx], arr[max_idx + 1:]))
    min_idx = new_arr.argmin()
    new_arr = np.concatenate((new_arr[:min_idx], new_arr[min_idx + 1:]))
    return new_arr

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def contrast_stretching(image):
    in_min, in_max = np.percentile(image, [0.1, 99.9])
    out_min, out_max = 0, 1 # Normalized range of the image
    
    # Stretch the image intensities
    out_image = (image - in_min) * ((out_max - out_min) / (in_max - in_min)) + out_min
    out_image = np.clip(out_image, out_min, out_max) # Ensure values remain between 0 and 1
    return out_image
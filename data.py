# Let's start by loading the uploaded .mat file to understand its structure and extract the necessary data.
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
import cv2

# Load the .mat file
file_path = 'Data.mat'
data = loadmat(file_path)

# Check the keys in the loaded .mat file to locate the desired variable
data.keys()

# Extract the 'act_in' variable
act_in = data['act_in']
act_out = data['act_out']

# Extract the first slice of 'act_in' - Note: MATLAB uses 1-based indexing, Python uses 0-based indexing.
# So, what is referred to as act_in(:,:,1) in MATLAB corresponds to act_in[:,:,0] in Python.

train_in_dir = "data/train/input_images"
val_in_dir = "data/val/input_images"
train_out_dir = "data/train/output_images"
val_out_dir = "data/val/output_images"

# Create directories if they don't exist
os.makedirs(train_in_dir, exist_ok=True)
os.makedirs(val_in_dir, exist_ok=True)
os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(val_out_dir, exist_ok=True)

# Iterate through the slices and split them into train and val sets
for i in range(act_in.shape[2]):
    act_in_slice = act_in[:, :, i]
    act_out_slice = act_out[:, :, i]

    # # Resize the slice to 256x256 pixels
    act_in_resized = resize(act_in_slice, (256, 256))
    act_out_resized = resize(act_out_slice, (256, 256))
    

    # Determine if the current slice belongs to train or val
    if i < int(0.8 * act_in.shape[2]):
        save_in_dir = train_in_dir
        save_out_dir = train_out_dir
    else:
        save_in_dir = val_in_dir
        save_out_dir = val_out_dir

    # Save the resized slice
    # print("Shape of act_in[:,:,0]:", act_in_slice.shape)
    plt.imsave(os.path.join(save_in_dir, f"act_in_slice_{i}.png"), act_in_slice, cmap='gray')
    plt.imsave(os.path.join(save_out_dir, f"act_out_slice_{i}.png"), act_out_slice, cmap='gray')
    
    image_in = cv2.imread(os.path.join(save_in_dir, f"act_in_slice_{i}.png"), cv2.IMREAD_GRAYSCALE)
    image_out = cv2.imread(os.path.join(save_out_dir, f"act_out_slice_{i}.png"), cv2.IMREAD_GRAYSCALE)
    resized_in_image = cv2.resize(image_in, (256, 256))
    resized_out_image = cv2.resize(image_out, (256, 256))
    resized_in_image_rgb = cv2.cvtColor(resized_in_image, cv2.COLOR_GRAY2RGB)
    resized_out_image_rgb = cv2.cvtColor(resized_out_image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(save_in_dir, f"act_in_slice_{i}.png"),resized_in_image_rgb)
    cv2.imwrite(os.path.join(save_out_dir, f"act_in_slice_{i}.png"),resized_out_image_rgb)


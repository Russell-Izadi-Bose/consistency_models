import os

import matplotlib.pyplot as plt
import numpy as np

# Replace 'your_file_path.npz' with the path to your .npz file
target_dir = "/home/consistency_models/samples"
npz_file_path = '/tmp/openai-2024-04-03-19-16-25-777911/samples_5x64x64x3.npz'

# Load data from the .npz file
data = np.load(npz_file_path)
if len(data) == 1:
    arr = next(iter(data.values()))
    label_arr = None
    print(arr.shape)
elif len(data) == 2:
    arr, label_arr = data.values()
    print(arr.shape, label_arr.shape)

# Number of images
N = arr.shape[0]

# Create a figure with subplots in a grid of N//5 x 5
fig, axs = plt.subplots(N//5, 5, figsize=(15, 3*(N//5)))

# Ensure axs is 2D
axs = axs.reshape(-1)

# Plot each image and its label
for i, ax in enumerate(axs):
    if i < N:
        ax.imshow(arr[i])
        ax.axis('off')
        # Display the label at the center of each image
        if label_arr is not None:
            ax.text(0.5, 0.5, str(label_arr[i]), color='white', fontsize=12,
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.axis('off')  # Turn off axis for empty subplots

# Adjust layout
plt.tight_layout()

# Save the figure
folder, name = npz_file_path.split(os.sep)[-2:]
name = name.split('.')[0]
png_file_path = os.path.join(
    target_dir, f"{folder}_{name}.png")
print(png_file_path)
plt.savefig(png_file_path)

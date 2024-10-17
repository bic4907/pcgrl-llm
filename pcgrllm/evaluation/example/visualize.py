import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_npy_files(folder_path: str):
    """
    Visualize all .npy files in a given folder path.
    """
    # List all files in the folder
    npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    # Determine the number of rows and columns based on the number of files
    num_files = len(npy_files)
    cols = 5  # Show 5 images per row
    rows = (num_files + cols - 1) // cols  # Calculate number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for idx, npy_file in enumerate(npy_files):
        npy_path = os.path.join(folder_path, npy_file)

        # Load the .npy file
        grid = np.load(npy_path)

        # Visualize the grid
        axes[idx].imshow(grid, cmap='gray', interpolation='nearest')
        axes[idx].set_title(os.path.basename(npy_file))
        axes[idx].axis('off')

    # Hide any empty subplots
    for ax in axes[num_files:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Paths to the folders with A and B variations
path_to_A = './A/iteration_1/inference/numpy/'
path_to_B = './B/iteration_1/inference/numpy/'
path_to_H = './H/iteration_1/inference/numpy/'

# Visualize A and B variations
print("Visualizing A variations:")
visualize_npy_files(path_to_A)

print("Visualizing B variations:")
visualize_npy_files(path_to_B)

print("Visualizing B variations:")
visualize_npy_files(path_to_H)
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

def find_unique_image_sizes(directory1, directory2):
    image_files = [f for f in os.listdir(directory1) if f.endswith('.mha')]
    label_files = [f for f in os.listdir(directory2) if f.endswith('.mha')]
    unique_sizes = set()  # Use a set to store unique sizes

    # Only plot a limited number of images for performance reasons
    plot_limit = 25
    images_plotted = 0

    for file, label_file in zip(image_files, label_files):
        file_path = os.path.join(directory1, file)
        image = sitk.ReadImage(file_path)
        label = sitk.ReadImage(os.path.join(directory2, label_file))
        image_array = sitk.GetArrayFromImage(image)
        label_array = sitk.GetArrayFromImage(label)
        size = image.GetSize()  # Get the size of the image
        unique_sizes.add(size)  # Add to the set of unique sizes

        if images_plotted < plot_limit:
            # Normalize image for display
            normalized_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
            plt.imshow(normalized_image[200,:,:], cmap='gray')  # Display the first slice for 3D images
            plt.title(f'File: {file} | Size: {size}')
            plt.show()

            plt.imshow(label_array[200,:,:]*50, cmap='gray')  # Display the first slice for 3D images
            plt.title(f'File: {file} | Size: {size}')
            plt.show()
            images_plotted += 1

    return unique_sizes

# Usage example:
directory1 = '/home/ubuntu/files/project_TransUNet/data/Penguin/train'
directory2 = '/home/ubuntu/files/project_TransUNet/data/Penguin/label'
unique_image_sizes = find_unique_image_sizes(directory1, directory2)
print("Unique image sizes in the dataset:")
for size in unique_image_sizes:
    print(size)


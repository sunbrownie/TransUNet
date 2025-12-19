import os
import h5py
import numpy as np

def load_data_from_h5(file_path):
    with h5py.File(file_path, 'r') as hf:
        image = hf['image'][:]  
        label = hf['label'][:]
    return image, label

val_dir = '/home/ubuntu/files/project_TransUNet/data/Penguin/test_224'
output_dir = '/home/ubuntu/files/project_TransUNet/data/Penguin/val_224'
output_file_list_path = '/home/ubuntu/files/project_TransUNet/data/Penguin/val_224/processed_files.txt'

# Open the file list for writing
with open(output_file_list_path, 'w') as train_list:
    for file in os.listdir(val_dir):
        image_array, label = load_data_from_h5(os.path.join(val_dir, file))
        for slice_index in range(image_array.shape[0]):
            slice_image = image_array[slice_index, :, :]
            slice_label = label[slice_index, :, :]

            # Debug print to verify slice shapes
            print(f"Processing {file}, slice {slice_index}: shape {slice_image.shape}")

            # Save slice image and label in an npz file
            output_file_name = os.path.splitext(file)[0] + f"_slice{slice_index:03d}.npz"
            np.savez_compressed(os.path.join(output_dir, output_file_name), image=slice_image, label=slice_label)
            
            # Write to the file list
            train_list.write(output_file_name + '\n')



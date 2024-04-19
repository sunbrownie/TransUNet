import os
import SimpleITK as sitk
import numpy as np
import shutil
import h5py

def resize_and_pad_image(image_array, target_size=(224, 224), pad_value=0):
    """
    Resize the image array to target size and pad if necessary.
    """
    image_sitk = sitk.GetImageFromArray(image_array)
    original_size = image_sitk.GetSize()
    original_spacing = image_sitk.GetSpacing()

    new_spacing = [
        original_spacing[0] * original_size[0] / target_size[0],
        original_spacing[1] * original_size[1] / target_size[1]
    ]

    # Resize the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(target_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resized_image = resampler.Execute(image_sitk)

    # Convert back to array
    resized_array = sitk.GetArrayFromImage(resized_image)

    # Padding if the resized image is smaller than the target size
    padding_needed = [max(0, target_size[i] - resized_array.shape[i]) for i in range(2)]
    if any(padding_needed):
        pad_width = [(0, padding_needed[i]) for i in range(2)]
        padded_array = np.pad(resized_array, pad_width, mode='constant', constant_values=pad_value)
    else:
        padded_array = resized_array

    return padded_array

def process_images(image_dir, label_dir, output_dir, test_dir):
    # Ensure the output directories exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Gather all MHA files in the image directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.mha')]
    np.random.shuffle(image_files)  # Randomly shuffle files
    
    # Split into train and test sets
    split_index = int(0.8 * len(image_files))
    train_files = image_files[:split_index]
    test_files = image_files[split_index:]
    
    # Create text files to store the image names
    train_list = open(os.path.join(output_dir, 'train.txt'), 'w')
    test_list = open(os.path.join(test_dir, 'test_vol.txt'), 'w')
    all_list = open(os.path.join(image_dir, 'all.lst'), 'w')
    
    # Process train files
    for file in train_files:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file)  
        
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        
        image_array = sitk.GetArrayFromImage(image)  # Convert to numpy array
        label_array = sitk.GetArrayFromImage(label)  # Convert label to numpy array
        
        # Save each slice and corresponding label slice as separate npz files
        for slice_index in range(image_array.shape[0]):
            slice_image = resize_and_pad_image(image_array[slice_index, :, :])
            slice_label = resize_and_pad_image(label_array[slice_index, :, :])
            
            print(slice_image.shape)
            # Save slice image and label in an npz file
            output_file_name = os.path.splitext(file)[0] + f"_slice{slice_index:03d}"
            np.savez_compressed(os.path.join(output_dir, output_file_name), image=slice_image, label=slice_label)
            train_list.write(output_file_name + '\n')

    # Process test files
    for file in test_files:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file)  # Assuming same file name for labels
        
        image = sitk.ReadImage(image_path)
        label = sitk.ReadImage(label_path)
        
        image_array = sitk.GetArrayFromImage(image)  # Convert to numpy array
        label_array = sitk.GetArrayFromImage(label)  # Convert label to numpy array
        
        # Save image and label in a .npy.h5 file
        output_file_name = os.path.splitext(file)[0] + '.npy.h5'
        with h5py.File(os.path.join(test_dir, output_file_name), 'w') as hf:
            hf.create_dataset('image', data=image_array, compression='gzip')
            hf.create_dataset('label', data=label_array, compression='gzip')
        test_list.write(output_file_name + '\n')


    # Populate the all.lst
    for file in image_files:
        all_list.write(file + '\n')
    
    # Close all files
    train_list.close()
    test_list.close()
    all_list.close()

# Usage example:
process_images('/home/ubuntu/files/project_TransUNet/data/Penguin/train',
               '/home/ubuntu/files/project_TransUNet/data/Penguin/label',
               '/home/ubuntu/files/project_TransUNet/data/Penguin/train_processed_224',
               '/home/ubuntu/files/project_TransUNet/data/Penguin/test_224')


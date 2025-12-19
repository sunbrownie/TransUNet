import os
import SimpleITK as sitk
import numpy as np
import shutil
import h5py

def calculate_class_imbalance(training_directory):
    label_counts = np.zeros(30)  # Assuming 30 classes
    train_files = os.listdir(training_directory)
    
    # Process each training file
    for file in train_files:
        full_path = os.path.join(training_directory, file)
        data = np.load(full_path)
        labels = data['label']  
        
        # Count occurrences of each label
        for i in range(30):  # Adjust the range if there are more or fewer classes
            label_counts[i] += np.count_nonzero(labels == i)
    
    # Normalize counts to get probabilities
    total_counts = np.sum(label_counts)
    if total_counts > 0:
        probabilities = label_counts / total_counts
    else:
        probabilities = np.zeros(30)
    
    # Calculate weights for weighted loss: Inverse of probabilities
    weights = 1 / (probabilities + 1e-6)  # Add a small constant to avoid division by zero
    normalized_weights = weights / np.sum(weights)  # Normalize weights to sum to 1
    
    return normalized_weights

weights = calculate_class_imbalance('/home/ubuntu/files/project_TransUNet/data/Penguin/train_processed_224')
print("Weights for loss function:", weights)



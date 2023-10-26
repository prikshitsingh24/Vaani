# Define the root directory containing your data
import os


data_root = "E:\\ml_start\\GAN_ISL\\Indian"

# Create a mapping between folder names and labels
class_labels = {}
for idx, folder_name in enumerate(os.listdir(data_root)):
    if os.path.isdir(os.path.join(data_root, folder_name)):
        class_labels[idx+1] = folder_name

# Create a list to store image paths and corresponding labels
image_paths = []
labels = []

print(f"{class_labels} ")
# Traverse the directories and assign labels
for label,folder_name in class_labels.items():
    folder_path = os.path.join(data_root, folder_name)
    for filename in os.listdir(folder_path):
        image_paths.append(os.path.join(folder_path, filename))
        labels.append(label)
        
        
# print(image_paths)
print(f"{class_labels} ")

# Now, 'image_paths' contains paths to the images, and 'labels' contains their corresponding labels.
# You can create a dataset and data loader for training, as shown in the previous response
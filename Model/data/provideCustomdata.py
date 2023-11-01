import os

from data.preprocessor import wordPreproccessor
# from preprocessor import wordPreproccessor


def getCustomdata():
    # Define the root directory containing your data
    
    data_root = "E:\\ml_start\\GAN_ISL\\Indian"

    # Create a mapping between folder names and labels
    class_labels = {}
    for idx, folder_name in enumerate(os.listdir(data_root)):
        if os.path.isdir(os.path.join(data_root, folder_name)):
            class_labels[int(wordPreproccessor(folder_name))] = folder_name

    # Create a list to store image paths and corresponding labels
    image_paths = []
    labels = []
    imageFolder=[]

    
    # Traverse the directories and assign labels
    for label,folder_name in class_labels.items():
        folder_path = os.path.join(data_root, folder_name)
        imageFolder.append(folder_path)
        for filename in os.listdir(folder_path):
            image_paths.append(os.path.join(folder_path, filename))
            labels.append(label)
            
            
    # print(imageFolder)
    # print(f"{labels} ")
    # print(f"{class_labels} ")
    # print(labels)
    return labels,imageFolder
# getCustomdata()   

# Now, 'image_paths' contains paths to the images, and 'labels' contains their corresponding labels.
# You can create a dataset and data loader for training, as shown in the previous response
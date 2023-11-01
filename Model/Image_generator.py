import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
import pandas as pd
from  Hyperparameter.config import config
from data.preprocessor import wordPreproccessor


device = "cuda" if torch.cuda.is_available() else print("running on cpu")
torch.cuda.set_device(device=0)
Z_DIM = config["Z_DIM"]
CHANNELS_IMG = config["CHANNELS_IMG"]
FEATURES_GEN = config["FEATURES_GEN"]
NUM_CLASSES = config["NUM_CLASSES"] # Number of classes for your custom dataset
IMG_SIZE = config["IMG_SIZE"]
GEN_EMBEDDING = config["GEN_EMBEDDING"]

LEARNING_RATE = config["LEARNING_RATE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
FEATURES_CRITIC = config["FEATURES_CRITIC"]
CRITIC_ITERATIONS = config["CRITIC_ITERATIONS"]
LAMBDA_GP = config["LAMBDA_GP"]
# Load the saved generator model
generator = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
generator.load_state_dict(torch.load("E:\\ml_start\\GAN_ISL\\Vaani\\vaani_models\\generator.pth"))
generator.eval()  # Set the generator in evaluation mode

userInput=wordPreproccessor(input("Enter alphabet or number between 0 - 9 : ").split(','))
# Specify the label you want to generate images for
desired_label = torch.tensor(userInput).to(device)  # Replace your_desired_label with the desired label

# Generate random noise vectors (you can change the number of vectors)
num_samples =len(desired_label) 
noise = torch.randn(num_samples, Z_DIM, 1, 1).to(device)
print("GENERATING IMAGE")
# Generate images with the desired label
with torch.no_grad():
    generated_images = generator(noise, desired_label)
   

# Visualize or save the generated images
# For example, you can use torchvision to save or display the images
import torchvision.utils as vutils

# Save the generated images to a directory
output_directory = "generated_images"
os.makedirs(output_directory, exist_ok=True)
image_filename = os.path.join(output_directory, "generated_images.png")
vutils.save_image(generated_images, image_filename, normalize=True)

print("Upscalling the image wait !")


import cv2
from cv2 import dnn_superres

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
image = cv2.imread('.\\generated_images\\generated_images.png')

# Read the desired model
path = "E:\\ml_start\\GAN_ISL\\archive\EDSR_x4.pb" #Give your specific path to SR model
sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("edsr", 4)

# # Upscale the image
# result = sr.upsample(image) #using SR method 
result = cv2.resize(image,dsize=None,fx=4,fy=4)

# Save the image
cv2.imwrite(".\\generated_images\\upscaleddnnEDSR.png", result)

# Display the generated images
from matplotlib import pyplot as plt
image_grid = vutils.make_grid(generated_images, normalize=True)
plt.figure(figsize=(15, 15))
plt.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
plt.axis('on')
plt.xlabel("GENRATED IMAGE")
plt.show()
# OpenCV upscaled
plt.figure(figsize=(15, 15))
plt.imshow(result[:,:,::-1])
plt.axis('on')
plt.xlabel("UPSCALED IMAGE")
plt.show()

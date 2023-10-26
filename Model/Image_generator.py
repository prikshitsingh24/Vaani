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

# Specify the label you want to generate images for
desired_label = torch.tensor([17,  6, 18, 30,  5, 33, 28, 13, 30,  9]).to(device)  # Replace your_desired_label with the desired label

# Generate random noise vectors (you can change the number of vectors)
num_samples = 10
noise = torch.randn(num_samples, Z_DIM, 1, 1).to(device)
print("iam here")
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

# Display the generated images
from matplotlib import pyplot as plt
image_grid = vutils.make_grid(generated_images, normalize=True)
plt.figure(figsize=(10, 10))
plt.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
plt.axis('off')
plt.show()

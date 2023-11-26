# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from utils import gradient_penalty, save_checkpoint, load_checkpoint
# from model import Discriminator, Generator, initialize_weights
# import pandas as pd

# from Hyperparameter.config import config
# from data.provideCustomdata import getCustomdata
# from data.customData import CustomDataset



# # Access hyperparameters
# # Hyperparameters etc.
# Z_DIM = config["Z_DIM"]
# CHANNELS_IMG = config["CHANNELS_IMG"]
# FEATURES_GEN = config["FEATURES_GEN"]
# NUM_CLASSES = config["NUM_CLASSES"] # Number of classes for your custom dataset
# IMG_SIZE = config["IMG_SIZE"]
# GEN_EMBEDDING = config["GEN_EMBEDDING"]

# device = "cuda" if torch.cuda.is_available() else print("running on cpu")
# torch.cuda.set_device(device=0)
# LEARNING_RATE = config["LEARNING_RATE"]
# BATCH_SIZE = config["BATCH_SIZE"]
# NUM_EPOCHS = config["NUM_EPOCHS"]
# FEATURES_CRITIC = config["FEATURES_CRITIC"]
# CRITIC_ITERATIONS = config["CRITIC_ITERATIONS"]
# LAMBDA_GP = config["LAMBDA_GP"]
# DATA_ROOT =config[" DATA_ROOT"]

# # Specify the directory to save the models
# save_dir = "./vaani_models"
# os.makedirs(save_dir, exist_ok=True) 

# # Define the transformations for your custom dataset
# transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
# ])
# data_root = "G:\\ALIML\\ganISL\\Indian"
# # labels,imageFolder=getCustomdata()



# # Create a custom dataset using ImageFolder with labels
# dataset = CustomDataset(data_root, transforms)

# # dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transforms)

# # print(dataset)
# # DataLoader should return (real images, labels) tuples
# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     pin_memory=True,
    
    
# )


# # for i in range (loader.dataset.__len__()):
# #     print(loader.dataset.__getitem__(i))

# # print(loader.dataset.__len__())

# # Initialize gen and critic
# gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
# critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)
# initialize_weights(gen)
# initialize_weights(critic)

# # Initialize optimizers
# opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
# opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# # For tensorboard plotting
# fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/GAN_Custom/real")
# writer_fake = SummaryWriter(f"logs/GAN_Custom/fake")
# step = 0

# gen.train()
# critic.train()

# for epoch in range(NUM_EPOCHS):
#     for batch_idx, (real, labels) in enumerate(tqdm(loader)):
        
#         real = real.to(device)
#         cur_batch_size = real.shape[0]
#         labels = labels.to(device)
        
        
#         # print(f"real {real}")
#         # print(f"batch_idx {batch_idx}")
#         # print(f"labels {labels}")

#         # Train Critic: max E[critic(real)] - E[critic(fake)]
#         # Equivalent to minimizing the negative of that
#         for _ in range(CRITIC_ITERATIONS):
#             noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
#             fake = gen(noise, labels)
#             critic_real = critic(real, labels).reshape(-1)
#             critic_fake = critic(fake, labels).reshape(-1)
#             gp = gradient_penalty(critic, real, labels, fake, device=device)
#             loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
#             critic.zero_grad()
#             loss_critic.backward(retain_graph=True)
#             opt_critic.step()

#         # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
#         gen_fake = critic(fake, labels).reshape(-1)
#         loss_gen = -torch.mean(gen_fake)
#         gen.zero_grad()
#         loss_gen.backward()
#         opt_gen.step()
#         print(batch_idx)
#         # Print losses occasionally and print to tensorboard
#         # if batch_idx % 9 == 0 and batch_idx > 0:
#         if batch_idx==1:
#             print(
#                 f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
#                   Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
#             )

#             with torch.no_grad():
#                 fake = gen(noise, labels)
#                 # Take out (up to) 32 examples
#                 img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
#                 img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

#                 writer_real.add_image("Real", img_grid_real, global_step=step)
#                 writer_fake.add_image("Fake", img_grid_fake, global_step=step)

#             step += 1
            
# # Save the generator and discriminator
# torch.save(gen.state_dict(), os.path.join(save_dir, "generator.pth"))
# torch.save(critic.state_dict(), os.path.join(save_dir, "discriminator.pth"))

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
import pandas as pd

from Hyperparameter.config import config
from data.provideCustomdata import getCustomdata
from data.customData import CustomDataset



# Access hyperparameters
# Hyperparameters etc.
Z_DIM = config["Z_DIM"]
CHANNELS_IMG = config["CHANNELS_IMG"]
FEATURES_GEN = config["FEATURES_GEN"]
NUM_CLASSES = config["NUM_CLASSES"] # Number of classes for your custom dataset
IMG_SIZE = config["IMG_SIZE"]
GEN_EMBEDDING = config["GEN_EMBEDDING"]

device = "cuda" if torch.cuda.is_available() else print("running on cpu")
torch.cuda.set_device(device=0)
LEARNING_RATE = config["LEARNING_RATE"]
BATCH_SIZE = config["BATCH_SIZE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
FEATURES_CRITIC = config["FEATURES_CRITIC"]
CRITIC_ITERATIONS = config["CRITIC_ITERATIONS"]
LAMBDA_GP = config["LAMBDA_GP"]
DATA_ROOT =config[" DATA_ROOT"]

# Specify the directory to save the models
save_dir = "./vaani_models"
os.makedirs(save_dir, exist_ok=True) 

# Define the transformations for your custom dataset
transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
data_root = "G:\\ALIML\\ganISL\\Indian"
# labels,imageFolder=getCustomdata()



# Create a custom dataset using ImageFolder with labels
dataset = CustomDataset(data_root, transforms)

# dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transforms)

# print(dataset)
# DataLoader should return (real images, labels) tuples
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    
    
)


# for i in range (loader.dataset.__len__()):
#     print(loader.dataset.__getitem__(i))

# print(loader.dataset.__len__())

# Initialize gen and critic
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMG_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMG_SIZE).to(device)
initialize_weights(gen)
initialize_weights(critic)

# Initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# For tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_Custom/real")
writer_fake = SummaryWriter(f"logs/GAN_Custom/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, labels) in enumerate(tqdm(loader)):
        
        real = real.to(device)
        cur_batch_size = real.shape[0]
        labels = labels.to(device)
        
        
        # print(f"real {real}")
        # print(f"batch_idx {batch_idx}")
        # print(f"labels {labels}")

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # Equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise, labels)
            critic_real = critic(real, labels).reshape(-1)
            critic_fake = critic(fake, labels).reshape(-1)
            gp = gradient_penalty(critic, real, labels, fake, device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        print(batch_idx)
        # Print losses occasionally and print to tensorboard
        # if batch_idx % 9 == 0 and batch_idx > 0:
        if batch_idx==1:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise, labels)
                # Take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
            
# Save the generator and discriminator
torch.save(gen.state_dict(), os.path.join(save_dir, "generator.pth"))
torch.save(critic.state_dict(), os.path.join(save_dir, "discriminator.pth"))

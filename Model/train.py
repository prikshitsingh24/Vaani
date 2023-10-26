# """
# Training of WGAN-GP

# Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
# * 2020-11-01: Initial coding
# * 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm
# from utils import gradient_penalty, save_checkpoint, load_checkpoint
# from model import Discriminator, Generator, initialize_weights
# import pandas as pd

# # Hyperparameters etc.
# device = "cuda" if torch.cuda.is_available() else "cpu"
# LEARNING_RATE = 1e-4
# BATCH_SIZE = 64
# IMG_SIZE = 64
# CHANNELS_IMG = 1
# NUM_CLASSES=10
# GEN_EMBEDDING=100
# Z_DIM = 100
# NUM_EPOCHS = 100
# FEATURES_CRITIC = 16
# FEATURES_GEN = 16
# CRITIC_ITERATIONS = 5
# LAMBDA_GP = 10

# transforms = transforms.Compose(
#     [
#         transforms.Resize(IMG_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
#         ),
#     ]
# )


# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# # comment mnist above and uncomment below for training on CelebA
# # dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# # initialize gen and disc, note: discriminator should be called critic,
# # according to WGAN paper (since it no longer outputs between [0, 1])
# gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN,NUM_CLASSES,IMG_SIZE,GEN_EMBEDDING).to(device)
# critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC,NUM_CLASSES,IMG_SIZE).to(device)
# initialize_weights(gen)
# initialize_weights(critic)

# # initializate optimizer
# opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
# opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# # for tensorboard plotting
# fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
# writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
# step = 0

# gen.train()
# critic.train()

# for epoch in range(NUM_EPOCHS):
#     # Target labels not needed! <3 unsupervised but we are making CGAN so labels have to be provided
#     for batch_idx, (real, labels) in enumerate(tqdm(loader)):
#         real = real.to(device)
#         cur_batch_size = real.shape[0]
#         labels=labels.to(device)

#         # Train Critic: max E[critic(real)] - E[critic(fake)]
#         # equivalent to minimizing the negative of that
#         for _ in range(CRITIC_ITERATIONS):
#             noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
#             fake = gen(noise,labels)
#             critic_real = critic(real,labels).reshape(-1)
#             critic_fake = critic(fake,labels).reshape(-1)
#             gp = gradient_penalty(critic, real, labels, fake, device=device)
#             loss_critic = (
#                 -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
#             )
#             critic.zero_grad()
#             loss_critic.backward(retain_graph=True)
#             opt_critic.step()

#         # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
#         gen_fake = critic(fake,labels).reshape(-1)
#         loss_gen = -torch.mean(gen_fake)
#         gen.zero_grad()
#         loss_gen.backward()
#         opt_gen.step()

#         # Print losses occasionally and print to tensorboard
#         if batch_idx % 100 == 0 and batch_idx > 0:
#             print(
#                 f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
#                   Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
#             )

#             with torch.no_grad():
#                 fake = gen(noise,labels)
#                 # take out (up to) 32 examples
#                 img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
#                 img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

#                 writer_real.add_image("Real", img_grid_real, global_step=step)
#                 writer_fake.add_image("Fake", img_grid_fake, global_step=step)

#             step += 1
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

# Specify the directory to save the models
save_dir = "./vaani_models"
os.makedirs(save_dir, exist_ok=True) 

# Define the transformations for your custom dataset
transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
data_root = "E:\\ml_start\\GAN_ISL\\Indian"
# Create a custom dataset using ImageFolder with labels
dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transforms)
# DataLoader should return (real images, labels) tuples
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
)

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

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0 and batch_idx > 0:
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

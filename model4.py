
#!/usr/bin/env python
# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/e9c8374ecc202120dc94db26bf08a00f/dcgan_faces_tutorial.ipynb
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
import os
import pathlib
import random
import argparse
from tokenize import String
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # ConvTranspose2d
            # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
            #                          groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * \
                               4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * \
                               2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(self.ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.main = nn.Sequential(
            # Conv2D
            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
            #                 dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

            # input is 3 x 64 x 64
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Set random seed for reproducibility
def set_random_seed():
    #manualSeed = 999
    manualSeed = random.randint(1, 10000)  # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


def load_data(dataroot: String, image_size: int, batch_size: int, workers: int):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    return dataloader


def plot_loss(path, G_losses, D_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path, dpi=300)

def plot_score(path, D_G_z1s, D_G_z2s, D_xs):
    plt.figure(figsize=(10, 5))
    plt.title("D(G1(x)),D(G2(x)),D(x) During Training")
    plt.plot(D_G_z1s, label="D_G_z1")
    plt.plot(D_G_z2s, label="D_G_z2")
    plt.plot(D_xs, label="D_xs")
    plt.xlabel("iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(path, dpi=300)

# Visualize G's progression as gif


def generate_gif(path, img_list):
    fig,ax = plt.subplots(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(img_list[i], (1, 2, 0)), animated=True), ax.annotate(f"Iter: {i*500}/{len(img_list)*500}", (0.5, 1.03), fontsize=15 ,xycoords="axes fraction", ha="center")]
           for i in range(len(img_list))]
    ani = animation.ArtistAnimation(
        fig, ims, interval=500, repeat_delay=1000, blit=True)

    ani.save(path)


def compare_real_and_fake(path, real_batch, img_list, device):
    # Plotting real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
        :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(path, dpi=300)


def generate_directory(cat_id):
    images_dir = f'results/{cat_id}/images'
    model_dir = f'results/{cat_id}/model'
    fig_dir = f'results/{cat_id}/fig'

    for path in [images_dir, model_dir, fig_dir]:
        if not os.path.exists(path):    
            # Create a new directory because it does not exist 
            os.makedirs(path)
            print(f"The new directory {path} is created!")


if __name__ == "__main__":
    # ----------
    # Initial Settings
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--cat_id", required=True, help="category")
    parser.add_argument("--dataroot",
                        default="data/data_resized/News&Politics", help="Root directory for dataset")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Spatial size of training images. All images will be resized to this size using a transformer.")
    parser.add_argument("--nz", type=int, default=100,
                        help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Size of feature maps in discriminator")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate for optimizers")
    parser.add_argument("--ngpu", type=int, default=1,
                        help="Number of GPUs available. Use 0 for CPU mode.")
    opt = parser.parse_args()

    set_random_seed()

    # Decide which device we want to run on
    device = torch.device("cuda" if (
        torch.cuda.is_available() and opt.ngpu > 0) else "cpu")

    print(vars(opt))
    print(device)

    # ----------
    # Data Loading
    # ----------
    dataloader = load_data(opt.dataroot, opt.image_size,
                           opt.batch_size, opt.workers)

    # ----------
    # Generator Construction
    # ----------
    # Create the generator
    netG = Generator(opt.ngpu, opt.nz, opt.ngf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(opt.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # ----------
    # Discriminator Construction
    # ----------
    # Create the Discriminator
    netD = Discriminator(opt.ngpu, opt.ndf).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (opt.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(opt.ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Model Summary
    print(netG)
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 0.8
    fake_label = 0.2

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Commented out IPython magic to ensure Python compatibility.
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_xs = []
    D_G_z1s = []
    D_G_z2s = []

    iters = 0

    # ----------
    # Training
    # ----------
    generate_directory(opt.cat_id)

    print(f"Starting Training with data in {opt.dataroot}")
    for epoch in range(opt.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full(
                (b_size,), real_label, dtype=torch.float, device=device
            )
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, opt.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_G_z1s.append(D_G_z1)
            D_G_z2s.append(D_G_z2)
            D_xs.append(D_x)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == opt.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

            iters += 1

        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, opt.num_epochs,
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if epoch % 5 == 0:
            vutils.save_image(fake, os.path.join(
                f'results/{opt.cat_id}/images', f'yt_{epoch}.png'))
    # save last image
    vutils.save_image(fake, os.path.join(
        f'results/{opt.cat_id}/images', f'yt_{epoch}.png'))
    img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))
    
    torch.save(netD.state_dict(), f'results/{opt.cat_id}/model/discriminator.pt')
    torch.save(netG.state_dict(), f'results/{opt.cat_id}/model/generator.pt')

    # Result Visualization
    plot_loss(f'results/{opt.cat_id}/fig/G&D_Loss.png', G_losses, D_losses)
    plot_score(f'results/{opt.cat_id}/fig/G&D_score.png', D_G_z1s, D_G_z2s, D_xs)
    generate_gif(f'results/{opt.cat_id}/fig/animation.gif', img_list)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    compare_real_and_fake(f'results/{opt.cat_id}/fig/fake_real.png',
                          real_batch, img_list, device)

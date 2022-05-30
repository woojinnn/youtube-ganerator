
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
from model4 import Generator, load_data
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3,
                                             shuffle=True, num_workers=workers)

    return dataloader


def compare_real_and_fake(path, real_batch, img, device):
    # Plotting real images
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
        :9], padding=5, normalize=True,nrow=3).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))
    plt.savefig(path, dpi=300)





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

    # ----------
    # Data Loading
    # ----------
    dataloader = load_data(opt.dataroot, opt.image_size,
                           opt.batch_size, opt.workers)

    # ----------
    # Generator Construction
    # ----------
    # Create the generator


    model = Generator(1, 100, 64)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(
        f"results/{opt.cat_id}/model/generator.pt"))
    model.to(device)

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.




    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_xs = []
    D_G_z1s = []
    D_G_z2s = []

    iters = 0

    noise = torch.randn(9, opt.nz, 1, 1, device=device)
    fake = model(noise)
    img = (vutils.make_grid(
                    fake, padding=2, normalize=True,nrow=3))

    # Result Visualization


    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))
    compare_real_and_fake(f'results/{opt.cat_id}/fig/fake_real3.png',
                          real_batch, img, device)

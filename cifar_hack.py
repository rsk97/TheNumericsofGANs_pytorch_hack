
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch import autograd

import copy
import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt
import sys, os

from torchvision.utils import save_image

from dataset.loaders import CIFAR
from models.dcgan import Generator, Discriminator
from utils.utils import net_losses, batch_net_outputs, batch_net_outputs_cifar, save_models, load_models
from utils.plot import plot_eigens_cifar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


z_dim = 256
gf_dim = 64
df_dim = 64
c_dim = 3
outdim = 32 
gamma = 1e-1
lr = 1e-4
steps = 150000
batch_size = 32 
method = 'ConsOpt'
eigen_path = "./eigens"
img_path = "./Images2"
model_path = "./Models2"
optim_name = "RMSProp"

lr_adam = 3e-4
beta = 0.55
alpha = 0.6


if __name__ == "__main__":
    
    train_loader, test_loader, classes = CIFAR(batch_size)    

    gen_net = Generator(z_dim, c_dim, gf_dim, outdim).to(device)
    disc_net = Discriminator(c_dim, gf_dim, outdim).to(device)

    params = list(gen_net.parameters()) + list(disc_net.parameters())

    if optim_name == "Adam":
        gen_opt = optim.Adam(gen_net.parameters(), lr=lr_adam, betas=(0.5, 0.9))
        disc_opt = optim.Adam(disc_net.parameters(), lr=lr_adam, betas=(beta, 0.9))
    else:
        gen_opt = optim.RMSprop(gen_net.parameters(), lr=lr)
        disc_opt = optim.RMSprop(disc_net.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    for i in range(steps+1):

        real_in, _ = next(iter(train_loader))
        real_in = real_in.to(device)

        gen_out, _, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out = batch_net_outputs_cifar(real_in, gen_net, disc_net,  batch_size, z_dim, device)
        gen_loss_detached, disc_loss_detached, gen_loss, disc_loss = net_losses(criterion, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out)

        if i%20 == 0:
            print(i)

        if i % 200 == 0:
            save_models(gen_net, disc_net, i, model_path)
            gen_out = 0.5 * gen_out + 0.5
            save_image(gen_out[:batch_size], f"{img_path}/{i}_{optim_name}_.png")
            p_count = torch.cat([x.flatten() for x in params]).shape[0]
            # plot_eigens_cifar(i, gen_net, disc_net, p_count, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out, gen_loss, disc_loss, method, optim_name)

        if method == 'ConsOpt':
            gen_net.zero_grad()
            gen_grad = autograd.grad(gen_loss, gen_net.parameters(), retain_graph=True, create_graph=True)
            disc_net.zero_grad()
            disc_grad = autograd.grad(disc_loss, disc_net.parameters(), retain_graph=True, create_graph=True)

            v = list(gen_grad) + list(disc_grad)
            # v = autograd.grad(total_loss, params, retain_graph=True, create_graph=True)
            v = torch.cat([t.flatten() for t in v])

            L = 1/2 * torch.dot(v, v)
            jgrads = autograd.grad(L, params, retain_graph=True)
        
            gen_opt.zero_grad()


            for i in range(len(params)):
               params[i].grad = jgrads[i] * gamma
            gen_loss_detached.backward(retain_graph=True, create_graph=True)
            gen_opt.step()

            disc_opt.zero_grad()

            for i in range(len(params)):
               params[i].grad = jgrads[i] * gamma
            disc_loss_detached.backward(retain_graph=True, create_graph=True)
            disc_opt.step()

        else:
            gen_opt.zero_grad()
            gen_loss_detached.backward(retain_graph=True, create_graph=True)
            gen_opt.step()

            disc_opt.zero_grad()
            disc_loss_detached.backward(retain_graph=True, create_graph=True)
            disc_opt.step()

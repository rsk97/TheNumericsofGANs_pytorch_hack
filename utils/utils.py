import torch
import copy
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt



def batch_net_outputs(gen, disc, batch_size, z_dim, sigma, device):
    z = torch.normal(mean=0, std=1, size=[batch_size, z_dim]).to(device)
    gen_out = gen(z)
    fake_d_out_gen = copy.deepcopy(disc)(gen_out)
    fake_d_out_disc = disc(gen_out.detach())
    fake_d_out = disc(gen_out)

    angles = torch.tensor([2*np.pi*k/8 for k in range(batch_size)]).to(device)
    mus = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    real_in = mus + torch.normal(mean=0, std=sigma, size=[batch_size, 2]).to(device)
    real_d_out = disc(real_in)

    return gen_out, real_in, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out


def net_losses(criterion, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out):
    disc_loss_detached = criterion(input=fake_d_out_disc, target=torch.zeros_like(fake_d_out_disc))
    disc_loss_detached += criterion(real_d_out, target=torch.ones_like(real_d_out))

    gen_loss_detached =  criterion(fake_d_out_gen, target=torch.ones_like(fake_d_out_gen))

    disc_loss = criterion(input=fake_d_out, target=torch.zeros_like(fake_d_out))
    disc_loss += criterion(real_d_out, target=torch.ones_like(real_d_out))

    gen_loss =  criterion(fake_d_out, target=torch.ones_like(fake_d_out))
    
    return gen_loss_detached, disc_loss_detached, gen_loss, disc_loss

def complex_scatter_plot(points, bbox=None, save_file="", xlabel="real part", ylabel="imaginary part", cmap='Blues'):
    fig, ax = plt.subplots()

    if bbox is not None:
        ax.axis(bbox)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx = [p.real for p in points]
    yy = [p.imag for p in points]
    
    plt.plot(xx, yy, 'X')
    plt.grid()

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def kde(mu, tau, bbox=[-1.6, 1.6, -1.6, 1.6], save_file="", xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off
    
    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    _ = ax.contourf(xx, yy, f, cmap=cmap)

    if save_file != "":
        plt.savefig(save_file, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
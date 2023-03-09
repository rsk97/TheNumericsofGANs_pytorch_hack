import torch
import numpy as np
from os.path import join
from torch import autograd
from utils.utils import batch_net_outputs, net_losses, complex_scatter_plot, kde



def plot_eigens(iteration, gen, disc, params, gamma, path,  batch_size, z_dim, sigma, criterion, device):
    _, _, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out = batch_net_outputs(gen, disc, batch_size, z_dim, sigma, device)
    _, _, gen_loss, disc_loss = net_losses(criterion, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out)
    p_count = torch.cat([x.flatten() for x in params]).shape[0]

    gen.zero_grad()
    gen_grad = autograd.grad(gen_loss, gen.parameters(), retain_graph=True, create_graph=True)
    disc.zero_grad()
    disc_grad = autograd.grad(disc_loss, disc.parameters(), retain_graph=True, create_graph=True)

    v = list(gen_grad) + list(disc_grad)
    v = torch.cat([t.flatten() for t in v])
    jacobian = torch.zeros([p_count * p_count]).to(device)

    for i in range(p_count):
        jacobian[i*p_count: (i+1)*p_count] = torch.cat([x.flatten() for x in autograd.grad(v[i], params, retain_graph=True)])

    jacobian = jacobian.reshape([p_count, -1])

    jacobian2 = jacobian - gamma * torch.mm(jacobian.T, jacobian)

    eigens = torch.linalg.eigvals(jacobian)
    eigens2 = torch.linalg.eigvals(jacobian2)


    save_path = join(path, 'Eig_v_' + str(iteration) + '.png')
    cmap='Blues'
    complex_scatter_plot(eigens.cpu().detach().numpy(), bbox=[-1.0, 1.0, -0.15, 0.15], save_file=save_path, cmap=cmap)

    save_path = join(path, 'Eig_w_' + str(iteration) + '.png')
    complex_scatter_plot(eigens2.cpu().detach().numpy(), bbox=[-1.0, 1.0, -0.15, 0.15], save_file=save_path, cmap=cmap)


def plot_kde(iteration, method, sigma, gen, path, device, batch_size, z_dim, real_input=False):
    z = torch.normal(mean=0, std=1, size=[batch_size * 5, z_dim]).to(device)
    inp = gen(z)

    save_path = join(path, 'KDE', method + "_" + str(iteration) + '.png')
    cmap='Blues'
    
    if real_input:
        angles = torch.tensor([2*np.pi*k/8 for k in range(batch_size * 5)]).to(device)
        mus = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        inp = mus + torch.normal(mean=0, std=sigma, size=[batch_size * 5, 2]).to(device)

        save_path = join(path, 'KDE', 'original.png')
        cmap='Reds'

    kde(inp[:, 0].cpu().detach().numpy(), inp[:, 1].cpu().detach().numpy(), save_file=save_path, cmap=cmap)

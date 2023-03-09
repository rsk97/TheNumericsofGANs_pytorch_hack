from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from models.gan import Gen, Dis

from utils.plot import plot_eigens, plot_kde
from utils.utils import batch_net_outputs, net_losses


#Parameters
z_dim = 16
gamma = 10.0
lr = 1e-4
sigma = 1e-2
steps = 20000
batch_size = 512
method = 'ConsOpt' #'SimGA' #'ConsOpt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = '/home/mila/r/rohan.sukumaran/repos/TheNumericsofGANs_pytorch/results/'


if __name__ == "__main__":
    gen_net = Gen(16, 2).to(device)
    disc_net = Dis(2, 1).to(device)

    params = list(gen_net.parameters()) + list(disc_net.parameters())

    gen_opt = optim.RMSprop(gen_net.parameters(), lr=lr)
    disc_opt = optim.RMSprop(disc_net.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()


    for i in range(steps+1):

        gen_out, real_in, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out = batch_net_outputs(gen_net, disc_net, batch_size, z_dim, sigma, device)
        gen_loss_detached, disc_loss_detached, gen_loss, disc_loss = net_losses(criterion, fake_d_out_gen, fake_d_out_disc, fake_d_out, real_d_out)

        if i%5000 == 0:
            if method == 'ConsOpt':
                plot_eigens(i, gen_net, disc_net, params, gamma, path, device)
            plot_kde(i,  method, sigma, gen_net, path, device, batch_size, z_dim, real_input=False)
        
            gen_path = join(path, 'Models', 'gen_' + method + "_" + str(i) + '.pt')
            disc_path = join(path, 'Models', 'disc_'+ method + "_" + str(i) + '.pt')
            torch.save(gen_net.state_dict(), gen_path)
            torch.save(disc_net.state_dict(), disc_path)      

        if method == 'ConsOpt':

            gen_net.zero_grad()
            gen_grad = autograd.grad(gen_loss, gen_net.parameters(), retain_graph=True, create_graph=True)
            disc_net.zero_grad()
            disc_grad = autograd.grad(disc_loss, disc_net.parameters(), retain_graph=True, create_graph=True)

            v = list(gen_grad) + list(disc_grad)
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


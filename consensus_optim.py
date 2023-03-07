import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
import torch.autograd as autograd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from torch.nn.utils import parameters_to_vector
from utils.optim import parameters_grad_to_vector
from dataset.loaders import get_8gaussians
from models.gan import Gen, Dis, weights_init
import os

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

update_rule = 'consensus'
dis_iter = 1
_batch_size = 256
dim = 2000
use_cuda = True
z_dim = 64

iterations = 1000
lr = 3e-4
beta = 0.55
alpha = 0.6



def get_dens_real(batch_size):
    data = get_8gaussians(batch_size).__next__()
    real = np.array(data.data.cpu())
    kde_real = gaussian_kde(real.T, bw_method=0.22)
    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_real = kde_real((x.ravel(), y.ravel())).reshape(*x.shape)
    return z_real

z_real = get_dens_real(1000)


def plot(fake, epoch, name):
    plt.figure(figsize=(20, 9))
    fake = np.array(fake.data.cpu())
    kde_fake = gaussian_kde(fake.T, bw_method=0.22)

    x, y = np.mgrid[-2:2:(200 * 1j), -2:2:(200 * 1j)]
    z_fake = kde_fake((x.ravel(), y.ravel())).reshape(*x.shape)

    ax1 = plt.subplot(1, 2, 1)
    ax1.pcolor(x, y, z_real, cmap='GnBu')

    ax2 = plt.subplot(1, 2, 2)
    ax2.pcolor(x, y, z_fake, cmap='GnBu')
    ax1.scatter(real.data.cpu().numpy()[:, 0],
                real.data.cpu().numpy()[:, 1])
    ax2.scatter(fake[:, 0], fake[:, 1])
    # plt.show()
    if not os.path.exists('8_G_res/_' + name):
        os.makedirs('8_G_res/_' + name)
    plt.savefig('8_G_res/_' + name + '/' + str(epoch) + '.png')
    plt.close()

dis = Dis()
gen = Gen()

dis.apply(weights_init)
gen.apply(weights_init)

if use_cuda:
    dis = dis.cuda()
    gen = gen.cuda()

if update_rule == 'adam':
    dis_optimizer = Adam(dis.parameters(),
                         lr=lr,
                         betas=(beta, 0.9))
    gen_optimizer = Adam(gen.parameters(),
                         lr=lr,
                         betas=(0.5, 0.9))
elif update_rule == 'sgd':
    dis_optimizer = SGD(dis.parameters(), lr=0.01)
    gen_optimizer = SGD(gen.parameters(), lr=0.01)

elif update_rule == 'consensus':
    dis_optimizer = Adam(dis.parameters(), lr=lr, betas=(beta, 0.9))
    gen_optimizer = Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

dataset = get_8gaussians(_batch_size)
criterion = nn.BCEWithLogitsLoss()

ones = Variable(torch.ones(_batch_size))
zeros = Variable(torch.zeros(_batch_size))
if use_cuda:
    criterion = criterion.cuda()
    ones = ones.cuda()
    zeros = zeros.cuda()

points = []
dis_params_flatten = parameters_to_vector(dis.parameters())
gen_params_flatten = parameters_to_vector(gen.parameters())

# just to fill the empty grad buffers
noise = torch.randn(_batch_size, z_dim)
if use_cuda:
    noise = noise.cuda()
noise = autograd.Variable(noise)
fake = gen(noise)
pred_fake = criterion(dis(fake), zeros).sum()
(0.0 * pred_fake).backward(create_graph=True)
gen_loss = 0
pred_tot = 0
elapsed_time_list = []

for iteration in range(iterations):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    noise = torch.randn(_batch_size, z_dim)
    if use_cuda:
        noise = noise.cuda()

    noise = autograd.Variable(noise)
    real = dataset.__next__()
    loss_real = criterion(dis(real), ones)
    fake = gen(noise)
    loss_fake = criterion(dis(fake), zeros)

    gradient_penalty = 0

    loss_d = loss_real + loss_fake + gradient_penalty

    grad_d = torch.autograd.grad(
        loss_d, inputs=(dis.parameters()), create_graph=True)
    for p, g in zip(dis.parameters(), grad_d):
        p.grad = g

    if update_rule == 'consensus':
        grad_gen_params_flatten = parameters_grad_to_vector(gen.parameters())
        grad_dis_params_flatten = parameters_grad_to_vector(dis.parameters())
        ham = grad_gen_params_flatten.norm(2) + grad_dis_params_flatten.norm(2)

        co_dis = torch.autograd.grad(
            ham, dis.parameters(), create_graph=True)
        dis_optimizer.step()
    else:
        dis_optimizer.step()

    noise = torch.randn(_batch_size, z_dim)
    ones = Variable(torch.ones(_batch_size))
    zeros = Variable(torch.zeros(_batch_size))
    if use_cuda:
        noise = noise.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    noise = autograd.Variable(noise)
    fake = gen(noise)
    loss_g = criterion(dis(fake), ones)
    grad_g = torch.autograd.grad(
        loss_g, inputs=(gen.parameters()), create_graph=True)
    for p, g in zip(gen.parameters(), grad_g):
        p.grad = g

    if update_rule == 'consensus':
        grad_gen_params_flatten = parameters_grad_to_vector(gen.parameters())
        ham = grad_gen_params_flatten.norm(2)

        co_gen = torch.autograd.grad(
            ham, gen.parameters(), create_graph=True)
    else:
        gen_optimizer.step()

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    if iteration > 3:
        elapsed_time_list.append(elapsed_time_ms)
    print(elapsed_time_ms)

    print("iteration: " + str(iteration))

avg_time = np.mean(elapsed_time_list)

print('avg_time: ' + str(avg_time))
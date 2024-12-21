# This code uses the well-trained model from free_energy_2nn.py to perform autodiff and FDM, and calculate the distance to ground_f
import sys

sys.path.append('../')
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import scipy.io
import optax
import time
from jax.nn import gelu, silu, tanh
from jax.lax import scan
from jax import random, jit, vmap, grad
import os
import scipy
import matplotlib.pyplot as plt
import argparse
from utils import normalization
from networks import get_network
import jax.numpy as jnp
import equinox as eqx

parser = argparse.ArgumentParser(description="2nn_derivatives")
parser.add_argument("--dt", type=int, default=0.01, help='t increment')
parser.add_argument("--dx", type=int, default=0.005, help='x increment')
parser.add_argument("--dy", type=int, default=0.005, help='y increment')
parser.add_argument("--gamma", type=int, default=1.0, help='sharpness parameter for Allen-Cahn equation')
parser.add_argument("--eps", type=int, default=0.01, help='parameter for sharpness in Allen-Cahn equation')
parser.add_argument("--ntest", type=int, default=50000, help = 'number of test points')

parser.add_argument("--interval1", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--interval2", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--ntrain", type=int, default=50000, help="the number of training dataset for each epochs")
parser.add_argument("--ite", type=int, default=20, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=5000, help="the number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--kanshape", type=str, default="16", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=100, help='degree of polynomials')
parser.add_argument("--features", type=int, default=100, help='width of the network')
parser.add_argument("--layers", type=int, default=10, help='depth of the network')
parser.add_argument("--len_h", type=int, default=2, help='lenth of k for sinckan')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--init_h", type=float, default=2.0, help='initial value of h')
parser.add_argument("--decay", type=str, default='inverse', help='exponent for h')
parser.add_argument("--skip", type=int, default=0, help='1: use skip connection for sinckan')
parser.add_argument("--activation", type=str, default='tanh', help="activation function for the networks")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--network", type=str, default="modifiedmlp", help="type of network")
parser.add_argument("--device", type=int, default=6, help="cuda number")

args = parser.parse_args()

def net(model, x, frozen_para):
    return model(jnp.stack(x), frozen_para)

# Load data
mat_data = scipy.io.loadmat('/home/qjw/code/python_code/phase_field/FullData_ACequ_2D.mat')
phiN = mat_data['phiN'].flatten(order = 'F')
phiN = phiN[:,None]
print(f'phiN shape: {phiN.shape}')
tN = mat_data['tN'].flatten()
t = np.repeat(tN, 256*256)
t = t.reshape(-1, 1)
print(f't = {t}')
xN = mat_data['xyN'][0, :].flatten()
yN = mat_data['xyN'][1, :].flatten()
xN = xN.reshape(-1,1)
yN = yN.reshape(-1,1)
xN = np.repeat(xN, 256, axis = 0)
xN_copies = np.tile(xN, (1,299))
x = xN_copies.flatten(order = 'F').reshape(-1,1)
print(f'x = {x}')
yN_copies = np.tile(yN, (1, 256))
y = yN_copies.flatten(order = 'F').reshape(-1,1)
y_copies = np.tile(y, (1, 299))
y = y_copies.flatten(order = 'F').reshape(-1,1)
print(f'y = {y}')
ob_txy = np.concatenate([t, x, y, phiN], -1)

# Load model
seed = args.seed
key = random.PRNGKey(seed)
keys = random.split(key, 2)
model_path = "modifiedmlp_0.eqx"
normalizer = normalization(ob_txy, args.normalization)
model_template = get_network(args, 3, 1, normalizer, keys)
model = eqx.tree_deserialise_leaves(model_path, model_template)

dt_phi = grad(model, 0)
dxx_phi = grad(grad(model, 1), 1)
dyy_phi = grad(grad(model, 2), 2)
frozen_para = model.get_frozen_para()

# Compare approximation accuracy of derivatives by autodiff and FDM
keys = random.split(keys[-1], 3)
N_test = args.ntest
eps = args.eps
gamma = args.gamma
test_set = random.choice(keys[0], ob_txy, shape=(N_test,), replace=False)
# Autodiff
dt_phi_val = vmap(net, (None, 0, None))(dt_phi, test_set[:, :3], frozen_para)
dxx_phi_val = vmap(net, (None, 0, None))(dxx_phi, test_set[:, :3], frozen_para)
dyy_phi_val = vmap(net, (None, 0, None))(dyy_phi, test_set[:, :3], frozen_para)
print(f'length of dt_phi_val is {len(dt_phi_val)}')
print(f'length of dxx_phi_val is {len(dxx_phi_val)}')
print(f'length of dyy_phi_val is {len(dyy_phi_val)}')
autodiff_f = (eps ** 2) * (dxx_phi_val + dyy_phi_val - (1/gamma) * dt_phi_val)
print(f'length of autodiff_f is {len(autodiff_f)}')
ground_f = test_set[:, 3] * (test_set[:, 3] ** 2 - 1)
print(f'length of ground_f is {len(ground_f)}')
autodiff_to_ground = jnp.sum(jnp.abs(autodiff_f - ground_f) ** 2) / len(autodiff_f)
print(f'autodiff_to_ground: {autodiff_to_ground:.2e}')

# FDM
dt = args.dt
dx = args.dx
dy = args.dy
test_set_dt = test_set.copy()
test_set_dt[:, 0] = test_set_dt[:, 0] + dt
test_set_plus_dx = test_set.copy()
test_set_plus_dx[:, 1] += dx
test_set_minus_dx = test_set.copy()
test_set_minus_dx[:, 1] -= dx
test_set_plus_dy = test_set.copy()
test_set_plus_dy[:, 2] += dy
test_set_minus_dy = test_set.copy()
test_set_minus_dy[:, 2] -= dy

for j in range(5):
    dt_phi_val_FDM = (vmap(net, (None, 0, None))(model, test_set_dt[:, :3], frozen_para) - vmap(net, (None, 0, None))(model, test_set[:, :3], frozen_para)) / (dt * 0.5**j)
    dxx_phi_val_FDM = (vmap(net, (None, 0, None))(model, test_set_plus_dx[:, :3], frozen_para) - 2 * vmap(net, (None, 0, None))(model, test_set[:, :3], frozen_para) + vmap(net, (None, 0, None))(model, test_set_minus_dx[:, :3], frozen_para)) / (dx * 0.5**j) ** 2
    dyy_phi_val_FDM = (vmap(net, (None, 0, None))(model, test_set_plus_dy[:, :3], frozen_para) - 2 * vmap(net, (None, 0, None))(model, test_set[:, :3], frozen_para) + vmap(net, (None, 0, None))(model, test_Set_minus_dy[:, :3], frozen_para)) / (dy * 0.5**j) ** 2
    FDM_f = (eps ** 2) * (dxx_phi_val_FDM + dyy_phi_val_FDM - (1/gamma) * dt_phi_val_FDM)
    ground_f = test_set[:, 3] * (test_set[:, 3] ** 2 - 1)
    FDM_to_ground = jnp.sum(jnp.abs(FDM_f - ground_f) ** 2) / len(FDM_f)
    print(f'FDM_to_ground: {FDM_to_ground:.2e}')








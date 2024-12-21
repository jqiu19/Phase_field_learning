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

parser = argparse.ArgumentParser(description="free_energy")
parser.add_argument("--interval1", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--interval2", type=str, default="-1.0,1.0", help='boundary of the interval')
parser.add_argument("--ntrain", type=int, default=50000, help="the number of training dataset for each epochs")
parser.add_argument("--ite", type=int, default=10, help="the number of iteration")
parser.add_argument("--epochs", type=int, default=5000, help="the number of epochs")
parser.add_argument("--gamma", type=int, default=1, help='coefficient for Allen-Cahn equation')
parser.add_argument("--eps", type=int, default=0.01, help='parameter for sharpness in Allen-Cahn equation')
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--kanshape", type=str, default="16", help='shape of the network (KAN)')
parser.add_argument("--degree", type=int, default=100, help='degree of polynomials')
parser.add_argument("--features", type=int, default=1000, help='width of the network')
parser.add_argument("--layers", type=int, default=20, help='depth of the network')
parser.add_argument("--len_h", type=int, default=2, help='lenth of k for sinckan')
parser.add_argument("--embed_feature", type=int, default=10, help='embedding features of the modified MLP')
parser.add_argument("--normalization", type=int, default=0, help="add normalization or not, 0: no normalization, "
                                                                 "1: add normalization")
parser.add_argument("--init_h", type=float, default=2.0, help='initial value of h')
parser.add_argument("--decay", type=str, default='inverse', help='exponent for h')
parser.add_argument("--skip", type=int, default=0, help='1: use skip connection for sinckan')
parser.add_argument("--activation", type=str, default='tanh', help="activation function for the networks")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--network", type=str, default="mlp", help="type of network")
parser.add_argument("--device", type=int, default=7, help="cuda number")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)


def net(model, x, frozen_para):
    return model(jnp.stack(x), frozen_para)[0]

def compute_loss(model, ob_txy, frozen_para):
    output = vmap(net, (None, 0, None))(model, ob_txy[:,:3], frozen_para)

    return jnp.sum(jnp.abs(output - ob_txy[:, 3]) ** 2) / len(output)

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step(model, ob_txy, frozen_para, optim, opt_state):
    loss, grads = compute_loss_and_grads(model, ob_txy, frozen_para)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def train(key):
    mat_data = scipy.io.loadmat('/home/qjw/code/python_code/phase_field/FullData_ACequ_2D.mat')
    phiN = mat_data['phiN'].flatten(order = 'F')
    phiN = phiN[:,None]
    print(phiN.shape)
    tN = mat_data['tN'].flatten()
    t = np.repeat(tN, 256*256)
    t = t.reshape(-1, 1)
    print(t)
    xN = mat_data['xyN'][0, :].flatten()
    yN = mat_data['xyN'][1, :].flatten()
    xN = xN.reshape(-1,1)
    yN = yN.reshape(-1,1)
    xN = np.repeat(xN, 256, axis = 0)
    xN_copies = np.tile(xN, (1,299))
    x = xN_copies.flatten(order = 'F').reshape(-1,1)
    print(x)
    yN_copies = np.tile(yN, (1, 256))
    y = yN_copies.flatten(order = 'F').reshape(-1,1)
    y_copies = np.tile(y, (1, 299))
    y = y_copies.flatten(order = 'F').reshape(-1,1)
    print(y)


    ob_txy = np.concatenate([t, x, y, phiN], -1)
    normalizer = normalization(ob_txy, args.normalization)
    input_dim = 3
    output_dim = 1
    # Choose the model
    keys = random.split(key, 2)
    model = get_network(args, input_dim, output_dim, normalizer, keys)
    frozen_para = model.get_frozen_para()
    # Hyperparameters
    N_train = args.ntrain
    N_epochs = args.epochs
    ite = args.ite
    gamma = args.gamma
    eps = args.eps
    N = len(ob_txy)

    # parameters of optimizer
    learning_rate = args.lr
    N_drop = 10000
    alpha = 0.95
    sc = optax.exponential_decay(learning_rate, N_drop, alpha)
    optim = optax.adam(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    keys = random.split(keys[-1], 2)
    x_train = random.choice(keys[0], ob_txy, shape=(N_train,), replace=False)
    history = []
    T = []
    for j in range(ite * N_epochs):
        T1 = time.time()
        loss, model, opt_state = make_step(model, x_train, frozen_para, optim, opt_state)
        T2 = time.time()
        T.append(T2 - T1)
        history.append(loss.item())
        if j % N_epochs == 0:
            keys = random.split(keys[-1], 2)
            x_train = random.choice(keys[0], ob_txy, shape=(N_train,), replace=False)
            train_y_pred = vmap(net, (None, 0, None))(model, x_train[:,:3], frozen_para)
            train_mse_error = jnp.sum(jnp.abs(train_y_pred - x_train[:, 3]) ** 2) / len(train_y_pred)
            train_relative_error = jnp.linalg.norm(train_y_pred - x_train[:, 3]) / jnp.linalg.norm(train_y_pred)
            print(f'ite:{j},mse:{train_mse_error:.2e},relative mse:{train_relative_error:.2e}')

    # eval
    avg_time = np.mean(np.array(T))
    print(f'time: {1 / avg_time:.2e}ite/s')
    keys = random.split(keys[-1], 2)
    x_train = random.choice(keys[0], ob_txy, shape=(N_train,), replace=False)
    train_y_pred = vmap(net, (None, 0, None))(model, x_train[:,:3], frozen_para)
    train_mse_error = jnp.sum(jnp.abs(train_y_pred - x_train[:, 3]) ** 2) / len(train_y_pred)
    train_relative_error = jnp.linalg.norm(train_y_pred - x_train[:, 3]) / jnp.linalg.norm(train_y_pred)
    print(f'ite:{j},mse:{train_mse_error:.2e},relative mse:{train_relative_error:.2e}')

    # save model and results
    path = f'{args.network}_{args.seed}.eqx'
    eqx.tree_serialise_leaves(path, model)
    path = f'{args.network}_{args.seed}.npz'
    np.savez(path, loss=history, avg_time=avg_time, y_pred=train_y_pred,
             mse=train_mse_error, relative_mse=train_relative_error)

    # write the reuslts on csv file
    header = "network, seed, final_loss_mean, training_time, total_ite, mse, relative_mse"
    save_here = "results.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    res = f"\n{args.network},{args.seed},{history[-1]},{np.sum(np.array(T))},{ite * N_epochs},{train_mse_error},{train_relative_error}"
    with open(save_here, "a") as f:
        f.write(res)

# def algebraic_train(key):



if __name__ == "__main__":
    seed = args.seed
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    train(key)
    #eval(key)
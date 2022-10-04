# https://keras.io/examples/generative/vae/
import os, json
import pickle

import numpy as np
import pandas as pd

from tensorflow import keras

import matplotlib.pyplot as plt



def plot_velocities(U, X, Y, case, plot_labels, path = '', show = False):
    fig = plt.figure(figsize = (10, 7))
    columns = 3
    rows = int(np.ceil(len(plot_labels) / columns))
    print('Columns:', columns, 'Rows:', rows, flush = True)
    cmin = np.amin(U)
    cmax = np.amax(U)

    for pi, plot_label in enumerate(plot_labels):
        fig.add_subplot(rows, columns, pi+1)
        cax = plt.pcolor(X, Y, U[:, :, pi])
        plt.title('{}, D = {:.3f}'.format(plot_labels[pi], case))
        plt.clim(cmin, cmax)
        #cbar = fig.colorbar(cax)
        plt.axis('off')

    if show:
        plt.show()
    if path:
        plt.savefig(path)
        


def calculate_UU(U, U_mean):
    U_fluc = U - U_mean
    UU = np.zeros(U_fluc.shape[1:-1] + (6,))
    k =	0
    for	i in range(3):
        for j in range(3):
            if j >= i: # uu, uv, uw, vv, vw, ww
                UU[:, :, k] = np.mean(
                    np.multiply(
                            U_fluc[:, :, :, i],
                            U_fluc[:, :, :, j]
                    ),
                    axis = 0
                )
                k += 1
    return UU #np.expand_dims(UU, axis = 0)    


if __name__ == '__main__':
    # Load test data --> Split into cases by label
    data_dir = 'urban-data'
    model_dir = 'urban-cvae'
    stats_dir = 'urban-stats'
    Lx = 100
    Ly = 100

    os.makedirs(stats_dir, exist_ok = True)

    X = np.load(os.path.join(data_dir, 'IR', 'interp-between-buildings', 'X-{}-{}.npy'.format(Lx,Ly)))
    Y = np.load(os.path.join(data_dir, 'IR', 'interp-between-buildings', 'Y-{}-{}.npy'.format(Lx,Ly)))
    X_test = np.load(os.path.join(model_dir, 'X_test.npy'))
    X_test_r = np.load(os.path.join(model_dir, 'X_test_r.npy'))
    labels_test = np.load(os.path.join(model_dir, 'labels_test.npy'))
    X_gen = np.load(os.path.join(model_dir, 'X_gen.npy'))
    labels_gen = np.load(os.path.join(model_dir, 'labels_gen.npy'))

    print('\nData shape',
          '\nX_test:     ', X_test.shape,
          '\nX_test_r:   ', X_test_r.shape,
          '\nlabels_test:', labels_test.shape,
          '\nX_gen:      ', X_gen.shape,
          '\nlabels_gen: ', labels_gen.shape,
          flush = True
    )

    # Split data in cases:
    cases = np.sort(np.unique(labels_test))

    for case in cases:
        cname = str(case)
        id_test = labels_test == case
        id_gen = labels_gen == case
        print('\nNumber of samples for case D={}: {}'.format(cname, str(len(id_test[id_test]))), flush = True)

        print('\nProcessing original data')
        U = X_test[id_test, :, :, :]
        U_mean = np.mean(U, axis = 0)
        plot_velocities(U_mean, X, Y, case, ['U', 'V', 'W'], path = os.path.join(stats_dir, 'original_U_mean_{}.png'.format(case)))
        UU = calculate_UU(U, U_mean)
        plot_velocities(UU, X, Y, case, ['uu', 'uv', 'uw', 'vv', 'vw', 'ww'], path = os.path.join(stats_dir, 'original_UU_mean_{}.png'.format(case)))

        print('\nProcessing reconstructed data')
        U = X_test_r[id_test, :, :, :]
        U_mean = np.mean(U, axis = 0)
        plot_velocities(U_mean, X, Y, case, ['U', 'V', 'W'], path = os.path.join(stats_dir, 'reconstructed_U_mean_{}.png'.format(case)))
        UU = calculate_UU(U, U_mean)
        plot_velocities(UU, X, Y, case, ['uu', 'uv', 'uw', 'vv', 'vw', 'ww'], path = os.path.join(stats_dir, 'reconstructed_UU_mean_{}.png'.format(case)))

        print('\nNumber of generated samples for case D={}: {}'.format(cname, str(len(id_gen[id_gen]))), flush = True)
        print('\nProcessing generated data')
        U = X_gen[id_gen, :, :, :]
        U_mean = np.mean(U, axis = 0)
        plot_velocities(U_mean, X, Y, case, ['U', 'V', 'W'], path = os.path.join(stats_dir, 'generated_U_mean_{}.png'.format(case)))
        UU = calculate_UU(U, U_mean)
        plot_velocities(UU, X, Y, case, ['uu', 'uv', 'uw', 'vv', 'vw', 'ww'], path = os.path.join(stats_dir, 'generated_UU_mean_{}.png'.format(case)))

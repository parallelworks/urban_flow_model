import pickle
import h5py
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def append(n,o, nns, oos):
    nns.append(n)
    oos.append(o)

def plot_1x1_single_component(X, Y, U, path = 'vel.png'):
    fig = plt.figure()
    cax = plt.pcolor(X, Y, U)
    cbar = fig.colorbar(cax)
    plt.savefig(path)


def slice_data(z_plane, fU, fV, fW, Unns, Vnns, Wnns):
    UVW = []
    print('Reading {} references'.format(len(Unns)))
    for i,nu in enumerate(Unns):
        print('Reading reference: ', i, flush = True)

        data_ref_U = fU['#refs#/' + nu]
        data_ref_V = fV['#refs#/' + Vnns[i]]
        data_ref_W = fW['#refs#/' + Wnns[i]]
         # There are 137 references. One of them is not them (#refs#/a) is not a velocity field
        # The other 136 are based on their min and max velocities and 2D plots. 
        try:
            # Assuming the references are sorted in the same way
            # Assuming the one which is not a velocity field is the same
            U = np.array(data_ref_U[0][:])
            V = np.array(data_ref_V[0][:])
            W = np.array(data_ref_W[0][:])
            print(min(U), max(U), U.shape)
            print(min(V), max(V), V.shape)
            print(min(W), max(W), W.shape)
        except:
            print(data_ref_U[0], data_ref_V[0], data_ref_W[0])
            continue
        
        # Assuming all arrays have the same mesh (values and order)
        if i == 0:
            df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'U': U, 'V': V, 'W': W})
            df = df.loc[df['z'].isin([z_plane])]
            indices = df.index
            xu = df.x.unique()
            xu = xu[~np.isnan(xu)]
            yu = df.y.unique()
            yu = yu[~np.isnan(yu)]
            Lx = len(xu)
            Ly = len(yu)
            # Some points are NaN (40000-33808)
            # ASSUMING NAN VALUES ARE THE "NEXT" UNIQUE VALUE. For example,
            # x = [1,2,3,4,1,nan,3,4] --> x = [1,2,3,4,1,2,3,4]
            # Points are sorted first in x, then in y
            X, Y = np.meshgrid(xu, yu)
        else:
            df = pd.DataFrame({'U': U, 'V': V, 'W': W})
            df = df.iloc[indices]

        # Assuming array is sorted in x,y
        UVW_ = np.transpose(np.array([df.U.to_numpy(), df.V.to_numpy(), df.W.to_numpy()]))
        UVW.append(np.reshape(UVW_, (Ly, Lx, 3)))
        break
    
    UVW = np.array(UVW)
    print(UVW.shape)
    return UVW, X, Y


if __name__ == '__main__':
    ds = '10m'
    rpath = 'IR/DATASET'
    print(rpath, ds)
    fU = h5py.File(rpath + ds + '/U.mat')
    fV = h5py.File(rpath + ds + '/V.mat')
    fW = h5py.File(rpath + ds + '/W.mat')

    # READ x, y and z (t points to a reference)
    # There is no np.array(f.get('x'))[1]
    x = np.array(fU.get('x'))[0]
    y = np.array(fU.get('y'))[0]
    z = np.array(fU.get('z'))[0]
    t = np.array(fU.get('t'))[0]

    print(x.shape, y.shape, z.shape, t.shape)
    print(min(x), min(y), min(z), min(t))
    print(max(x), max(y), max(z), max(t))

    z_unique = np.unique(z)
    z_unique_abs = np.abs(z_unique)
    z_sorted = [zi for _, zi in sorted(zip(z_unique_abs, z_unique))]
    print('z_sorted: ', z_sorted[:4])
    z_plane = z_sorted[0]
    
    # READ U (There are references)
    Unns = []
    Vnns = []
    Wnns = []
    oos = []

    #fW['#refs#'].visititems(lambda n,o: append(n, o, Wnns, oos))
    fU['#refs#'].visititems(lambda n,o: append(n, o, Unns, oos))
    fV['#refs#'].visititems(lambda n,o: append(n, o, Vnns, oos))
    fW['#refs#'].visititems(lambda n,o: append(n, o, Wnns, oos))

    # Average closest two planes to z = 0
    UVW1, X, Y = slice_data(z_sorted[0], fU, fV, fW, Unns, Vnns, Wnns)
    UVW2, X, Y = slice_data(z_sorted[1], fU, fV, fW, Unns, Vnns, Wnns)
    UVW = (UVW1 + UVW2) / 2
    
    for i in range(3):
        path = os.path.join(rpath + ds, 'vel_{}-ds-'.format(str(i)) + ds + '.png')
        plot_1x1_single_component(X, Y, UVW[0,:,:,i], path = path)


    fpath = os.path.join(
        rpath + ds,
        'ds-' + ds + '.npy'
    )
    with open(fpath, 'wb') as f:
        np.save(f, UVW)


    fpath = os.path.join(
        rpath + ds,
        'X-' + ds + '.npy'
    )
    with open(fpath, 'wb') as f:
        np.save(f, X)

    fpath = os.path.join(
        rpath + ds,
        'Y-' + ds + '.npy'
    )
    with open(fpath, 'wb') as f:
        np.save(f, Y)
    #UVW = np.load(fpath)
    #print(UVW.shape)
    print(rpath, ds)

import os, glob

import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d
import numpy as np

urban_data_dir = 'urban-data'
# Building width                   
Bw = 0.5
x0 = Bw/2 # End of first building
# Common DT for all fields:
dt_ref = 0.07

# Dimensions of new mesh:
Ly = 100
Lx = 100


def load_resample_merge_fields(data_dir):
    data_by_dt = {}
    for fpath in glob.glob(os.path.join(data_dir, 'dt-*')):
        dt = os.path.basename(fpath).split('-')[1]
        if dt not in data_by_dt:
            data_by_dt[dt] = []
        data_by_dt[dt].append(fpath)

    data = []

    for dt,fpaths in data_by_dt.items():
        print('DT: ', dt)
        print('Paths: ' + ' '.join(fpaths))
        data_dt = np.concatenate([ np.load(fpath) for fpath in fpaths ])
        print('Original shape: ', data_dt.shape)
        if float(dt) < dt_ref:
            ds = int(dt_ref / float(dt))
        else:
            # Only makes sense if dt_ref is large enough to consider fields independent!                    
            ds = 1
        data_dt = data_dt[0::ds, :, :]
        print('Processed shape: ', data_dt.shape)
        data.append(data_dt)
    data = np.concatenate(data)
    print('Resampled merged data: ', data.shape)
    return np.load(os.path.join(data_dir, 'X.npy')), np.load(os.path.join(data_dir, 'Y.npy')), data


# Map function so that space between buildings (d) is between 0 and 1 
def map_x(x, x0, d):
    return x/d-x0

def load_merge_fields(data_dir):
    data = []
    for fpath in glob.glob(os.path.join(data_dir, 'ds-*.npy')):
        data.append(np.load(fpath))
    return np.load(os.path.join(data_dir, 'X.npy')), np.load(os.path.join(data_dir, 'Y.npy')), np.concatenate(data, axis = 0)

def plot_1x1_single_component(X, Y, U, path = 'vel.png'):
    fig = plt.figure()
    cax = plt.pcolor(X, Y, U)
    cbar = fig.colorbar(cax)
    plt.savefig(path)


def interpolate_tensor_2d(X, Y, U, Xc, Yc):
    Uc = np.zeros((U.shape[0], Yc.shape[0], Xc.shape[0], U.shape[3]))
    for sample in range(U.shape[0]):
        for vi in range(U.shape[3]):
            finterp = interp2d(X, Y, U[sample, :, :, vi], kind = 'cubic')
            Uc[sample, :, :, vi] = finterp(Xc, Yc)
    return Uc

def trim_tensor(X, Y, x_min, x_max, y_min, y_max, U = None):
    x = X[0, :]
    y = Y[:, 0]
    xi_min = np.absolute(x - x_min).argmin()
    xi_max = np.absolute(x - x_max).argmin()
    # Y is sorted the wrong way!
    yi_max = np.absolute(y - y_max).argmin()
    yi_min = np.absolute(y - y_min).argmin()
    if isinstance(U, (list, tuple, np.ndarray)):
        return X[yi_min:yi_max, xi_min:xi_max], Y[yi_min:yi_max, xi_min:xi_max], U[:, yi_min:yi_max, xi_min:xi_max, :]
    else:
        return X[yi_min:yi_max, xi_min:xi_max], Y[yi_min:yi_max, xi_min:xi_max]
    


def process_field(field_info, Xc, Yc):
    DIR = field_info['DIR']
    OUT_DIR = os.path.join(DIR, 'interp-between-buildings')
    os.makedirs(OUT_DIR, exist_ok = True)
    SCALE = field_info['DISTANCE'] - Bw

    #X, Y, U = load_merge_fields(DIR)
    X, Y, U = load_resample_merge_fields(DIR)
    X = np.flip(X, axis = 0)
    Y = np.flip(Y, axis = 0)
    U = np.flip(U, axis = 1)
    #U = U[0:1, :, :, :]                                                                                                                                 
    
    X = (X-x0)/SCALE

    # This is for efficiency only!
    X, Y, U = trim_tensor(X, Y, -0.1, 1.1, 0, 1, U = U)

    print(X.shape, Y.shape, U.shape, Xc.shape, Yc.shape)

    U = interpolate_tensor_2d(X[0, :], Y[:, 0], U, Xc[0, :], Yc[:, 0])
    X = Xc
    Y = Yc

    print(X.shape, Y.shape, U.shape)

    # Set the walls to 0!
    U[:, 0, :, :] = 0
    U[:, :, 0, :] = 0
    U[:, :, Xc.shape[1]-1, :] = 0

    #X, Y, U = trim_tensor(X, Y, 0, 1, 0, 1, U = U)
    #print(X.shape, Y.shape, U.shape)

    for i in range(3):
        path = os.path.join(OUT_DIR, 'vel_{}-ds-'.format(str(i)) + '.png')
        plot_1x1_single_component(X, Y, U[0, :, :, i], path = path)

    with open(os.path.join(OUT_DIR, 'X-{}-{}.npy'.format(Lx, Ly)), 'wb') as f:
        np.save(f, X)
    
    with open(os.path.join(OUT_DIR, 'Y-{}-{}.npy'.format(Lx, Ly)), 'wb') as f:
        np.save(f, Y)
    
    with open(os.path.join(OUT_DIR, 'UVW-{}-{}.npy'.format(Lx, Ly)), 'wb') as f:
        np.save(f, U)

if __name__ == '__main__':

    Fields = {
        'SF': {
            'DIR': os.path.join(urban_data_dir, 'SF'),
            'DISTANCE': 1.5
        },
        'WI': {
            'DIR': os.path.join(urban_data_dir, 'WI'),
            'DISTANCE': 2.5
        },
        'IR': {
            'DIR': os.path.join(urban_data_dir, 'IR'),
            'DISTANCE': 4.5
        }
    } 

    
    # Base mesh is SF mesh
    # - All cases are interpolated to this mesh!
    #REF_MESH = 'IR'
    #Xc = np.load(os.path.join(Fields[REF_MESH]['DIR'], 'X.npy'))
    #Xc = (Xc-x0)/(Fields[REF_MESH]['DISTANCE'] - Bw)
    #Yc = np.load(os.path.join(Fields[REF_MESH]['DIR'], 'Y.npy'))
    #Xc = np.flip(Xc, axis = 0)
    #Yc = np.flip(Yc, axis = 0)
    #print(Xc.shape, Yc.shape)
    #Xc, Yc = trim_tensor(Xc, Yc, 0, 1, 0, 1)
    #print(Xc.shape, Yc.shape)

    step_x = 1/(Lx-1)
    step_y = 0.03 #1/(Ly-1)
    
    xi = np.arange(0, 1+step_x, step_x)
    yi = np.geomspace(step_y, 1+step_y, Ly) - step_y

    Xc, Yc = np.meshgrid(xi, yi)
    #print(Ly)
    #print(xi.shape)
    #print(xi)
    #print(Yc[:,0])
    # Distance between x points is 0.0058, there is no point at 0 or 1
    #print(min(Xc[0,:]), max(Xc[0, :]))
    #quit()
    for k,v in Fields.items():
        print('Processing: ', k)
        process_field(v, Xc, Yc)

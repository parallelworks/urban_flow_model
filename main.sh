#!/bin/bash
set -e
data_dir='urban-data'
model_dir='urban-cvae'
stats_dir='urban-stats'
Lx=100
Ly=100

# Map region between buildings to 0 <= x <= 1 and 0 <= y <= 1
# INPUTS:
# - urban_data_dir: Directory with the centerplane data
#       Structure:
#           urban-data/
#           ├── IR
#           ├── SF
#           └── WI
# - Bw: Building width
# - dt_ref: Reference time step
#       If dt is a multiple of dt_ref: resamples every dt_ref/dt
#       If dt is not a multiple of dt_ref: resamples every int(dt_ref/dt)
#       If dt_ref < dt: Use all samples (resample every 1). This assumes that the samples are already independent!
# - Ly: Dimensions of the new mesh in the y-direction
# - Lx: Dimensions of the new mesh in the x-direction
# OUTPUTS:
# - Interpolated data in directories:
#       urban-data/
#       ├── IR
#       │   └── interp-between-buildings
#       ├── SF
#       │   └── interp-between-buildings
#       └── WI
#       └── interp-between-buildings
# - Every subdirectory contains three numpy arrays:
#       Mesh: X-<Lx>-<Ly>.npy and Y-<Lx>-<Ly>.npy (Same for all cases)
#       Velocities: UVW-<Lx>-<Ly>.npy (shape: nsamples x Ly x Lx x 3)
python3 map-interpolate.py \
	--urban_data_dir ${data_dir} \
	--Bw 0.5 \
	--dt_ref 0.07 \
	--Ly ${Ly} \
	--Lx ${Lx}


# Train ML model and/or use existing model to generate synthetic data
# INPUTS:
# - data_dir: Directory with centerplane data (same as urban_data_dir)
# - Lx and Ly: Dimensions of the mesh in the x and y directions. Here they are only used to load the data
# - num_samples_per_case: Number of samples per case (SF, IR, WI) to avoid overrepresenting one case
# - latent_dim: Latent dimension of the model
# - epochs: Maximum number of epochs. Training stops if validation loss does not improve after <patience> iterations
# - patience: Training stops if validation loss does not improve after <patience> iterations
# - batch_size: Batch size
# - n_conv_layers: Number of convolutional layers for the encoder and decoder
# - strides: Strides for each layer separated by ---
#       If there are less strides than layers the last stride is used for the inner layers
#       Stride must be 1 or 2 (code fails if >2)
# - base_filters: Number of filters of the first conv layer. Filters double with every layer
# - kernel_size: Width and height of the 2D convolution window or receptive field. USE ONLY 3!
# - lr: Initial learning rate for the learning schedule
# - train: True to train model or False to load existing model
# - model_dir: Directory to save or load the model
# - num_synthetic_samples: Number of synthetic samples to generate for original and interpolated labels
# - interp_labels: Interpolated labels to generate synthetic data
# OUTPUTS:
# - All outputs are written to the model_dir
# - Outputs include images, weights, reconstructed and generated data
python3 main-cvae.py \
	--data_dir ${data_dir} \
	--num_samples_per_case 441 \
	--Lx ${Lx} \
	--Ly ${Ly} \
	--latent_dim 10 \
	--epochs 1000 \
	--patience 20 \
	--batch_size 128 \
	--n_conv_layers 3 \
	--strides 2---2 \
	--base_filters 64 \
	--kernel_size 3 \
	--lr 0.01 \
	--train False \
	--model_dir ${model_dir} \
	--num_synthetic_samples 3000 \
	--interp_labels 0.1---0.2---0.4---0.5---0.6---0.7---0.8---0.9 


# Turbulent statistics: Mean velocities and Reynolds stress components
# INPUTS:
# OUTPUTS:
python3 turb-statistics.py \
	--data_dir ${data_dir} \
	--model_dir ${model_dir} \
	--stats_dir ${stats_dir} \
	--Lx ${Lx} \
	--Ly ${Ly}
	

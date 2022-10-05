# https://keras.io/examples/generative/vae/
import os, json
import pickle
import argparse

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical   

import matplotlib.pyplot as plt

def read_args():
    parser=argparse.ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg)
    args=vars(parser.parse_args())
    print(args)
    return args


# CNN require huge amounts of RAM, specially during training
# Computational requirements for a single CNN layer:
#     - Number of parameters: (+1 is the bias term)
#         (filter width * filter height * channels + 1 ) * feature maps 
#     - Number of float multiplications:
#         feature maps * feature map width * feature map height * filter width * filfet height * channels
#             ---> For the first layer feature map width and height are the inputs width and height divided by the strides!!
#     - Output Memory:
#         feature maps * feature map width * feature map height * 32 bits * training batch size 
#             ---> Reduce batch size to fir in memory!
# Memory for multiple layers:
#    - During training: The sum of all the layers
#    - During inference: Two consecutive layers

# If you have memory issues:
# 1. Reuduce batch size
# 2. Increase stride
# 3. Remove layers
# 4. Use 16-bit floats
# 5. Distribute across multiple devices

# Pooling layers: shrink the input image to reduce computational load, memory usage and number of parameters
# - Pooling neurons have no weights
# - keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# -              AvgPool2D
# - Only takes the maximum (or average, minimum, etc) value of the pooling_size window
# - If stride > 1 reduces the number of parameters. Stride defaults to pool_size
# - Introduces a level of invariance to small translations and some rotational and scale invariance
# - Invariance is desirable for classification but undersirable for semantic segmentation
# - Invariance is desirable if a small change in the inputs should lead to a small change on the outputs
# - MaxPool2D typically has better performance
# - Can also be applied to depth (channels) instead of spatial direction (see page 469)
# - GlobalAvgPool2D: Calculates the average of the entire feature map

# Typical CNN architecture:
# - Convolutions(s) with same number of feature maps --> Pooling --> Convolution(s) --> Pooling --> repeat --> Flatten --> Fully connected layer
# - Image gets smaller (less width and height) and deeper (more feature maps)                                  --> Dense --> Dropout --> Repeat  
# - Avoid large kernels (window sizes)--> use more layers (except in first layer)
# - Number of repetitions is a hyperparameter to tune
# - Double number of filters after each pooling layer
# - Add keras.layers.Dropout(0.5) to reduce overfitting between dense layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# Expand strides. For example:
# 1. stride = 2 and 3 layers ----------> stride = [2, 2, 2]
# 2. stride = [1,2] and 3 layers ------> stride = [1, 2, 2]
# 3. stride = [1, 2, 3] and 3 layers --> stride = [1, 2, 3]
def get_stride(n, strides):
    if type(strides) != list:
        strides = [strides]
    
    if n <= len(strides)-1:
        return strides[n]
    else:
        return strides[-1]
         

# Multiplies base filter by stride maintain constant number of neurons per layer
def strides2filters(base_filter, strides):
    filters = [base_filter]
    for stride in strides:
        base_filter = base_filter * stride
        filters.append(base_filter)
    return filters[:-1]
    
    
def build_encoder(latent_dim, height, width, channels, label_dim, n_conv_layers = 2, kernel_size = 3, strides = 2, base_filters = 32):
    encoder_inputs = keras.Input(shape=(height, width, channels))
    label_inputs = keras.layers.Input(shape = [label_dim]) # FIXME: shape may vary
    # Number of filters: 32
    # kernel_size: 3 (integer or list of two integers) -> Width and height of the 2D convolution window or receptive field
    # strides: Shift from one window to the next. The output size is the input size size divided by the stride (rounded up)
    # pading:
    #     - same: Uses zero padding when stride and filter width don't match input width
    #     - valid: Ignores inputs to fit the input width to the stride and filter width
    #x = layers.Conv2D(32, 20, activation="relu", strides=(7,10), padding="same")(encoder_inputs)

    # Expand strides. For example:
    # 1. stride = 2 and 3 layers ----------> stride = [2, 2, 2]
    # 2. stride = [1,2] and 3 layers ------> stride = [1, 2, 2]
    # 3. stride = [1, 2, 3] and 3 layers --> stride = [1, 2, 3]    
    strides = [get_stride(n,strides) for n in range(n_conv_layers) ]
    # Multiplies base filter by stride maintain constant number of neurons per layer
    filters = strides2filters(base_filters, strides)
    filters = strides2filters(base_filters, strides)
    final_height = calculate_final_shape(height, strides)
    final_width = calculate_final_shape(width, strides)
    final_filters = filters[-1]


    
    print('Strides: ', strides)
    print('Filters: ', filters)
    for l in range(n_conv_layers):
        if l == 0:
            x = layers.Conv2D(filters[l], kernel_size, activation="relu", strides=strides[l], padding="same")(encoder_inputs)
        else:
            x = layers.Conv2D(filters[l], kernel_size, activation="relu", strides=strides[l], padding="same")(x)
        # FIXME: Why are there no pooling layers?
        
    x = layers.Flatten()(x)
    concat = layers.Concatenate()([x, label_inputs])
    #x = layers.Dense(final_height * final_width * final_filters, activation="relu")(concat) #(x)
    x = layers.Dense(1000, activation="relu")(concat) #(x)
    #x = layers.Dense(500, activation="relu")(x) #(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model([encoder_inputs, label_inputs], [z_mean, z_log_var, z], name="encoder")
    print(encoder.summary())
    return encoder



def calculate_final_shape(height, strides):
    for s in strides:
        height = int(np.ceil(height / s))
    return height

# Only works for stride 2!!
# - Stride must be greater than output padding
def calculate_output_paddings(height, strides):
    print(strides)
    output_paddings = []
    for s in strides:
        output_paddings.append(s - (height % s) - 1)
        height = int(np.ceil(height / s))

    output_paddings.reverse()
    return output_paddings

def build_decoder(latent_dim, height, width, channels, label_dim, n_conv_layers = 2, kernel_size = 3, strides = 2, base_filters = 32):
    latent_inputs = keras.Input(shape=(latent_dim,))
    label_inputs = keras.layers.Input(shape = [label_dim]) # FIXME: shape may vary

    # Expand strides. For example:
    # 1. stride = 2 and 3 layers ----------> stride = [2, 2, 2]
    # 2. stride = [1,2] and 3 layers ------> stride = [1, 2, 2]
    # 3. stride = [1, 2, 3] and 3 layers --> stride = [1, 2, 3]
    strides = [ get_stride(n,strides) for n in range(n_conv_layers) ]
    # Multiplies base filter by stride maintain constant number of neurons per layer
    filters = strides2filters(base_filters, strides)
    
    final_height = calculate_final_shape(height, strides)
    final_width = calculate_final_shape(width, strides)
    height_paddings = calculate_output_paddings(height, strides)
    width_paddings = calculate_output_paddings(width, strides)
    final_filters = filters[-1]

    #final_filters = base_filters*np.power(strides, n_conv_layers-1)[-1]
    concat = layers.Concatenate()([latent_inputs, label_inputs])
    x = layers.Dense(1000, activation="relu")(concat) #(latent_inputs)
    #x = layers.Dense(1000, activation="relu")(x) #(latent_inputs)
    x = layers.Dense(final_height * final_width * final_filters, activation="relu")(x)
    #x = layers.Dense(final_height * final_width * final_filters, activation="relu")(concat) #(latent_inputs)
    x = layers.Reshape((final_height, final_width, final_filters))(x)

    strides.reverse()
    filters.reverse()
    print('Final height:    ', final_height, flush = True)
    print('Final width:     ', final_width, flush = True)
    print('Strides:         ', strides, flush = True)
    print('Filters:         ', filters, flush = True)
    print('Height paddings: ', height_paddings, flush = True)
    print('Width paddings:  ', width_paddings, flush = True)

    for l in range(n_conv_layers):
        x = layers.Conv2DTranspose(
            filters[l],
            kernel_size,
            activation="relu",
            strides = strides[l],
            padding="same",
            output_padding = [
                height_paddings[l],
                width_paddings[l]
            ]
        )(x)
         
    decoder_outputs = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model([latent_inputs, label_inputs], decoder_outputs, name="decoder")
    print(decoder.summary())
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, loss_scale, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.loss_scale = loss_scale

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data = list(data[0])
        X, Y, labels = data[0], data[1], data[2]
    
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([X, labels])
            reconstruction = self.decoder([z, labels])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(Y, reconstruction), axis=(1, 2)
                )
            )/self.loss_scale
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))/self.loss_scale
            total_loss = (reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # Needed to validate (validation loss) and to evaluate
    def test_step(self, data):
        try:
            X, Y, labels = data
        except:
            data = list(data[0])
            X, Y, labels = data[0], data[1], data[2]    
    
        z_mean, z_log_var, z = self.encoder([X, labels])
        reconstruction = self.decoder([z, labels])
        # FIXME: Normalize loss with the number of features (28*28)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(Y, reconstruction), axis=(1, 2)
            )
        )/self.loss_scale
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))/self.loss_scale
        total_loss = (reconstruction_loss + kl_loss)
        #grads = tape.gradient(total_loss, self.trainable_weights)
        #self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }



def plot_velocities(U, label, rows, columns, X, Y, vc = 0, path = '', show = False):
    fig = plt.figure(figsize = (10, 7))

    for i, ui in enumerate(U):
        fig.add_subplot(rows, columns, i+1)
        cax = plt.pcolor(X, Y, ui[:,:,vc])
        plt.title('D = {:.3f}'.format(label[i]))
        plt.clim(0, 1)
        #cbar = fig.colorbar(cax)
        plt.axis('off')

    if show:
        plt.show()
    if path:
        plt.savefig(path)



def load_urban_data(data_dir, n_samples, Lx, Ly):
    Lstr = str(Lx) + '-' + str(Ly)
    SF_U = np.load('urban-data/SF/interp-between-buildings/UVW-{}.npy'.format(Lstr))[-n_samples:]
    WI_U = np.load('urban-data/WI/interp-between-buildings/UVW-{}.npy'.format(Lstr))[-n_samples:]
    IR_U = np.load('urban-data/IR/interp-between-buildings/UVW-{}.npy'.format(Lstr))[-n_samples:]
    X = np.load('urban-data/IR/interp-between-buildings/X-{}.npy'.format(Lstr))
    Y = np.load('urban-data/IR/interp-between-buildings/Y-{}.npy'.format(Lstr))

    print('SF', SF_U.shape, 'WI', WI_U.shape, 'IR', IR_U.shape)
    SF_labels = np.full(SF_U.shape[0], 1.5, dtype = float)
    WI_labels = np.full(SF_U.shape[0], 2.5, dtype = float)
    IR_labels = np.full(SF_U.shape[0], 4.5, dtype = float)
    U = np.concatenate([SF_U, WI_U, IR_U])
    D = np.concatenate([SF_labels, WI_labels, IR_labels])
    
    #U = SF_U
    #D = SF_labels
    print(U.shape, D.shape)
    # U scale:
    USC_max = np.amax(U)
    USC_min = np.amin(U)
    # D scale:
    DSC_max = np.amax(D)
    DSC_min = np.amin(D)
    # Scale:
    U = (U - USC_min) / (USC_max - USC_min)
    D = (D - DSC_min) / (DSC_max - DSC_min)
    print(np.amin(U), np.amax(U))

    return U, D, USC_max, USC_min, DSC_max, DSC_min, X, Y
    

    

if __name__ == '__main__':
    args = read_args()
    data_dir = args['data_dir']
    num_samples_per_case = int(args['num_samples_per_case'])
    Lx = args['Lx']
    Ly = args['Ly']
    latent_dim = int(args['latent_dim']) #  best(100, 50, 10, 5) # Worse (2) best (10 with 64 base filters)
    train = args['train']
    if train.lower() == 'true':
        train = True
    else:
        train = False
    epochs = int(args['epochs'])
    batch_size = int(args['batch_size'])
    patience = int(args['patience'])
    n_conv_layers = int(args['n_conv_layers'])  # Best (3, 4) Worse (5)
    stride = [ int(i) for i in args['strides'].split('---') ] # # Use 1 or 2 only
    kernel_size = int(args['kernel_size']) # Use only 3!
    base_filters = int(args['base_filters'])  # best (64, 32, 16) # worse (128)
    model_dir = str(args['model_dir'])
    num_synthetic_samples = int(args['num_synthetic_samples'])  # Number of samples to be generated by the model
    lr = float(args['lr'])  #(best) 0.01 (nan) 0.0001, 0.005 (worse)
    os.makedirs(model_dir, exist_ok = True)

    # X and Y here are the mesh!
    data, labels, USC_max, USC_min, DSC_max, DSC_min, X, Y = load_urban_data(data_dir, num_samples_per_case, Lx, Ly)

    # X_* and Y_* here are the features and the target, respectively
    X_train, X_test, labels_train, labels_test = train_test_split(data, labels)
    X_train, X_valid, labels_train, labels_valid = train_test_split(X_train, labels_train)
    Y_train = X_train
    Y_valid = X_valid
    Y_test = X_test
    
    # INCREASE NUMBER OF SAMPLES AND NOT USE TEST SET (DID NOT IMPROVE RESULTS)
    #X_train = np.concatenate([X_train, X_test])
    #Y_train = np.concatenate([Y_train, Y_test])
    #labels_train = np.concatenate([labels_train, labels_test])

    print('\nDATA SHAPES:',
          '\n    X_train:     ', X_train.shape,
          '\n    Y_train:     ', Y_train.shape,
          '\n    labels_train:', labels_train.shape,
          '\n    X_valid:     ', X_valid.shape,
          '\n    Y_valid:     ', Y_valid.shape,
          '\n    labels_valid:', labels_valid.shape,
          '\n    X_test:      ', X_test.shape,
          '\n    Y_test:      ', Y_test.shape,
          '\n    labels_test: ', labels_test.shape,
          flush = True
    )

    num_samples, height, width, channels = X_train.shape
    
    if len(labels_train.shape) > 1:
        label_dim = labels_train.shape[1]
    else:
        label_dim = 1
    
    # Plot some inputs
    rows = 3
    columns = 4
    plot_velocities(X_train[0:rows*columns], labels_train[0:rows*columns], rows, columns, X, Y, vc = 0, path = os.path.join(model_dir, 'Ux.png'))
    plot_velocities(X_train[0:rows*columns], labels_train[0:rows*columns], rows, columns, X, Y, vc = 1, path = os.path.join(model_dir, 'Uy.png'))
    plot_velocities(X_train[0:rows*columns], labels_train[0:rows*columns], rows, columns, X, Y, vc = 2, path = os.path.join(model_dir, 'Uz.png'))

    encoder = build_encoder(latent_dim, height, width, channels, label_dim, n_conv_layers, kernel_size, stride, base_filters)
    decoder = build_decoder(latent_dim, height, width, channels, label_dim, n_conv_layers, kernel_size, stride, base_filters)
    vae = VAE(encoder, decoder, height*width)

    if train:
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=100,
            decay_rate=0.9
        )

        optimizer = keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=True,
            name="RMSprop"
        )
        
        vae.compile(optimizer = optimizer)

        early_stopping_cb = keras.callbacks.EarlyStopping(patience = patience, restore_best_weights = True)
        history = vae.fit(
            (X_train, Y_train, labels_train), epochs = epochs, batch_size = batch_size,
            callbacks = [early_stopping_cb],
            validation_data = (X_valid, Y_valid, labels_valid)
        )
        vae.save_weights(os.path.join(model_dir, 'vae'))

        hist_pd = pd.DataFrame(history.history)
        hist_pd.to_csv(os.path.join(model_dir, 'history.csv'), index = False)
        hist_plot = hist_pd.plot()
        hist_plot.figure.savefig(os.path.join(model_dir, 'history.png'), dpi = 500)
        
        test_loss = vae.evaluate((X_test, Y_test, labels_test))
        test_loss = dict(zip(["loss", "reconstruction_loss", "kl_loss"], test_loss))
        print('Test loss:')
        print(test_loss)
        with open(os.path.join(model_dir, 'test_loss.json'), 'w') as json_file:
            json.dump(test_loss, json_file, indent = 4)
            
    else:
        vae.load_weights(os.path.join(model_dir, 'vae'))

        
    print('GENERATING SYNTHETIC NEW IMAGES', flush = True)
    codings = tf.random.normal(shape = [num_synthetic_samples, latent_dim])
    labels_gen = np.random.choice(labels, size = num_synthetic_samples)
    X_gen = vae.decoder([codings, labels_gen]).numpy()

    # Sample and plot data
    columns = 4
    rows = 4
    plot_velocities(X_gen[0:rows*columns], labels[0:rows*columns], rows, columns, X, Y, vc = 0, path = os.path.join(model_dir, 'Ux-gen.png'))
    plot_velocities(X_gen[0:rows*columns], labels[0:rows*columns], rows, columns, X, Y, vc = 1, path = os.path.join(model_dir, 'Uy-gen.png'))
    plot_velocities(X_gen[0:rows*columns], labels[0:rows*columns], rows, columns, X, Y, vc = 2, path = os.path.join(model_dir, 'Uz-gen.png'))

    # Save data
    with open(os.path.join(model_dir, 'X_gen.npy'), 'wb') as f:
        np.save(f, X_gen)
    with open(os.path.join(model_dir, 'labels_gen.npy'), 'wb') as f:
        np.save(f, labels_gen)

    print('RECONSTRUCTING (ENCODE -> DECODE) TEST DATA', flush = True)
    z_mean, z_log_var, z = vae.encoder([ X_test, labels_test ])
    X_test_r = vae.decoder([ z, labels_test ])

    # Sample and plot data:
    rows = 3
    columns = 4
    X_test_s = X_test[0:rows*columns]
    labels_test_s = labels_test[0:rows*columns]
    X_test_r_s = X_test_r[0:rows*columns]
    plot_velocities(X_test_s, labels_test_s, rows, columns, X, Y, vc = 0, path = os.path.join(model_dir, 'Ux-test-orig.png'))
    plot_velocities(X_test_s, labels_test_s, rows, columns, X, Y, vc = 1, path = os.path.join(model_dir, 'Uy-test-orig.png'))
    plot_velocities(X_test_s, labels_test_s, rows, columns, X, Y, vc = 2, path = os.path.join(model_dir, 'Uz-test-orig.png'))
    plot_velocities(X_test_r_s, labels_test_s, rows, columns, X, Y, vc = 0, path = os.path.join(model_dir, 'Ux-test-reconstructed.png'))
    plot_velocities(X_test_r_s, labels_test_s, rows, columns, X, Y, vc = 1, path = os.path.join(model_dir, 'Uy-test-reconstructed.png'))
    plot_velocities(X_test_r_s, labels_test_s, rows, columns, X, Y, vc = 2, path = os.path.join(model_dir, 'Uz-test-reconstructed.png'))

    # Save data:
    with open(os.path.join(model_dir, 'labels_test.npy'), 'wb') as f:
        np.save(f, labels_test)
        
    with open(os.path.join(model_dir, 'X_test.npy'), 'wb') as f:
        np.save(f, X_test)

    with open(os.path.join(model_dir, 'X_test_r.npy'), 'wb') as f:
        np.save(f, X_test_r)

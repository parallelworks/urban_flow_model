# Urban Flow Model
The aim of this repository is to create a ML generative model of urban flows using numerical data from LES simulations as the training data. The numerical data must be downloaded before running the code.

### 1. extract-centerplane.py:
- Takes the full 3D velocity fields and extracts and saves the centerplane (z=0).

### 2. map-interpolate.py:
- Loads, trims, maps and interpolates the centerplane data to generate the training data.

### 3. main-cvae.py:
- Trains a CVAE model using the traning data or loads an existing model
- Generates synthetic velocity fields
- Encodes and decodes the test set
- Can use the docker repository `tensorflow/tensorflow:latest-gpu-jupyter` to train the model (`pip install scikit-learn pandas`)


#### 4. turb-statustics.py:
- Calculates the mean velocity fields and the components of the Reynolds stress tensor for the original, reconstructed (encode --> decode) and generated (random --> decode) data from 3. 
# Facial-Recognition-VGG-Face
# Facial-Recognition-VGG-Face
# Deep Learning Facial Recognition System Using VGGFace, Keras and TensorFlow.

## Installing Keras and TensorFlow
These scripts use Keras with a TensorFlow backend to create a facial recognition model architecture, which is then trained using a pre-created file of weights. You will need to install Keras from [here](https://keras.io/#installation) and TensorFlow from [here.](https://www.tensorflow.org/install/)

### Note
At the time of writing this, TensorFlow will not install on python 3.8. To check your python version run
```
python --version
```
If it is higher than 3.7.5, you can run 
```
pip install python=3.7.5
```
to downgrade it. Then TensorFlow should install correctly.

## Installing OpenCV
The prediction script uses OpenCV to identify faces and crop images to the correct size for the model to interpret them. OpenCV can be found [here.](https://opencv.org/releases.html)
It can then be installed using
```
pip install opencv-python
```

## Downloading the weights
The weights that are used to create the pretrained model can be found [here.](http://www.vlfeat.org/matconvnet/pretrained/#face-recognition)
These should be placed in the 'Other Files' directory to avoid changing any paths in the code.

## Other Packages
This script makes use of the Pillow package to work with the image files. It can be installed with 
```
pip install Pillow
```

These scripts use numpy to work with arrays
```
pip install numpy
```

The models are created as HDF5 files. Python uses h5py to work with these files. h5py should be included with the keras package, but if pip you don't have it you can install it by running
```
pip install h5py
```

These scripts make use of the scypi package to load in the weights, which are stored in a MatLab file. 
```
pip install scypi
```


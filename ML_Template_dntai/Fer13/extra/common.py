import time, cv2
import numpy as np
from math import ceil
from IPython import display
import matplotlib.pyplot as plt
import pandas as pd

# Keras
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy, categorical_crossentropy

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation

from keras.models import Model, Sequential, load_model

from keras import backend as K

from keras.callbacks import CSVLogger, LambdaCallback, ModelCheckpoint, EarlyStopping, TensorBoard
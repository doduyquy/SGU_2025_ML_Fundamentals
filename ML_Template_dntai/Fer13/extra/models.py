from keras.layers import Flatten, Dropout, Dense
from keras.models import Model
from keras.regularizers import l2
from keras.applications import InceptionV3, ResNet50, VGG16

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation

from keras.models import Model, Sequential, load_model

global weight_decay
weight_decay = 1e-4

""" *****************************************
MODELS
***************************************** """
torch_imagenet_mean = [0.485, 0.456, 0.406]
torch_imagenet_std = [0.229, 0.224, 0.225]
    
caffe_imagenet_mean = [103.939, 116.779, 123.68] # BGR
def caffe_preprocessing_input(x): # BGR
    x[..., 0] -= caffe_imagenet_mean[0]
    x[..., 1] -= caffe_imagenet_mean[1]
    x[..., 2] -= caffe_imagenet_mean[2]
    return x
# caffe_preprocessing_input

def torch_preprocessing_input(x): # BGR
    """
    torch: will scale pixels between 0 and 1 
    and then will normalize each channel with respect to the
    ImageNet dataset.
    """
    x = x[...,::-1] # BGR --> RGB
    x /= 255.
    x[..., 0] -= torch_imagenet_mean[0]
    x[..., 1] -= torch_imagenet_mean[1]
    x[..., 2] -= torch_imagenet_mean[2]

    x[..., 0] /= torch_imagenet_std[0]
    x[..., 1] /= torch_imagenet_std[1]
    x[..., 2] /= torch_imagenet_std[2]
    return x
# torch_preprocessing_input

def tf_preprocess_input(x):
    """
    Image RGB
    tf: will scale pixels between -1 and 1, sample-wise.
    """
    x = x[...,::-1] # BGR --> RGB
    x = x / 127.5
    x -= 1.0
    return x
# tf_preprocess_input

def vgg16_preprocessing_input(x):
    return caffe_preprocessing_input(x)
# vgg16_preprocessing_input

def resnet50_preprocessing_input(x):
    return caffe_preprocessing_input(x)
# resnet50_preprocessing_input

def inceptionv3_preprocessing_input(x):
    return tf_preprocess_input(x)
# inceptionv3_preprocessing_input

def build_common_model(weight_path = None, model_name = None, nb_classes = 7, fc=[2048, 0], dropout = [0.1, 0.1, 0.0], input_shape = None):
    # base model
    base_model = None
    if model_name=="imagenet_inception_v3":
        input_shape = (96, 96, 3) if input_shape is None else input_shape
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    elif model_name=="inception_v3":
        input_shape = (96, 96, 3) if input_shape is None else input_shape
        base_model = InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    elif model_name=="imagenet_resnet50":
        input_shape = (48, 48, 3) if input_shape is None else input_shape
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    elif model_name=="resnet50":
        input_shape = (48, 48, 3) if input_shape is None else input_shape
        base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    elif model_name=="imagenet_vgg16":
        input_shape = (48, 48, 3) if input_shape is None else input_shape
        base_model = VGG16(include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    elif model_name=="vgg16":
        input_shape = (48, 48, 3) if input_shape is None else input_shape
        base_model = VGG16(include_top=False, weights=None, input_tensor=None, input_shape=input_shape, pooling="avg", classes=7)
    # if
    
    model = None
    if base_model is not None:
        x = base_model.output
        if dropout[0]>0: x = Dropout(dropout[0])(x)
        if fc[0]>0: x = Dense(fc[0], activation='relu')(x)
        if dropout[1] > 0: x = Dropout(dropout[1])(x)
        if fc[1]>0: x = Dense(fc[1], activation='relu')(x)
        if dropout[2] > 0: x = Dropout(dropout[2])(x)
        x = Dense(nb_classes,
                  activation='softmax',
                  name='predictions',
                  use_bias=False, trainable=True,
                  kernel_initializer='orthogonal',
                  kernel_regularizer=l2(weight_decay))(x)
        model = Model(inputs=base_model.input, outputs=x)
    # if
    
    if weight_path is not None: model.load_weights(weights_path, by_name = True)
    return model
# build_common_model


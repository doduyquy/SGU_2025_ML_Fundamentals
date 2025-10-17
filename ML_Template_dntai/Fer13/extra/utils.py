import os, numpy as np, tqdm, pandas as pd, matplotlib.pyplot as plt, cv2, time
from albumentations import Compose, Resize, CenterCrop, PadIfNeeded, BasicTransform, ToFloat
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import Iterator as KerasIterator
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

# Keras
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense
from keras.layers import Flatten, Dropout, Concatenate, BatchNormalization, Input, Convolution2D, MaxPooling2D, concatenate, Activation
from keras.models import Model, Sequential, load_model
from keras import backend as K
from keras.callbacks import LambdaCallback, Callback, EarlyStopping, TensorBoard

# emotion labels in FER2013 corresponding with [0-6]
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

""" *****************************************
DATA LOADER, AUGMENTATION, AND GENERATOR
***************************************** """ 
class DataGenerator(KerasIterator):
    """
    Keras data generator for emotion recognition data loader
    """
    def __init__(self, dataloader, batch_size = 32, shuffle = True, preprocessing_image_fn = None, n_channels = 1, seed = None):
        """
        """
        self.dataloader = dataloader
        (x, y)          = self.dataloader[0]
        self.x_shape    = x.shape
        self.y_shape    = y.shape
        self.current_batch = np.zeros(batch_size) # save current batch for easy debug
        self.preprocessing_image_fn = preprocessing_image_fn
        self.n_channels = n_channels
        super().__init__(len(self.dataloader), batch_size, shuffle, seed)
    # __init__

    def _get_batches_of_transformed_samples(self, index_array):
        self.current_batch = index_array

        # (height, width, channel)
        if self.n_channels == 1:
            batch_x = np.zeros((len(index_array),) + self.x_shape + (1,), dtype=np.float32)
            batch_y = np.zeros((len(index_array),) + self.y_shape, dtype=np.float32)
            for idx, value in enumerate(index_array):
                x, y = self.dataloader[value]
                batch_x[idx, :, :, 0] = x
                batch_y[idx, ...] = y
            # for
        elif self.n_channels == 3:
            batch_x = np.zeros((len(index_array),) + self.x_shape + (3,), dtype=np.float32)
            batch_y = np.zeros((len(index_array),) + self.y_shape, dtype=np.float32)
            for idx, value in enumerate(index_array):
                x, y = self.dataloader[value]
                batch_x[idx, :, :, 0 : 3] = np.dstack([x, x, x])
                batch_y[idx, ...] = y
            # for
        # if
        if self.preprocessing_image_fn is not None: batch_x = self.preprocessing_image_fn(batch_x)

        output = (batch_x, batch_y)
        return output
        
    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    # next
# DataGenerator

def normal_valid_aug(image_size, p=1.0):
    return Compose([
        Resize(image_size, image_size, p=1),
        PadIfNeeded(min_height=image_size, min_width=image_size, p=1)
    ], p=p)
# normal_valid_aug

def normal_train_aug(image_size, keras_data_aug, resize = 5, p=1.0):
    return Compose([
        Resize(image_size + resize, image_size + resize),
        CenterCrop(image_size, image_size),
        KerasDataAugment(image_data_augument = keras_data_aug),
        PadIfNeeded(min_height=image_size, min_width=image_size, p = 1)
    ], p=p)
# normal_train_aug

class KerasDataAugment(BasicTransform):
    def __init__(self, image_data_augument = None, fit_data_augument = None, always_apply=True, p=1.0):       
        # Data Argument
        self.image_data_augument = dict(
            # NEED TO FEED AND CAREFULLY (SAME WITH VALID), NEED TO FIT
            # divide inputs by std of the dataset
            featurewise_std_normalization = False, 

            # apply ZCA whitening
            zca_whitening      = False, 
            zca_epsilon        = 1e-6,       # epsilon for ZCA whitening   
            featurewise_center = False,# set input mean to 0 over the dataset
                        
            # NEED TO FEED AND CAREFULLY (SAME WITH VALID)
            rescale            = None, #  rescaling factor. If None or 0, no rescaling is applied, otherwise multiply the data by the value provided
    
            samplewise_std_normalization = False, # divide each input by its std
    
            # points outside the boundaries ('constant', 'nearest', 'reflect' or 'wrap'). 
            # 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k) 
            # 'nearest': aaaaaaaa|abcd|dddddddd 
            # 'reflect': abcddcba|abcd|dcbaabcd 
            # 'wrap': abcdabcd|abcd|abcdabcd
            # fill_mode          = 'nearest',
            # cval = 0, #value used for fill_mode 'constant'
                        
            # rotation_range     = 10.0, # degrees (0 to 180)
                        
            # width_shift_range = 0.1, # fraction of total width, if < 1, or pixels if >= 1.
            # height_shift_range= 0.1, # fraction of total height, if < 1, or pixels if >= 1.
            # shear_range       = 0.0, # shear intensity (shear angle in degrees).
            # zoom_range        = 0.1, # amount of zoom. if scalar z, zoom will be randomly picked in the range [1-z, 1+z]. A sequence of two can be passed instead to select this range.
                       
            # horizontal_flip   = True, # whether to randomly flip images horizontally
            # vertical_flip     = False  # whether to randomly flip images vertically
        )
        if image_data_augument is not None: self.image_data_augument.update(**image_data_augument)
        self.image_data_generator = ImageDataGenerator(**image_data_augument)
        if fit_data_augument is not None: self.image_data_generator.fit(**fit_data_augument)
        
        super().__init__(always_apply=always_apply, p=p)
    # __init__

    def __call__(self, force_apply=True, **kwargs):
        """
        """
        self.seed = int(time.time())
        if "mask" in kwargs.keys(): kwargs.update({"mask": self.apply(kwargs["mask"])})
        if "image" in kwargs.keys(): kwargs.update({"image": self.apply(kwargs["image"])})
        return kwargs
    # __call__

    def apply(self, image, **params):
        if len(image.shape) == 2:
            np.random.seed(self.seed)
            img = self.image_data_generator.random_transform(image.reshape(image.shape + (1,)))
            img = self.image_data_generator.standardize(img)
            img = img.reshape(img.shape[0:2])
        elif len(image.shape)>2:
            np.random.seed(self.seed)
            img = self.image_data_generator.random_transform(image)
            img = self.image_data_generator.standardize(img)
        else:
            img = None
        return img
    # apply
    
    def get_transform_init_args_names(self):
        return "image_data_augument", "fit_data_augument"
    # get_transform_init_args_names
# KerasDataAugment

class DataLoader(object):
    def __init__(self, 
                 data,               # (x_train, y_train)
                 transforms = None,  # data augmentation
                 mode = "train",     # train, valid, test
                 capacity = 0,
                 n_classes = 7,      # number of emotions
                 preprocessing_image_fn = None, 
                 **kwargs):
        self.x_data, self.y_data = data[0], data[1]
        self.n_data = len(self.x_data)
        self.capacity = capacity
        self.mode = mode
        self.transforms = transforms
        self.preprocessing_image_fn = preprocessing_image_fn
        self.n_classes  = n_classes
        
        self.idx_all = np.arange(self.n_data)
        pass
    # __init__
    
    def __len__(self):
        if self.capacity == 0: return self.n_data
        return self.capacity
    # __len__
    
    def __getitem__(self, index):
        if index >= self.n_data: index = index % self.n_data
        
        image = self.x_data[index]
        
        if self.transforms is not None: 
            image = self.transforms(image = image.astype(np.float))["image"]
        # if
        
        if self.preprocessing_image_fn is not None:
            image = self.preprocessing_image_fn(image)
        # if
        
        y_data = None
        if self.y_data is not None:
            y_data = np.zeros(self.n_classes)
            y_data[self.y_data[index]] = 1.0
        # if

        return image, y_data
    # __getitem__
    
    def view_image(self, view_ids = list(range(16)), rows = 4, cols = 4, figsize = (8,8), save_path = None):
        if view_ids is None or len(view_ids)==0:
            view_ids = np.random.choice(len(self.x_data), size = rows * cols)
        # if
        a_data = []
        a_label = []
        for idx in view_ids:
            x_data, y_data = self[idx]
            a_data.append(x_data)
            a_label.append(np.argmax(y_data))
        # for
        view_images(a_data, a_label, list(range(len(view_ids))), rows, cols, figsize, save_path)
    # view_image
# DataLoader

""" *****************************************
DATA PRE-PROCESSING
***************************************** """ 

def processing_data(csv_path, output_dir): 
    """
    Read Fer13 dataset and save to numpy array (train, valid, test) with two keys: images, labels
    Input: csv_path, output_dir
    Output: train.npz, valid.npz, test.npz at output_dir
    Usage:
        csv_path = os.path.join(dataset_dir, "fer2013", "fer2013.csv"),
        output_dir = os.path.join(data_dir, "preprocessing")
        processing_data(csv_path, output_dir)
    """
    # read data from csv file
    print("Read csv file: ", csv_path)
    df_fer13 = pd.read_csv(csv_path)

    # read the pixels column into numpy array
    col_images = []
    for idx in tqdm.tqdm(range(len(df_fer13)), desc="Load images: "):
        # split pixels column by blank separator, convert into integer and reshape (48,48)
        image = np.array([int(p) for p in df_fer13["pixels"][idx].split(" ")]).reshape(48, 48)
        # append to list
        col_images.append(image)
    # for
    col_images = np.array(col_images) # convert to numpy array (number images, 48, 48)
    
    # get infomation from emotion (0 - 6), and usage (PrivateTest, PublicTest, Training)
    col_emotion = df_fer13["emotion"].values
    col_usage = df_fer13["Usage"].values
    if (output_dir is not None) and (output_dir!="") and ( os.path.exists(output_dir)==False): 
        print("Make dir: ", output_dir)
        os.makedirs(output_dir)
    # if
    
    for s_filter, s_type in zip(["Training", "PublicTest", "PrivateTest"], ["train", "valid", "test"]):
        bool_filter = (col_usage==s_filter) # create boolean filter for s_filter
        col_emotion_filter = col_emotion[bool_filter]
        col_images_filter  = col_images[bool_filter]
        print("Save %s with %d images at %s" % (s_type, np.sum(bool_filter), "%s/%s.npz"%(output_dir, s_type)))
        np.savez("%s/%s.npz"%(output_dir, s_type), images=col_images_filter, labels = col_emotion_filter)
    # for
# processing_data

def load_data(train_path, valid_path, test_path):
    """
    Load data from saved npz files (train, valid, test)
    Usage:
        train_path = os.path.join(data_dir, "preprocessing", "train.npz")
        valid_path = os.path.join(data_dir, "preprocessing", "valid.npz")
        test_path = os.path.join(data_dir, "preprocessing", "test.npz")
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(train_path, valid_path, test_path)
    """
    print("Read train images: ", train_path)
    info = dict(np.load(train_path, allow_pickle=True))
    x_train, y_train = info["images"], info["labels"]

    print("Read valid images: ", valid_path)
    info = dict(np.load(valid_path, allow_pickle=True))
    x_valid, y_valid = info["images"], info["labels"]

    print("Read test images: ", test_path)
    info = dict(np.load(test_path, allow_pickle=True))
    x_test, y_test = info["images"], info["labels"]
    
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
# load_data

def view_hist_data(x_train, y_train, x_valid, y_valid, x_test, y_test, save_path = None):
    """
    View distribution of train, valid, test data
    Usage:
        train_path = os.path.join(data_dir, "preprocessing", "train.npz")
        valid_path = os.path.join(data_dir, "preprocessing", "valid.npz")
        test_path = os.path.join(data_dir, "preprocessing", "test.npz")
        save_path = os.path.join(data_dir, "preprocessing", "distribution_data.png")
        
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(train_path, valid_path, test_path)
        view_hist_data(x_train, y_train, x_valid, y_valid, x_test, y_test)
    """
    plt.figure(figsize=(24, 8))

    train_hist, _ = np.histogram(y_train, bins = 7, range = (0, 7))
    valid_hist, _ = np.histogram(y_valid, bins = 7, range = (0, 7))
    test_hist, _  = np.histogram(y_test, bins = 7, range = (0, 7))

    print("Training images: ", len(x_train), " - shape: ", x_train[0].shape)
    print("Validating images: ", len(x_valid), " - shape: ", x_valid[0].shape)
    print("Testing images: ", len(x_test), " - shape: ", x_test[0].shape)

    plt.subplot(1,3,1)
    plt.bar(range(7), train_hist, tick_label  = emotion_labels)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=45)
    plt.xlabel("Emotion", fontsize = 16)
    plt.ylabel("Number of images", fontsize = 16)
    plt.title("Training Distribution (%d images)\n"%(len(x_train)), fontsize = 20)

    plt.subplot(1,3,2)
    plt.bar(range(7), valid_hist, tick_label  = emotion_labels)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=45)
    plt.xlabel("Emotion", fontsize = 16)
    plt.ylabel("Number of images", fontsize = 16)
    plt.title("Validating Distribution (%d images)\n"%(len(x_valid)), fontsize = 20)

    plt.subplot(1,3,3)
    plt.bar(range(7), test_hist, tick_label  = emotion_labels)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14, rotation=45)
    plt.xlabel("Emotion", fontsize = 16)
    plt.ylabel("Number of images", fontsize = 16)
    plt.title("Testing Distribution (%d images)\n"%(len(x_test)), fontsize = 20)
    
    # save figure
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "" and os.path.exists(save_dir)==False: os.makedirs(save_dir)
        plt.savefig(save_path)
    # if
    
    plt.show()
    plt.close()
# view_hist_data

def view_images(data, labels = None, view_ids = list(range(16)), rows = 4, cols = 4, figsize = (8,8), save_path = None):
    """
    View sample images in data with labels
    Usage:
        train_path = os.path.join(data_dir, "preprocessing", "train.npz")
        valid_path = os.path.join(data_dir, "preprocessing", "valid.npz")
        test_path = os.path.join(data_dir, "preprocessing", "test.npz")
        save_path = os.path.join(data_dir, "preprocessing", "distribution_data.png")
        
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(train_path, valid_path, test_path)
        save_path = os.path.join(data_dir, "preprocessing", "sample_training_images.png")
        
        print("Sample training images")
        view_images(x_train, y_train, list(range(16)), rows = 4, cols = 4, figsize = (8, 8), 
                    save_path = os.path.join(data_dir, "preprocessing", "sample_training_images.png"))
    """
    if view_ids is None or len(view_ids)==0:
        view_ids = np.random.choice(len(data), size = rows * cols)
    # if

    plt.figure(figsize=figsize)
    for row in range(rows):
        for col in range(cols):
            id_pos = row * cols + col
            if id_pos >= len(view_ids): continue
            view_data  =  data[view_ids[id_pos]]
            view_label = labels[view_ids[id_pos]] if labels is not None else None

            plt.subplot(rows, cols, row * cols + col + 1)
            plt.imshow(view_data, cmap='gray')
            if labels is not None:
                plt.title(emotion_labels[view_label])
            plt.axis("off")
        # for
    # for

    # save figure
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "" and os.path.exists(save_dir)==False: os.makedirs(save_dir)
        plt.savefig(save_path)
    # if

    plt.show()
    plt.close()
# view_images

""" *****************************************
UTILS
***************************************** """ 

def check_tensorflow_environment():
    from distutils.version import LooseVersion
    import warnings
    import tensorflow as tf
    from tensorflow.python.client import device_lib

    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# check_gpu

def choose_keras_environment(gpus = ["0"], keras_backend = "tensorflow", verbose = 1): # gpus = ["-1"], ["0", "1"]
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(gpus) # run GPU 0
    # os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"        # run GPU 0,1
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"          # don't run GPU
    os.environ['KERAS_BACKEND'] = keras_backend
    if verbose == 1:
        print("Environment GPUs:")
        print("+ Choose GPUs: ", ",".join(gpus))
        print("+ Keras backend: ", keras_backend)
    # if
# choose_keras_envs

# init_session
def init_session():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
# init_session

def visualize_logs(visualize_data, group_data, figsize = (10, 5), is_show = True, label_size = 12, title_size = 16, save_path = None):
    """
    visualize_data = df_logs[50:].to_dict(orient='list')
    group_data     = [{"x": "epoch", "y": ["mean_squared_error", "val_mean_squared_error"], 
                       "title": "Error at Epoch {epoch}: {val_mean_squared_error:.4f}\n", 
                       "style": "min"},
                      {"x": "epoch", "y": ["loss", "val_loss"], 
                       "title": "Loss at Epoch {epoch}: {val_loss:.4f}\n", 
                       "style": "min"}]
    figsize        = (10, 5)
    visualize_logs(visualize_data, group_data, figsize)
    """
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt, numpy as np
    
    fig = plt.figure(figsize=figsize)
    for idx, group in enumerate(group_data):
        x_axis_name, y_axis_names, title, style = group["x"], group["y"], group["title"], group["style"]
        x_values = visualize_data[x_axis_name]
        end_logs = dict([(x_axis_name, x_values[-1])])

        ax = plt.subplot(1, len(group_data), idx%len(group_data) + 1)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        for y_axis_name in y_axis_names:
            y_values = visualize_data[y_axis_name]
            if style is not None and style != "":
                y_values = []
                for i in range(len(visualize_data[y_axis_name])):
                    if style == "median":
                        y_values.append(np.median(visualize_data[y_axis_name][:i+1]))
                    elif style == "avg":
                        y_values.append(np.average(visualize_data[y_axis_name][:i+1]))
                    elif style == "max":
                        y_values.append(np.max(visualize_data[y_axis_name][:i+1]))
                    elif style == "min":
                        y_values.append(np.min(visualize_data[y_axis_name][:i+1]))
            # if
            end_logs.update(dict([(y_axis_name, y_values[-1])]))
            ax.plot(x_values, y_values)
        pass
        title = title.format(**end_logs)

        ax.set_xlabel(x_axis_name, fontsize = label_size)
        ax.legend(y_axis_names, loc='upper left', fontsize = label_size)
        ax.set_title(title, fontsize = title_size)
    # for
    
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "" and os.path.exists(save_dir) == False: os.makedirs(save_dir)
        fig.savefig(save_path)
    # if
    
    if is_show == True: 
        plt.show()
        return None
    return fig
# visualize_logs

""" *****************************************
LEARNING RATE SCHEDULE
***************************************** """
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    # https://github.com/bckenstler/CLR/blob/master/clr_callback.py
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
# CyclicLR

""" *****************************************
VALIDATION
***************************************** """ 
def plot_confusion_matrix(y_test, y_pred, classes=None,
                          normalize=True,
                          title='Average accuracy \n',
                          cmap=plt.cm.Blues,
                          verbose=0, precision=0,
                          text_size=10,
                          title_size=25,
                          axis_label_size=16,
                          tick_size=14, save_path=None, 
                          has_colorbar = False):
    """
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    import itertools

    cm = confusion_matrix(y_test, y_pred)
    acc = sum(cm.diagonal() / cm.sum()) * 100.0
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100.0
        if verbose == 1: print("Normalized confusion matrix")
    else:
        if verbose == 1: print('Confusion matrix, without normalization')

    if verbose == 1: print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.format_map({'acc': acc}), fontsize=title_size)
    if has_colorbar == True: plt.colorbar()

    if classes is not None:
        ax = plt.gca()
        # tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45, fontsize=tick_size)
        # plt.yticks(tick_marks, classes, fontsize=tick_size)
        ax.set_xticklabels(classes, fontsize=tick_size, rotation=45)
        ax.set_yticklabels(classes, fontsize=tick_size)

    fmt = '{:.' + '%d' % (precision) + 'f} %' if normalize else '{:d} %'
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=text_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_label_size)
    plt.xlabel('Predicted label', fontsize=axis_label_size)

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname != "" and os.path.exists(dirname) == False: os.makedirs(dirname)
        plt.savefig(save_path)
# plot_confusion_matrix

""" *****************************************
VISUALIZATION
***************************************** """ 
def show_heatmap(model, image, layer_name = "conv2d_50"):
    images = image.reshape((1,) + image.shape)
    preds  = model.predict(images)

    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer_name)

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function(inputs=[model.input], outputs=[pooled_grads, last_conv_layer.output])
    pooled_grads_value, conv_layer_output_value = iterate([images])

    
    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[0, :, :, i] *= pooled_grads_value[i]

    conv_layer_output_value = conv_layer_output_value.reshape(conv_layer_output_value.shape[1:])
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    scale   = np.ptp(heatmap)
    if scale == 0: scale = 1
    heatmap = (heatmap - np.min(heatmap)) / scale
    
    scale   = np.ptp(image)
    if scale == 0: scale = 1
    image = (image - np.min(image)) / scale

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap[heatmap<0.3] = 0
    output = image * 0.4 + np.dstack([np.zeros_like(heatmap), np.zeros_like(heatmap), heatmap]) * 0.6
    
    scale   = np.ptp(output)
    if scale == 0: scale = 1
    output = (output - np.min(output)) / scale

    plt.imshow(output)
    plt.axis("off")
# show_heatmap

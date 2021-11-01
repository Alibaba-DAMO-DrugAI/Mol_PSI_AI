import os
import sys
import keras
import numpy as np
import cv2

from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras import backend
from keras.layers import Add, AveragePooling2D, Flatten, Dense, Input, ZeroPadding2D, MaxPooling2D, Activation, Conv2D, BatchNormalization
from keras.initializers import glorot_uniform
from keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array

from scipy.stats import pearsonr
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    # First component of main path
    X = Conv2D(filters=F1,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='valid',
               name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='valid',
               name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(filters=F1,
               kernel_size=(1, 1),
               strides=(s, s),
               padding='valid',
               name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    # Second component of main path
    X = Conv2D(filters=F2,
               kernel_size=(f, f),
               strides=(1, 1),
               padding='same',
               name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    # Third component of main path
    X = Conv2D(filters=F3,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='valid',
               name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters=F3,
                        kernel_size=(1, 1),
                        strides=(s, s),
                        padding='valid',
                        name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,
                                    name=bn_name_base + '1')(X_shortcut)
    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def ResNet50(input_shape=(100, 100, 50)):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    # Stage 1
    X = Conv2D(64, (7, 7),
               strides=(2, 2),
               name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    # Stage 2
    X = convolutional_block(X,
                            f=3,
                            filters=[64, 64, 256],
                            stage=2,
                            block='a',
                            s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    # Stage 3
    X = convolutional_block(X,
                            f=3,
                            filters=[128, 128, 512],
                            stage=3,
                            block='a',
                            s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    # Stage 4
    X = convolutional_block(X,
                            f=3,
                            filters=[256, 256, 1024],
                            stage=4,
                            block='a',
                            s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    # Stage 5
    X = convolutional_block(X,
                            f=3,
                            filters=[512, 512, 2048],
                            stage=5,
                            block='a',
                            s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)
    # output layer
    X = Flatten()(X)
    X = Dense(1, kernel_initializer=glorot_uniform(seed=0))(X)
    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    return model


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


class pcc_callback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)[:, 0]
        pcc = pearsonr(self.y, y_pred)[0]

        y_pred_val = self.model.predict(self.x_val)[:, 0]
        pcc_val = pearsonr(self.y_val, y_pred_val)[0]

        print("\npcc\t", pcc, "pcc_val\t", pcc_val)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# imgs with mutiple channels
y_train = np.load("PDB_data/y_train_2016.npy")
y_test = np.load("PDB_data/y_test_2016.npy")

X_train = np.transpose(
    np.load("PDB_data/2016_norm_img/X_train_test.npy")[:y_train.shape[0]],
    (0, 2, 3, 1))
X_test = np.transpose(
    np.load("PDB_data/2016_norm_img/X_train_test.npy")[-y_test.shape[0]:],
    (0, 2, 3, 1))


# imgs with single channel
train_2007 = [line.strip() for line in open('PDB_data/train_2007.txt')]
test_2007 = [line.strip() for line in open('PDB_data/core_2007.txt')]

# ref_2013 =
# ref_2016 =

def tensor_loader(src_dir, pdb_id):
    sorted_dir = sorted([file for file in os.listdir(src_dir) if file.startswith(pdb_id)],
                        key=lambda x: int(x.strip('.png').split('combo')[1]))
    # print(sorted_dir)
    img_list = [cv2.cvtColor(cv2.imread(src_dir + '/' + file),cv2.COLOR_BGR2GRAY)
        for file in sorted_dir]
    X_combo = np.array(img_list).squeeze()
    return X_combo

def data_loader(scr_dir, ref_list):
    tensor_list = [tensor_loader(scr_dir, pdb_id) for pdb_id in tqdm(ref_list)]
    X = np.array(tensor_list)
    return X

src_dir = "/mnt/med/Temp_img/2007_elecDist_sigma_0.5"

X_train = data_loader(src_dir, train_2007)
X_test = data_loader(src_dir, test_2007)
# np.save('Temp_img/2007_tensors/2007_elecDist_sigma_0.5_X_train.npy', X_train)
# np.save('Temp_img/2007_tensors/2007_elecDist_sigma_0.5_X_test.npy', X_test)

y_train = np.load("PDB_data/y_train_2007.npy")
y_test = np.load("PDB_data/y_test_2007.npy")

##############
model = ResNet50(input_shape=(100, 100, 50))
model.compile(
    loss=rmse,
    optimizer="adam",
)  # metrics=[rmse]
#optimizer = keras.optimizers.Adam(lr=0.01, )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=5,
                              min_delta=0.0001,
                              min_lr=1e-8)

tensorboard = TensorBoard(log_dir='log')
checkpoint = ModelCheckpoint(
    filepath='models/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    monitor='val_loss',
    mode='auto',
    save_weights_only=True,
    save_best_only='True')
pcc_monitor = pcc_callback(training_data=[X_train, y_train],
                           validation_data=[X_test, y_test]),

callback_lists = [
    tensorboard,
    checkpoint,
    reduce_lr,
    pcc_monitor,
]

model.fit(X_train,
          y_train,
          batch_size=350,
          epochs=200,
          callbacks=callback_lists,
          validation_data=(X_test, y_test),
          shuffle=True)  #verbose = 2 # callback_lists

# y_predict = model.predict(X_test)
# print(y_predict[:, 0], y_test)
# pcc = np.corrcoef(y_predict[:, 0], y_test)
# print("2016 norm test pearson coefficient", pcc)

########### plot and utils ###################################

model_best = ResNet50(input_shape=(100, 100, 36))
model_best.load_weights("models/weights.10-1.22.hdf5")
y_best = model_best.predict(X_test)[:, 0]

import seaborn as sns
from matplotlib import pyplot as plt

font = {
    'family': 'arial',
    'weight': 'bold',
    'style': 'normal',
    'size': 16,
}  #italic bold italic

fig = plt.figure(figsize=(10, 10))
g = sns.jointplot(-y_test,
                  -y_best,
                  kind="reg",
                  color="green",
                  xlim=[-12, -2],
                  ylim=[-12, -2],
                  marginal_ticks=False)  #绘制散点图
g.ax_joint.plot([-12, -2], [-12, -2], ls="--", c=".3")
plt.grid(linestyle='--')
plt.xlabel("Experimental binding affinity (kcal/mol)", font)
plt.ylabel("Predicted binding affinity (kcal/mol)", font)
plt.text(-11, -3.5, "PDBBind-2016", size=18)
plt.text(-11, -4.5, "Pcc=0.744", size=14, style='italic')
plt.text(-11, -5, "RMSEs=1.22", size=14, style='italic')
plt.tight_layout()
#plt.savefig("1.pdf")
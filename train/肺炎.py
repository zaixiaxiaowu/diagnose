import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import os
import keras
import numpy as np
import sklearn.model_selection
from keras import layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

dir = './chest_xray'
test_dir = os.path.join(dir,'test')
train_dir =os.path.join(dir,'train')
val_dir =os.path.join(dir,'val')

categ = ['NORMAL', 'PNEUMONIA']
for ca in categ:
    path = os.path.join(train_dir,ca)
    for img in os.listdir(path):
        img_arr =Image.open(os.path.join(path,img))
        arr = np.array(img_arr)
        plt.imshow(np.array(img_arr),cmap='gray')
        plt.show()
        break

img_size = 80
new_img = img_arr.resize((img_size,img_size))
plt.imshow(new_img,cmap='gray')
plt.show()

for ca in categ:
    path = os.path.join(test_dir,ca)
    for img in os.listdir(path):
        img_arr =Image.open(os.path.join(path,img))
        print(np.array(img_arr).shape)
        plt.imshow(img_arr,cmap='gray')
        plt.show()
        break
    break

def creat_data(my_list,categ,my_dir):
    img_size = 150
    for ca in categ:
        path = os.path.join(my_dir,ca)
        class_num = categ.index(ca)
        for img in os.listdir(path):
            try:
                img_arr =Image.open(os.path.join(path,img))
                new_img = img_arr.resize((img_size,img_size))
                new_img = np.asarray(new_img)
                arr = new_img.reshape((img_size, img_size, 1))
                my_list.append([arr,class_num])
            except Exception as e:
                e = e

train_list = []
test_list = []
val_list = []
creat_data(train_list,categ,train_dir)
print(len(train_list))

creat_data(test_list,categ,test_dir)
print(len(test_list))

creat_data(val_list,categ,val_dir)
print(len(val_list))

for i in range(608):
    ele = train_list.pop(0)
    val_list.append(ele)

print(len(train_list))
print(len(val_list))

def split_data(X,y,my_list):
    img_size = 150
    for fe,la in my_list:
        X.append(fe)
        y.append(la)
    X = np.array(X).reshape(-1, img_size, img_size, 1)

X_train = []
y_train = []
X_test = []
y_test = []
X_val = []
y_val = []
split_data(X_train, y_train, train_list)

split_data(X_test, y_test, test_list)
split_data(X_val, y_val, val_list)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

X_train = X_train/255.0
X_test = X_test/255.0
X_val = X_val/255.0


early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)

print(X_train.shape)



inputs = keras.Input(shape=(150, 150, 1))

# data augmentation
x = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')(inputs)
x = tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.20, width_factor=0.20)(x)
x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.09)(x)


def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation('relu')(x)
    return x

# model itself
x = keras.layers.Conv2D(filters=64, kernel_size=3, input_shape=(150, 150, 1), padding='same', activation="relu")(x)
x = keras.layers.Conv2D(filters=32, kernel_size=3, input_shape=(64, 64, 1), padding='same', activation="relu")(x)
num_res_net_blocks = 18
for i in range(num_res_net_blocks):
    x = res_net_block(x, 32, 5)
x = layers.Dropout(0.5)(x)
for i in range(num_res_net_blocks):
    x = res_net_block(x, 32, 5)
x = layers.Conv2D(25, 3, activation='relu')(x)
x = layers.GlobalMaxPooling2D()(x)
x = keras.layers.Dense(256, activation='sigmoid')(x)
x = keras.layers.Dense(32, activation='sigmoid')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test,y_test), callbacks=[early_stop])

model.evaluate(X_val,y_val)

model_file = "top_model_resnet_2.h5"
model.save(model_file)
print("Model saved!")
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D,
                                     MaxPool2D, Concatenate, Dropout)


def U_net(input_shape, classes, dropout=0.2):
    
    x = inputs = Input(shape = input_shape)
    
    # encoder part:
    x = Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = x1 = Conv2D(32, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = MaxPool2D(pool_size=2)(x)
    
    x = Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = x2 = Conv2D(64, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = MaxPool2D(pool_size=2)(x)
    
    x = Conv2D(128, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = x3 = Conv2D(128, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = MaxPool2D(pool_size=2)(x)
    
    x = Conv2D(256, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = x4 = Conv2D(256, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = MaxPool2D(pool_size=2)(x)
    
    
    x = Conv2D(512, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(512, kernel_size = 3, activation = 'relu', padding = 'same')(x)
    x = Dropout(dropout)(x)

    
    # decoder part:
    x = Conv2D(256, kernel_size = 2, activation = 'relu', padding = 'same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x4,x])
    x = Conv2D(256, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same')(x)

    x = Conv2D(128, kernel_size = 2, activation = 'relu', padding = 'same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x3,x])
    x = Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same')(x)

    x = Conv2D(64, kernel_size = 2, activation = 'relu', padding = 'same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x2,x])
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    
    x = Conv2D(32, kernel_size = 2, activation = 'relu', padding = 'same')(UpSampling2D(size=2)(x))
    x = Concatenate(axis=-1)([x1,x])
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(x)
    
    x = outputs = Conv2D(classes, kernel_size=1, padding='same', activation = 'softmax')(x)
    
    return tf.keras.Model(inputs = inputs, outputs = outputs)


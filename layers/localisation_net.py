import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D

class LocalisationNet(tf.keras.layers.Layer):
    def __init__(self):
        super(LocalisationNet, self).__init__()
        self.conv_1 = Conv2D(8, kernel_size=7, activation="relu", kernel_initializer="he_normal")
        self.maxpool_1 = MaxPool2D(strides=2)
        self.conv_2 = Conv2D(10, kernel_size=5, activation="relu", kernel_initializer="he_normal")
        self.maxpool_2 = MaxPool2D(strides=2)
        
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.maxpool_1(x)
        x = self.conv_2(x)
        x = self.maxpool_2(x)
        return x
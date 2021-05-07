import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class AffineDense(tf.keras.layers.Layer):
    def __init__(self):
        super(AffineDense, self).__init__()
        self.dense_1 = Dense(32, activation="relu", kernel_initializer="he_normal")
        identity_transform_params = tf.keras.initializers.Constant([1, 0, 0, 0, 1, 0])
        self.dense_2 = Dense(3 * 2, kernel_initializer="zeros", bias_initializer=identity_transform_params)
        
    def call(self, inputs):
        x = tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
        x = self.dense_1(x)
        theta = self.dense_2(x)
        theta = tf.reshape(theta, (-1, 2, 3))
        return theta
import tensorflow as tf
from layers.localisation_net import LocalisationNet
from layers.affine_dense import AffineDense
from layers.sampling_grid import SamplingGrid
from layers.bilinear_sampler import BilinearSampler

class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialTransformer, self).__init__()
        self.loc_net = LocalisationNet()
        self.affine_dense = AffineDense()
        self.sampler = BilinearSampler()
        
    def call(self, inputs):
        x = self.loc_net(inputs)
        theta = self.affine_dense(x)
        grid = SamplingGrid(inputs.shape[1], inputs.shape[2])(theta)
        x = self.sampler([inputs, grid])
        return x
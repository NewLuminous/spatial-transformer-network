import tensorflow as tf

class SamplingGrid(tf.keras.layers.Layer):
    def __init__(self, height, width):
        super(SamplingGrid, self).__init__()
        self.height = height
        self.width = width
        
        # create normalized 2D grid
        x = tf.linspace(-1.0, 1.0, width)
        y = tf.linspace(-1.0, 1.0, height)
        x_t, y_t = tf.meshgrid(x, y)
        
        # flatten
        x_t_flat = tf.reshape(x_t, [-1])
        y_t_flat = tf.reshape(y_t, [-1])
        
        # reshape to [x_t, y_t, 1] - (homogeneous form)
        ones = tf.ones_like(x_t_flat)
        self.sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])
        
        # repeat grid num_batch times
        self.sampling_grid = tf.expand_dims(self.sampling_grid, axis=0)
        
        # cast to float32 (required for matmul)
        self.sampling_grid = tf.cast(self.sampling_grid, 'float32')
        
    def call(self, theta): 
        # cast to float32 (required for matmul)
        theta = tf.cast(theta, 'float32')
        
        # transform the sampling grid - batch multiply
        batch_grids = tf.matmul(theta, self.sampling_grid)
        # batch grid has shape (num_batch, 2, H*W)
        
        # reshape to (num_batch, H, W, 2)
        batch_grids = tf.reshape(batch_grids, (-1, 2, self.height, self.width))

        return batch_grids
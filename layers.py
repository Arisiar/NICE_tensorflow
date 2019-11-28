import tensorflow as tf
import numpy as np

class NICEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(NICEBlock, self).__init__()
        self.dense = tf.keras.Sequential([
            tf.keras.Sequential([tf.keras.layers.Dense(1000, activation='relu') for _ in range(5)]),
            tf.keras.Sequential([tf.keras.layers.Dense(392, activation='relu')])
        ])
        self.shuffle = ShufflingLayer()
        self.concat = CombiningLayer()
        self.couple = CouplingLayer()
        self.split = SplitingLayer()

    def call(self, inputs):
        x = inputs
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        x2 = x2 + self.dense(x1)
        x = self.concat([x1, x2])
        return x

    def inverse(self, inputs):
        x = inputs
        x1, x2 = self.concat.inverse()(x)
        x2 = x2 - self.dense(x1)
        x = self.split.inverse()([x1, x2])
        x = self.shuffle(x)
        return x

class ShufflingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ShufflingLayer, self).__init__()

    def build(self, input_shape):
        self.idxs = list(range(input_shape[-1]))[::-1]

    def call(self, inputs):            
        inputs = tf.transpose(inputs)
        outputs = tf.gather(inputs, self.idxs)
        outputs = tf.transpose(outputs)
        return outputs

class CouplingLayer(tf.keras.layers.Layer):
    def __init__(self, is_inverse = False):
        super(CouplingLayer, self).__init__()
        self.is_inverse = is_inverse

    def call(self, inputs):
        x, y = inputs
        return (x - y if self.is_inverse else x + y)

    def inverse(self):
        return CouplingLayer(is_inverse = True)

class SplitingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitingLayer, self).__init__()

    def call(self, inputs):
        dim = int(inputs.shape[-1] / 2)
        inputs = tf.reshape(inputs, (-1, dim, 2))
        return [inputs[:, :, 0], inputs[:, :, 1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
    def inverse(self):
        return CombiningLayer()

class CombiningLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CombiningLayer, self).__init__()
    def call(self, inputs):
        inputs = [tf.expand_dims(i, 2) for i in inputs]
        inputs = tf.concat(inputs, 2)
        return tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        return SplitingLayer()
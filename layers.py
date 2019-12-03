import tensorflow as tf
import numpy as np

class NICEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(NICEBlock, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.Sequential([tf.keras.layers.Dense(1000, activation='relu') for _ in range(5)]),
            tf.keras.Sequential([tf.keras.layers.Dense(392, activation='relu')])
        ])

        self.shuffle = ShufflingLayer()
        self.split = SplitingLayer()
        self.concat = CombiningLayer()
        self.couple = CouplingLayer()

    def call(self, inputs):
        x = inputs
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.model(x1)
        x2 = self.couple([x2, mx1])
        x = self.concat([x1, x2])
        return x

    def inverse(self):
        x_in = tf.keras.Input(shape = (784,))
        x1, x2 = self.split(x_in)
        mx1 = self.model(x1)
        x2 = self.couple.inverse()([x2, mx1])
        x = self.concat([x1, x2])
        x = self.shuffle(x)
        return tf.keras.Model(x_in, x)        

class CouplingLayer(tf.keras.layers.Layer):
    def __init__(self, is_inverse = False):
        super(CouplingLayer, self).__init__()
        self.is_inverse = is_inverse

    def call(self, inputs):
        x, y = inputs
        return (x + y if self.is_inverse else x - y)

    def inverse(self):
        return CouplingLayer(is_inverse = True)

class ShufflingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ShufflingLayer, self).__init__()

    def build(self, input_shape):
        self.idxs = list(range(input_shape[-1]))[::-1]

    def call(self, inputs):            
        x = tf.transpose(inputs)
        x = tf.gather(x, self.idxs)
        x = tf.transpose(x)
        return x

class SplitingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitingLayer, self).__init__()

    def call(self, inputs):
        dim = int(inputs.shape[-1] / 2)
        inputs = tf.reshape(inputs, (-1, dim, 2))
        return [inputs[:, :, 0], inputs[:, :, 1]]

class CombiningLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CombiningLayer, self).__init__()
    def call(self, inputs):
        inputs = [tf.expand_dims(i, 2) for i in inputs]
        inputs = tf.concat(inputs, 2)
        return tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
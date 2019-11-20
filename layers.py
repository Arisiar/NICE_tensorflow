import tensorflow as tf
import numpy as np

class NICEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(NICEBlock, self).__init__()
        self.dense = tf.keras.Sequential([
            tf.keras.Sequential([tf.keras.layers.Dense(1000, activation='relu') for _ in range(5)]),
            tf.keras.Sequential([tf.keras.layers.Dense(392, activation='relu')])
        ])

    def build(self, input_shape):
        self.idxs = list(range(input_shape[-1]))[::-1]

    def call(self, inputs):
        x = inputs
        x = tf.transpose(tf.gather(tf.transpose(x), self.idxs))
        x_1, x_2 = spilt(x)

        mx1 = self.dense(x_1)
        
        x_2 = x_2 + mx1
        x = concat([x_1, x_2])
        return x

    def inverse(self, inputs):
        x = inputs
        x_1, x_2 = spilt(x)

        mx1 = self.dense(x_1)

        x_2 = x_2 - mx1
        x = concat([x_1, x_2])
        x = tf.transpose(tf.gather(tf.transpose(x), self.idxs))
        return x

def spilt(x):
    dim = x.shape[-1]
    x = tf.reshape(x, [-1, int(dim / 2), 2])
    x_1, x_2 = x[:,:,0], x[:,:,1]
    return x_1, x_2

def concat(x):
    x = [tf.expand_dims(i, 2) for i in x]
    x = tf.concat(x, axis = 2)
    x = tf.reshape(x, (-1, np.prod(x.shape[1:])))
    return x

class Shuffle(tf.keras.layers.Layer):
    def __init__(self, idxs=None, **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs

    def call(self, inputs):
        v_dim = inputs.shape[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))[::-1]
        inputs = tf.transpose(inputs)
        outputs = tf.gather(inputs, self.idxs)
        outputs = tf.transpose(outputs)
        return outputs

    def inverse(self):
        v_dim = len(self.idxs)
        _ = sorted(zip(range(v_dim), self.idxs), key=lambda s: s[1])
        reverse_idxs = [i[0] for i in _]
        return Shuffle(reverse_idxs)

class SplitVector(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SplitVector, self).__init__(**kwargs)
    def call(self, inputs):
        v_dim = inputs.shape[-1]
        inputs = tf.reshape(inputs, (-1, v_dim//2, 2))
        return [inputs[:,:,0], inputs[:,:,1]]
    def compute_output_shape(self, input_shape):
        v_dim = input_shape[-1]
        return [(None, v_dim//2), (None, v_dim//2)]
    def inverse(self):
        layer = ConcatVector()
        return layer

class ConcatVector(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConcatVector, self).__init__(**kwargs)
    def call(self, inputs):
        inputs = [tf.expand_dims(i, 2) for i in inputs]
        inputs = tf.concat(inputs, 1)
        # return inputs
        return tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer

class AddCouple(tf.keras.layers.Layer):
    """加性耦合层
    """
    def __init__(self, isinverse=False, **kwargs):
        self.isinverse = isinverse
        super(AddCouple, self).__init__(**kwargs)
    def call(self, inputs):
        part1, part2, mpart1 = inputs
        if self.isinverse:
            return [part1, part2 + mpart1] # 逆为加
        else:
            return [part1, part2 - mpart1] # 正为减
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        layer = AddCouple(True)
        return layer
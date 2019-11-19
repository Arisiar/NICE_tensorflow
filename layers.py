import tensorflow as tf
import numpy as np

def network(hidden_dim, num_layers = 5):
    inputs = tf.keras.Input(hidden_dim)
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    return tf.keras.Model(inputs, x)

class NICEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(NICEBlock, self).__init__()
        self.network = network
    def build(self, input_shape):
        self.network = network(int(input_shape[-1] / 2))

    def call(self, inputs, is_inverse = False):

        x = tf.reshape(inputs, [-1, int(inputs.shape[-1] / 2), 2])
        x_1, x_2 = x[:,:,0], x[:,:,1]
        
        mx1 = self.network(x_1)
        
        x_2 = x_2 - mx1 if is_inverse else x_2 + mx1

        x = tf.concat([x_1, x_2], axis = 1)
    
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
        inputs = tf.concat(inputs, 2)
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
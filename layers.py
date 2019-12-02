import tensorflow as tf
import numpy as np

class Encoder(tf.keras.layers.Layer):
    def __init__(self, model):
        super(Encoder, self).__init__()
        self.model = model
        self.shuffle = Shuffle()
        self.split = SplitVector()
        self.couple = AddCouple()
        self.concat = ConcatVector()

    def call(self, inputs):
        x = inputs
        x = self.shuffle(x)
        x1, x2 = self.split(x)
        mx1 = self.model(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model
        self.shuffle = Shuffle()
        self.split = SplitVector()
        self.couple = AddCouple()
        self.concat = ConcatVector()

    def call(self, inputs):
        x = inputs
        x1, x2 = self.concat.inverse()(x)
        mx1 = self.model(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle.inverse()(x)
        return x



class Shuffle(tf.keras.layers.Layer):
    def __init__(self, idxs = None, mode = 'reverse', **kwargs):
        super(Shuffle, self).__init__(**kwargs)
        self.idxs = idxs
        self.mode = mode
    def call(self, inputs):
        v_dim = inputs.shape[-1]
        if self.idxs == None:
            self.idxs = list(range(v_dim))
            if self.mode == 'reverse':
                self.idxs = self.idxs[::-1]
            elif self.mode == 'random':
                np.random.shuffle(self.idxs)
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
        inputs = tf.concat(inputs, axis = 2)
        return tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
    def compute_output_shape(self, input_shape):
        return (None, sum([i[-1] for i in input_shape]))
    def inverse(self):
        layer = SplitVector()
        return layer

class AddCouple(tf.keras.layers.Layer):
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

class Scale(tf.keras.layers.Layer):
    """尺度变换层
    """
    def __init__(self, **kwargs):
        super(Scale, self).__init__(**kwargs)
        self.kernel = tf.Variable(tf.initializers.glorot_normal()(shape = [1, 784]), trainable = True, name = 'kernel')
    def call(self, inputs):
        self.add_loss(-tf.reduce_sum(self.kernel)) # 对数行列式
        return tf.math.exp(self.kernel) * inputs
    def inverse(self):
        scale = tf.math.exp(-self.kernel)
        return tf.keras.layers.Lambda(lambda x: scale * x)



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

class CombiningLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(CombiningLayer, self).__init__()
    def call(self, inputs):
        inputs = [tf.expand_dims(i, 2) for i in inputs]
        inputs = tf.concat(inputs, 2)
        return tf.reshape(inputs, (-1, np.prod(inputs.shape[1:])))
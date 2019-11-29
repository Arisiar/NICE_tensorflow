import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock

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
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(1, input_shape[1]),
                                      initializer='glorot_normal',
                                      trainable=True)
    def call(self, inputs):
        self.add_loss(-tf.reduce_sum(self.kernel)) # 对数行列式
        return tf.math.exp(self.kernel) * inputs
    def inverse(self):
        scale = tf.math.exp(-self.kernel)
        return tf.keras.layers.Lambda(lambda x: scale * x)

def build_basic_model(v_dim):
    _in = tf.keras.Input(shape=(v_dim,))
    _ = _in
    for i in range(5):
        _ = tf.keras.layers.Dense(1000, activation='relu')(_)
    _ = tf.keras.layers.Dense(v_dim, activation='relu')(_)
    return tf.keras.Model(_in, _)

shuffle1 = Shuffle()
shuffle2 = Shuffle()
shuffle3 = Shuffle()
shuffle4 = Shuffle()

split = SplitVector()
couple = AddCouple()
concat = ConcatVector()
scale = Scale()

basic_model_1 = build_basic_model(392)
basic_model_2 = build_basic_model(392)
basic_model_3 = build_basic_model(392)
basic_model_4 = build_basic_model(392)

def keras_model():
    x_in = tf.keras.Input(shape=(784,))
    x = x_in

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = scale(x)

    return tf.keras.Model(x_in, x)
class NICEModel(tf.keras.Model):
    def __init__(self, inverse = False, **kwargs):
        super(NICEModel, self).__init__(**kwargs)
        self.inverse = inverse

        self.block_1 = NICEBlock()
        self.block_2 = NICEBlock()
        self.block_3 = NICEBlock()
        self.block_4 = NICEBlock()

        self.diag = tf.Variable(tf.initializers.glorot_normal()(shape = [1, 784]), trainable = True)

    def call(self, inputs):
        x = inputs
        if self.inverse:
            x = x * tf.exp(-self.diag)
            x = self.block_4.inverse(x)
            x = self.block_3.inverse(x)
            x = self.block_2.inverse(x)
            x = self.block_1.inverse(x)
        else:
            x = self.block_1(x)
            x = self.block_2(x)
            x = self.block_3(x)
            x = self.block_4(x)
            x = x * tf.exp(self.diag)
            self.add_loss(-tf.reduce_sum(self.diag))
        return x
    
    def get_diag(self):
        return tf.reduce_mean(self.diag)

def logistic_loss(x):
    # x = tf.clip_by_value(x, 1e-10, 1.0)
    # return tf.reduce_sum(-tf.math.log1p(tf.exp(x)) - tf.math.log1p(tf.exp(-x)), axis = 1)
    return tf.reduce_sum((0.5 * x ** 2), axis = 1)

def model(args, dataset, dim):
    
    # nice = NICEModel()
    x_in = tf.keras.Input(shape=(784,))
    x = x_in

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = shuffle1(x)
    x1,x2 = split(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple([x1, x2, mx1])
    x = concat([x1, x2])

    x = scale(x)

    nice = tf.keras.Model(x_in, x)
    
    opt = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = opt, net = nice)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep = 3)
    
    @tf.function
    def train_step(net, inputs, optimizer):
        with tf.GradientTape() as tape:
            predictions = net(images)
            loss = tf.reduce_mean(logistic_loss(predictions))
        grads = tape.gradient(loss, nice.trainable_weights)
        optimizer.apply_gradients(zip(grads, nice.trainable_weights))
        return loss

    for epoch in range(args.epochs):
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for images in dataset:
            
            loss = train_step(nice, images, opt)
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 100 == 0:
                save_path = manager.save()
                print("loss: {:1.2f}".format(loss.numpy()))

    # ckpt.restore(manager.latest_checkpoint) 
    x = x_in
    x = scale.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_4(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle1.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_3(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle1.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_2(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle1.inverse()(x)

    x1,x2 = concat.inverse()(x)
    mx1 = basic_model_1(x1)
    x1, x2 = couple.inverse()([x1, x2, mx1])
    x = split.inverse()([x1, x2])
    x = shuffle1.inverse()(x)

    decoder = tf.keras.Model(x_in, x)
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))

    for i in range(n):
        for j in range(n):
            z_sample = np.array(np.random.randn(1, 784)) * 0.75 # 标准差取0.75而不是1
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit


    figure = np.clip(figure*255, 0, 255)
    imageio.imwrite('test.png', figure)

    # nice.inverse = True

    # sample = np.array(np.random.randn(1, dim), dtype = np.float32) * 0.75
    # figure = nice(sample)
    # figure = np.clip(figure * 255, 0, 255)
    # imageio.imwrite('test.png', np.reshape(figure, [28, 28]))
    # imageio.imwrite('noise.png', np.reshape(sample, [28, 28]))

        # n = 10
        # digit_size = 28
        # figure = np.zeros((digit_size * n, digit_size * n))

        # for i in range(n):
        #     for j in range(n):
        #         sample = np.array(np.random.randn(1, dim), dtype = np.float32) * 0.75
        #         x_decoded = nice(images).numpy()
        #         digit = x_decoded[0].reshape(digit_size, digit_size)
        #         figure[i * digit_size: (i + 1) * digit_size,
        #             j * digit_size: (j + 1) * digit_size] = digit

        # figure = np.clip(figure * 255, 0, 255)
        # imageio.imwrite('test.png', figure)





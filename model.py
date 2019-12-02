import tensorflow as tf
import numpy as np
import imageio
import os
from layers import Encoder, Decoder

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
    # return tf.keras.Sequential([
    #         tf.keras.Sequential([tf.keras.layers.Dense(1000, activation='relu') for _ in range(5)]),
    #         tf.keras.Sequential([tf.keras.layers.Dense(392, activation='relu')])
    #     ])

# shuffle = Shuffle()
# split = SplitVector()
# couple = AddCouple()
# concat = ConcatVector()
# scale = Scale()

# basic_model_1 = build_basic_model(392)
# basic_model_2 = build_basic_model(392)
# basic_model_3 = build_basic_model(392)
# basic_model_4 = build_basic_model(392)

class NICEModel(tf.keras.Model):
    def __init__(self):
        super(NICEModel, self).__init__()
        # self.kernel = tf.Variable(tf.initializers.glorot_normal()(shape = [1, 784]), trainable = True, name = 'kernel')
        # self.encoder_1 = Encoder(basic_model_1)
        # self.encoder_2 = Encoder(basic_model_2)
        # self.encoder_3 = Encoder(basic_model_3)
        # self.encoder_4 = Encoder(basic_model_4)

        # self.decoder_1 = Encoder(basic_model_4)
        # self.decoder_2 = Encoder(basic_model_3)
        # self.decoder_3 = Encoder(basic_model_2)
        # self.decoder_4 = Encoder(basic_model_1)
        
        self.diag = tf.Variable(tf.initializers.glorot_normal()(shape = [1, 784]), trainable = True)

        self.shuffle1 = Shuffle()
        self.shuffle2 = Shuffle()
        self.shuffle3 = Shuffle()
        self.shuffle4 = Shuffle()

        self.split = SplitVector()
        self.couple = AddCouple()
        self.concat = ConcatVector()
        self.scale = Scale()

        self.basic_model_1 = build_basic_model(392)
        self.basic_model_2 = build_basic_model(392)
        self.basic_model_3 = build_basic_model(392)
        self.basic_model_4 = build_basic_model(392)

    def call(self, inputs):
        x = inputs
        x = self.shuffle1(x)
        x1,x2 = self.split(x)
        mx1 = self.basic_model_1(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])

        x = self.shuffle2(x)
        x1,x2 = self.split(x)
        mx1 = self.basic_model_2(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])

        x = self.shuffle3(x)
        x1,x2 = self.split(x)
        mx1 = self.basic_model_3(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])

        x = self.shuffle4(x)
        x1,x2 = self.split(x)
        mx1 = self.basic_model_4(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])
        # x = self.scale(x)
        
        # x = self.encoder_1(x)
        # x = self.encoder_2(x)
        # x = self.encoder_3(x)
        # x = self.encoder_4(x)
        # self.add_loss(-tf.reduce_sum(self.diag))
        x = tf.exp(self.diag) * x  
        return x
    def inverse(self):
        x_in = tf.keras.Input(shape = (784,))
        x = tf.keras.layers.Lambda(lambda s: tf.exp(-self.diag) * s)(x)
        # x = self.scale.inverse()(x_in)
        
        x1,x2 = self.concat.inverse()(x)
        mx1 = self.basic_model_4(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle4.inverse()(x)

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.basic_model_3(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle3.inverse()(x)

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.basic_model_2(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle2.inverse()(x)

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.basic_model_1(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle1.inverse()(x)

        # x = self.decoder_1(x)
        # x = self.decoder_2(x)
        # x = self.decoder_3(x)
        # x = self.decoder_4(x)
        return tf.keras.Model(x_in, x)

def logistic_loss(x):
    # x = tf.clip_by_value(x, 1e-10, 1.0)
    # return tf.reduce_sum(-tf.math.log1p(tf.exp(x)) - tf.math.log1p(tf.exp(-x)), axis = 1)
    return tf.reduce_sum((0.5 * x ** 2), axis = 1)

def model(args, dataset, dim):
     
    nice = NICEModel()
    
    opt = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = opt, net = nice)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep = 3)
    
    @tf.function
    def train_step(net, inputs, optimizer):
        with tf.GradientTape() as tape:
            predictions = net(inputs)
            loss = tf.reduce_mean(logistic_loss(predictions)) + tf.reduce_mean(-tf.reduce_sum(net.diag))
        grads = tape.gradient(loss, net.trainable_weights)
        optimizer.apply_gradients(zip(grads, net.trainable_weights))
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
                print("loss: {:1.2f}".format(loss.numpy()))
                save_path = manager.save()

    ckpt.restore(manager.latest_checkpoint)
    decoder = nice.inverse()

    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))  
    for i in range(n):
        for j in range(n):
            z_sample = np.array(np.random.randn(1, 784)) * 0.75 # 标准差取0.75而不是1
            x_decoded = decoder(z_sample).numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    figure = np.clip(figure*255, 0, 255)
    imageio.imwrite('test.png', figure)  





import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock

class Shuffle(tf.keras.layers.Layer):
    def __init__(self, idxs=None, mode='reverse', **kwargs):
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
        inputs = tf.concat(inputs, 2)
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
            return [part1, part2 - mpart1] # 逆为加
        else:
            return [part1, part2 + mpart1] # 正为减
    def compute_output_shape(self, input_shape):
        return [input_shape[0], input_shape[1]]
    def inverse(self):
        layer = AddCouple(True)
        return layer

def build_basic_model(v_dim = 392):
    _in = tf.keras.Input(shape=(v_dim,))
    _ = _in
    for i in range(5):
        _ = tf.keras.layers.Dense(1000, activation='relu')(_)
    _ = tf.keras.layers.Dense(v_dim, activation='relu')(_)
    return tf.keras.Model(_in, _)

class NICEModel(tf.keras.Model):
    def __init__(self):
        super(NICEModel, self).__init__()
        # self.block_1 = NICEBlock()
        # self.block_2 = NICEBlock()
        # self.block_3 = NICEBlock()
        # self.block_4 = NICEBlock()
        
        self.noise_1 = tf.keras.layers.Lambda(lambda s: tf.keras.backend.in_train_phase(s - 0.01 * tf.random.uniform(tf.shape(s)), s)) 

        self.sh1 = Shuffle()
        self.sh2 = Shuffle()
        self.sh3 = Shuffle()
        self.sh4 = Shuffle()

        self.split = SplitVector()
        self.concat = ConcatVector()
        self.couple = AddCouple()

        self.m1 = build_basic_model()
        self.m2 = build_basic_model()
        self.m3 = build_basic_model()
        self.m4 = build_basic_model()

    def build(self, input_shape):
        self.diag = tf.Variable(tf.initializers.glorot_normal()(shape = [1, input_shape[1]]), trainable = True, name = 'diag')

    def call(self, inputs):
        x = inputs
        x = self.noise_1(x)
        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # x = self.block_4(x)

        x = self.sh1(x)
        x1, x2 = self.split(x)
        mx1 = self.m1(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])
    
        x = self.sh2(x)
        x1, x2 = self.split(x)
        mx1 = self.m2(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])

        x = self.sh3(x)
        x1, x2 = self.split(x)
        mx1 = self.m3(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])

        x = self.sh4(x)
        x1, x2 = self.split(x)
        mx1 = self.m4(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2]) 

        x = x * tf.exp(self.diag)
        
        return x
        
    def inverse(self, inputs):
        x = inputs
        x = x * tf.exp(-self.diag)

        x1, x2 = self.concat.inverse()(x)
        mx1 = self.m4(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.sh4.inverse()(x)

        x1, x2 = self.concat.inverse()(x)
        mx1 = self.m3(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.sh3.inverse()(x)

        x1, x2 = self.concat.inverse()(x)
        mx1 = self.m2(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.sh2.inverse()(x)

        x1, x2 = self.concat.inverse()(x)
        mx1 = self.m1(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.sh1.inverse()(x)

        # x = self.block_4.inverse(x)
        # x = self.block_3.inverse(x)
        # x = self.block_2.inverse(x)
        # x = self.block_1.inverse(x)

        return x

def logistic_loss(x, diag):
    x = tf.clip_by_value(x, 1e-10, 1.0)
    # return tf.reduce_sum(-tf.math.log1p(tf.exp(x)) - tf.math.log1p(tf.exp(-x)), axis = 1) + tf.reduce_sum(diag)
    return tf.reduce_sum(-(0.5 * x ** 2), axis = 1) + tf.reduce_sum(diag)

def model(args, dataset, dim):
    
    nice = NICEModel()
    
    opt = tf.keras.optimizers.Adam(learning_rate = args.lr,
                                beta_1 = args.beta1, beta_2 = args.beta2, epsilon = args.epsilon)
    
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = opt, net = nice)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep = 3)

    for epoch in range(args.epochs):
        print('Start of epoch %d' % (epoch))
        # ckpt.restore(manager.latest_checkpoint)
        # if manager.latest_checkpoint:
        #     print("Restored from {}".format(manager.latest_checkpoint))
        # else:
        #     print("Initializing from scratch.")
    
        for step, images in enumerate(dataset):
            with tf.GradientTape() as tape:
                pred = nice(images)
                test = nice.inverse(pred)
                # print(tf.reduce_mean(images))
                # print(tf.reduce_mean(test))
                log_loss = -logistic_loss(pred, nice.diag) 
                loss = tf.reduce_mean(log_loss)
            grads = tape.gradient(loss, nice.trainable_weights)
            opt.apply_gradients(zip(grads, nice.trainable_weights))

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 100 == 0:
                # save_path = manager.save()
                # print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss: {:1.2f}, diag: {:1.2f}, pred: {:1.2f}".format(loss, tf.reduce_sum(nice.diag), tf.reduce_mean(pred)))
                if tf.reduce_mean(pred) > 1.0:
                    exit()
                # if np.isnan(tf.reduce_mean(pred)):
                #     exit()
                n = 10
                digit_size = 28
                figure = np.zeros((digit_size * n, digit_size * n))

                for i in range(n):
                    for j in range(n):
                        sample = np.array(np.random.randn(1, dim), dtype = np.float32) * 0.75
                        x_decoded = nice.inverse(sample).numpy()
                        digit = x_decoded[0].reshape(digit_size, digit_size)
                        figure[i * digit_size: (i + 1) * digit_size,
                            j * digit_size: (j + 1) * digit_size] = digit

                figure = np.clip(figure * 255, 0, 255)
                imageio.imwrite('test.png', figure)
                





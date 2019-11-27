import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock

def network(hidden_dim = 392, num_layers = 5):
    inputs = tf.keras.Input(hidden_dim)
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    return tf.keras.Model(inputs, x)

class NICEModel(tf.keras.Model):
    def __init__(self):
        super(NICEModel, self).__init__()
        self.block_1 = NICEBlock()
        self.block_2 = NICEBlock()
        self.block_3 = NICEBlock()
        self.block_4 = NICEBlock()
        
        self.noise_1 = tf.keras.layers.Lambda(
            lambda s: tf.keras.backend.in_train_phase(s - 0.01 * tf.random.uniform(tf.shape(s)), s)) 

    def build(self, input_shape):
        self.diag = tf.Variable(tf.initializers.glorot_normal()(shape = [1, input_shape[1]]), 
                        trainable = True, name = 'diag')

    def call(self, inputs):
        x = inputs
        x = self.noise_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = x * tf.exp(self.diag)
        
        return x
        
    def inverse(self, inputs):
        x = inputs
        x = x * tf.exp(-self.diag)
        x = self.block_4.inverse(x)
        x = self.block_3.inverse(x)
        x = self.block_2.inverse(x)
        x = self.block_1.inverse(x)

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
                





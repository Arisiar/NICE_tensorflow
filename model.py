import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock, ScaleLayer

def m_loss(x, diag):
    return -(tf.reduce_sum(diag) - tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1))
    
class NICEModel(object):
    def __init__(self, dim):
        self.block_1 = NICEBlock()
        self.block_2 = NICEBlock()
        self.block_3 = NICEBlock()
        self.block_4 = NICEBlock()
        self.scale_1 = ScaleLayer()

        self.inputs = tf.keras.Input(shape = dim)

    def encoder(self):
        x = self.block_1(self.inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.scale_1(x)
        return tf.keras.Model(self.inputs, x)
        
    def decoder(self):
        x = self.scale_1.inverse()(self.inputs)
        x = self.block_4.inverse()(x)
        x = self.block_3.inverse()(x)
        x = self.block_2.inverse()(x)
        x = self.block_1.inverse()(x)
        return tf.keras.Model(self.inputs, x)

def model(args, dataset, dim):
    
    nice = NICEModel(dim)
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    encoder = nice.encoder()

    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = opt, net = encoder)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    for epoch in range(args.epochs):
        print('Start of epoch %d' % (epoch))
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        for step, images in enumerate(dataset):
            with tf.GradientTape() as tape:
                prediction = encoder(images)
                loss = tf.reduce_mean(tf.keras.backend.sum(0.5 * prediction ** 2, 1))
            
            grads = tape.gradient(loss, encoder.trainable_weights)
            opt.apply_gradients(zip(grads, encoder.trainable_weights))        
            
            ckpt.step.assign_add(1)
            if int(ckpt.step) % 100 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss: {:1.2f}".format(loss.numpy()))
            
                decoder = nice.decoder()

                n = 10
                digit_size = 28
                figure = np.zeros((digit_size * n, digit_size * n))

                for i in range(n):
                    for j in range(n):
                        z_sample = np.array(np.random.randn(1, dim)) * 0.75
                        x_decoded = decoder.predict(z_sample)
                        digit = x_decoded[0].reshape(digit_size, digit_size)
                        figure[i * digit_size: (i + 1) * digit_size,
                            j * digit_size: (j + 1) * digit_size] = digit

                figure = np.clip(figure * 255, 0, 255)
                imageio.imwrite('test.png', figure)






import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock

class NICEModel(tf.keras.Model):

    def __init__(self):
        super(NICEModel, self).__init__()
        self.block_1 = NICEBlock()
        self.block_2 = NICEBlock()
        self.block_3 = NICEBlock()
        self.block_4 = NICEBlock()

    def build(self, input_shape):
        glorot_init = tf.initializers.glorot_normal()(shape = [1, input_shape[-1]])
        self.diag = tf.Variable(glorot_init, trainable = True)

    def call(self, inputs):
        x = inputs
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        return tf.multiply(tf.exp(self.diag),x)

    def inverse(self):
        x_in = tf.keras.Input(shape = (784,))
        x = tf.multiply(tf.exp(-self.diag), x_in)
        x = self.block_4.inverse()(x)
        x = self.block_3.inverse()(x)
        x = self.block_2.inverse()(x)
        x = self.block_1.inverse()(x)
        return tf.keras.Model(x_in, x)

def logistic_loss(x):
    # x = tf.clip_by_value(x, 1e-10, 1.0)
    # return tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1)
    return tf.reduce_sum((0.5 * x ** 2), axis = 1)

def model(args, dataset):
     
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
            if int(ckpt.step) % args.save_step == 0:
                print("loss: {:1.2f}".format(loss.numpy()))
                save_path = manager.save()

        images = np.zeros((args.img_size * args.test_num, args.img_size * args.test_num))  
        for i in range(args.test_num):
            for j in range(args.test_num):
                sample = np.array(np.random.randn(1, args.img_size * args.img_size), dtype = 'float32')
                x = nice.inverse()(sample).numpy()[0]
                x = np.reshape(x, [args.img_size, args.img_size])
                images[i * args.img_size: (i + 1) * args.img_size,
                    j * args.img_size: (j + 1) * args.img_size] = x
        imageio.imwrite('test.png', np.clip(images * 255, 0, 255).astype('uint8'))  




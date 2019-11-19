import tensorflow as tf
import numpy as np
import imageio
import os
from layers import NICEBlock, Shuffle, SplitVector, ConcatVector, AddCouple

def network(hidden_dim = 392, num_layers = 5):
    inputs = tf.keras.Input(hidden_dim)
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    return tf.keras.Model(inputs, x)

class NICEModel(tf.keras.Model):
    def __init__(self, dim):
        super(NICEModel, self).__init__()
        self.block_1 = NICEBlock()
        self.block_2 = NICEBlock()
        self.block_3 = NICEBlock()
        self.block_4 = NICEBlock()
        
        self.shuffle1 = Shuffle()
        self.shuffle2 = Shuffle()
        self.shuffle3 = Shuffle()
        self.shuffle4 = Shuffle()
        
        self.split = SplitVector()
        self.couple = AddCouple()
        self.concat = ConcatVector()

        self.network1 = network()
        self.network2 = network()
        self.network3 = network()
        self.network4 = network()
        # self.noise_1 = tf.keras.layers.Lambda(
        #     lambda s: tf.keras.backend.in_train_phase(s - 0.01 * tf.random.uniform(tf.shape(s)), s)) 

    def build(self, input_shape):
        self.diag = self.add_weight(name = 'diag',
                                shape = [input_shape[-1]],
                                initializer= 'glorot_normal', 
                                trainable = True)

    def call(self, inputs):
        # x = self.noise_1(inputs)
        # x = self.block_1(x)
        # x = self.block_2(x)
        # x = self.block_3(x)
        # x = self.block_4(x)
        
        x = self.shuffle1(inputs)
        x1,x2 = self.split(x)
        mx1 = self.network1(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])    

        x = self.shuffle2(inputs)
        x1,x2 = self.split(x)
        mx1 = self.network2(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2]) 

        x = self.shuffle3(inputs)
        x1,x2 = self.split(x)
        mx1 = self.network3(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2]) 

        x = self.shuffle4(inputs)
        x1,x2 = self.split(x)
        mx1 = self.network4(x1)
        x1, x2 = self.couple([x1, x2, mx1])
        x = self.concat([x1, x2])    

        x = tf.matmul(x, tf.linalg.diag(tf.exp(self.diag)))

        return x
        
    def inverse(self, inputs):
        x = tf.matmul(inputs, tf.linalg.diag(tf.math.reciprocal(tf.exp(-self.diag))))

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.network4(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle4.inverse()(x)    

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.network3(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle3.inverse()(x)  

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.network2(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle2.inverse()(x)  

        x1,x2 = self.concat.inverse()(x)
        mx1 = self.network1(x1)
        x1, x2 = self.couple.inverse()([x1, x2, mx1])
        x = self.split.inverse()([x1, x2])
        x = self.shuffle1.inverse()(x)  

        # x = self.block_4(x, True)
        # x = self.block_3(x, True)
        # x = self.block_2(x, True)
        # x = self.block_1(x, True)

        return x

def logistic_loss(x, diag):
    return (tf.reduce_sum(diag) - (tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1)))

def model(args, dataset, dim):
    
    nice = NICEModel(dim)

    opt = tf.keras.optimizers.Adam(learning_rate = args.lr,
                                beta_1 = args.beta1, 
                                beta_2 = args.beta2, 
                                epsilon = args.epsilon)
    
    ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = opt, net = nice)
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
                prediction = nice(images)
                loss = tf.reduce_mean(-logistic_loss(prediction, nice.diag))
            
            grads = tape.gradient(loss, nice.trainable_weights)
            opt.apply_gradients(zip(grads, nice.trainable_weights))        

            ckpt.step.assign_add(1)
            if int(ckpt.step) % 100 == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print("loss: {:1.2f}".format(loss.numpy()))
            
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






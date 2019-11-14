import tensorflow as tf
import numpy as np
import os
from layers import additiveCoupleLayer, multiplicativeCoupleLayer

class NICE():
    def __init__(self, input_dim):
        self.input_dim = input_dim

        self.layer_1 = additiveCoupleLayer()
        self.layer_2 = additiveCoupleLayer() 
        self.layer_3 = additiveCoupleLayer()
        self.layer_4 = additiveCoupleLayer()
        self.diag = tf.Variable(tf.ones(input_dim), name = 'diag')

    def encoder(self, x):
        x = self.layer_1.forward(x, self.input_dim, name = 'en_f_1')
        x = self.layer_2.forward(x, self.input_dim, name = 'en_f_2')
        x = self.layer_3.forward(x, self.input_dim, name = 'en_f_3')
        x = self.layer_4.forward(x, self.input_dim, name = 'en_f_4')
        x = tf.matmul(x, tf.diag(tf.exp(self.diag)))

        return x

    def decoder(self, x):
        x = tf.matmul(x, tf.diag(tf.reciprocal(tf.exp(self.diag))))
        x = self.layer_4.inverse(x, self.input_dim, name = 'en_b_1')
        x = self.layer_3.inverse(x, self.input_dim, name = 'en_b_1')
        x = self.layer_2.inverse(x, self.input_dim, name = 'en_b_1')
        x = self.layer_1.inverse(x, self.input_dim, name = 'en_b_1')

        return x

######################  Tensorflow 2.x ######################
from tensorflow import keras
from layers import CoupleLayer, ScaleLayer

def m_loss(x, diag):
    return -(tf.reduce_sum(diag) - tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1))
    
class NICEModel(keras.models.Model):
    def __init__(self):
        super(NICEModel, self).__init__()
        self.couple_1 = CoupleLayer()
        self.couple_2 = CoupleLayer()
        self.couple_3 = CoupleLayer()
        self.couple_4 = CoupleLayer()
        self.scale = ScaleLayer()

    def call(self, inputs):
        x = self.couple_1(inputs)
        x = self.couple_2(x)
        x = self.couple_3(x)
        x = self.couple_4(x)
        x, diag = self.scale(x)
        return x, diag
        
    def inverse(self):
        x = self.scale.inverse()(inputs)
        x = self.couple_4.inverse()(x)
        x = self.couple_3.inverse()(x)
        x = self.couple_2.inverse()(x)
        x = self.couple_1.inverse()(x)
        return x

def model(args, dataset):

    nice = NICEModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    loss_metric = tf.keras.metrics.Mean()

    for epoch in range(args.epochs):
        for images, labels in dataset:
            with tf.GradientTape() as tape:
                prediction, diag = nice(images)
                loss = m_loss(prediction, diag)
            
            grads = tape.gradient(loss, nice.trainable_weights)
            optimizer.apply_gradients(zip(grads, nice.trainable_weights))        
            
            loss_metric(loss)

        print('epoch %s: loss = %s' % (epoch, loss_metric.result()))


    # model = NICE(_x.shape[-1])
    # with tf.variable_scope('encoder', reuse = reuse):
    #     x = model.encoder(_x)
    
    # loss = m_loss(x, model.diag)
    
    # var = tf.trainable_variables()
    # with tf.name_scope('optimizer'):
    #     optimizer = tf.train.AdamOptimizer(args.lr, beta1 = args.beta1, beta2 = args.beta2).minimize(loss, var_list = var)

    # saver = tf.train.Saver()


    # init = tf.compat.v1.global_variables_initializer()
    # sess.run(init)

    # for i in range(1, args.epochs):
    #     batch = (/ args.batch)
    #     for j in range():
    #     _, train_loss = sess.run([optimizer, loss])
    #     if (i % args.save_epochs) == 0:
    #         saver.save(sess, args.save_path + 'model.ckpt')
            # print("Epoch [%d / %d] [%d / %d]: Loss: %f" % (i, args.epochs, ))
    



    



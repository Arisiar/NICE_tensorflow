import tensorflow.compat.v1 as tf
import numpy as np

def dense(x, latent_dim, hidden_dim, num = 5, name = 'dense'):
    with tf.variable_scope(name):
        for i in range(num):
            x = tf.layers.dense(x, hidden_dim, activation = 'relu', name = 'dense_' + str(i))
        x = tf.layers.dense(x, latent_dim, activation = 'relu', name = 'output') 
    return x

def split(x):
    with tf.name_scope('split'):
        hidden_dim = int(x.shape[-1] / 2)
        x = tf.reshape(x, [-1, hidden_dim, 2])
    return x[:,:,0], x[:,:,1]

def concat(x):
    with tf.name_scope('concat'):
        x = tf.concat(x, axis = 1)
    return x

class baseCoupleLayer:
    def forward(self, x, hidden_dim, name = 'forward'):
        latent_dim = int(hidden_dim / 2)
        x1, x2 = split(x)
        x2 = self.coupling(x2, dense(x1, latent_dim, hidden_dim, name = name), inverse = False)
        x = concat([x1, x2])
        return x
    
    def inverse(self, x, hidden_dim, name = 'inverse'):
        latent_dim = int(hidden_dim / 2)
        x1, x2 = split(self.x)
        x2 = self.coupling(x2, dense(x1, latent_dim, hidden_dim, name = name), inverse = True)
        x = concat([x1, x2])
        return x
    
    def coupling(self, a, b, inverse):
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer!")

class additiveCoupleLayer(baseCoupleLayer):
    def coupling(self, a, b, inverse):
        if inverse:
            return a - b
        else:
            return a + b

class multiplicativeCoupleLayer(baseCoupleLayer):
    def coupling(self, a, b, inverse):
        if inverse:
            return tf.multiply(a, tf.reciprocal(b))        
        else:
            return tf.multiply(a, b) 

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

def model(sess, train_iterator, test_iterator, dim = 784, reuse = False):
    def m_loss(x, diag, labels = 10):
        with tf.variable_scope('model', reuse = reuse):
            loss = tf.reduce_mean(-(tf.reduce_sum(diag) \
                - tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1)))
        return loss
    _x, _y = train_iterator
    model = NICE(_x.shape[-1])
    with tf.variable_scope('encoder', reuse = reuse):
        x = model.encoder(_x)
    
    loss = m_loss(x, model.diag)
    
    var = tf.trainable_variables()
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(loss, var_list = var)

    # train = sess.run([optimizer, en_loss])

    



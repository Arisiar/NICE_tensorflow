import tensorflow as tf
import numpy as np

def dense(x, latent_dim, hidden_dim, num, name = 'dense'):
    with tf.variable_scope(name):
        for i in range(num):
            x = tf.layers.dense(x, hidden_dim, activation = 'relu', name = 'dense_' + str(i))
        x = tf.layers.dense(x, latent_dim, activation = 'relu', name = 'output') 
    return x

def split(x):
    with tf.name_scope('split'):
        dim = self.x.shape()[-1]
        x = tf.reshape(self.x, [-1, dim / 2, 2])
    return x[:,:,0], x[:,:,1]

def concat(x):
    with tf.name_scope('concat'):
        x = tf.concat(x, axis = 2)
        x = tf.reshape(x, np.prod(x.shape[1:]))
    return x

class baseCoupleLayer:
    def __init__(self, x, dim):
        self.x = x
        self.hidden_dim = dim
        self.latent_dim = int(dim / 2)

    def forward(self):
        x1, x2 = split(self.x)
        x2 = self.coupling(x2, dense(x1, self.latent_dim, self.hidden_dim, name = 'forward'), inverse = False)
        x = concat([x1, x2])
        return x
    
    def inverse(self):
        x1, x2 = split(self.x)
        x2 = self.coupling(x2, dense(x1, self.latent_dim, self.hidden_dim, name = 'inverse'), inverse = True)
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
        self.layer_1 = additiveCoupleLayer()
        self.layer_2 = additiveCoupleLayer() 
        self.layer_3 = additiveCoupleLayer()
        self.layer_4 = additiveCoupleLayer()
        self.diag = tf.Variable(tf.ones(input_dim), name = 'diag')

    def encoder(self, x):
        x = self.layer_1.forward(x)
        x = self.layer_2.forward(x)
        x = self.layer_3.forward(x)
        x = self.layer_4.forward(x)
        x = tf.matmul(x, tf.diag(tf.exp(self.diag)))

        return x

    def decoder(self, x):
        x = tf.matmul(x, tf.diag(tf.reciprocal(tf.exp(self.diag))))
        x = self.layer_4.inverse(x)
        x = self.layer_3.inverse(x)
        x = self.layer_2.inverse(x)
        x = self.layer_1.inverse(x)

        return x

def model(x, y, dim = 784):
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, dim], name='image')
        Y = tf.placeholder(tf.int32, [None], name='label')

    model = NICE(input_dim)
    with tf.variable_scope('encoder', reuse = reuse):
        x = model.encoder(X)

    en_loss = loss(x, model.diag)
    
    en_var = [var for var in tf.trainable_variables() if 'encoder' in var.name]
    with tf.name_Scope('en_optmize'):
        en_optm = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(en_loss, var_list = en_var)



def loss(x, diag, labels = 10):
    with tf.variable_scope('model', reuse = reuse):
        loss = tf.reduce_mean(-(tf.reduce_sum(diag) \
            - tf.reduce_sum(tf.math.log1p(tf.exp(x)) + tf.math.log1p(tf.exp(-x)), axis = 1)))
    return loss

    



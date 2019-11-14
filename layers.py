######################  Tensorflow 1.x ###################### [UNDONE] 11.13
import tensorflow as tf

def dense(x, latent_dim, hidden_dim, num = 5, name = 'dense'):
    with tf.compat.v1.variable_scope(name):
        for i in range(num):
            x = tf.compat.v1.layers.dense(x, hidden_dim, activation = 'relu', name = 'dense_' + str(i))
        x = tf.compat.v1.layers.dense(x, latent_dim, activation = 'relu', name = 'output') 
    return x

def split(inputs):
    dim = inputs.shape[-1] // 2
    x = tf.reshape(inputs, [-1, dim, 2])
    return x[:,:,0], x[:,:,1]

def concat(inputs):
    x = tf.concat(inputs, axis = 1)
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
        x1, x2 = split(x)
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


######################  Tensorflow 2.x / Keras ######################
from tensorflow import keras

def dense(dim):
    inputs = keras.Input(dim)
    x = inputs
    for _ in range(5):
        x = keras.layers.Dense(1000, activation='relu')(x)
    x = keras.layers.Dense(dim, activation='relu')(x)
    return keras.Model(inputs, x)

class CoupleLayer(keras.layers.Layer):
    def __init__(self, isInverse = False):
        super(CoupleLayer, self).__init__()
        self.isInverse = isInverse
  
    def build(self, input_shape):
        dim = int(input_shape[-1] / 2)
        self.dense = dense(dim)

    def call(self, inputs):
        x_1, x_2 = split(inputs)

        mx1 = self.dense(x_1)
        
        x_2 = x_2 + mx1 if not self.isInverse else x_2 - mx1

        x = concat([x_1, x_2])
        
        return x

    def inverse(self):
        return CoupleLayer(isInverse = True)

class ScaleLayer(keras.layers.Layer):
    def __init__(self, isInverse = False, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.isInverse = isInverse

    def build(self, input_shape):
        self.diag = self.add_weight(
                shape = input_shape,
                initializer = 'random_normal', 
                trainable = True,
                name = 'diag')

    def call(self, inputs):
        if self.isInverse:
            return tf.matmul(inputs, tf.linalg.diag(tf.math.reciprocal(tf.exp(self.diag))))
        else:
            return tf.matmul(inputs, tf.linalg.diag(tf.exp(self.diag))), self.diag

    def inverse(self):
        return ScaleLayer(isInverse = True)
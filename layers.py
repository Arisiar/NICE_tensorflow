import tensorflow as tf

def network(latent_dim, hidden_dim, num_layers = 5):
    inputs = tf.keras.Input(hidden_dim)
    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    return tf.keras.Model(inputs, x)

class NICEBlock(tf.keras.layers.Layer):
    def __init__(self, is_inverse = False):
        super(NICEBlock, self).__init__()
        self.is_inverse = is_inverse
  
    def build(self, input_shape):
        self.network = network(int(input_shape[-1]), int(input_shape[-1] / 2))

    def call(self, inputs):
        x = tf.reshape(inputs, [-1, int(inputs.shape[-1] / 2), 2])
        x_1, x_2 = x[:,:,0], x[:,:,1]
        
        mx1 = self.network(x_1)
        
        x_2 = x_2 - mx1 if self.is_inverse else x_2 + mx1

        x = tf.concat([x_2, x_1], axis = 1)
    
        return x

    def inverse(self):
        return NICEBlock(True)

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, is_inverse = False):
        super(ScaleLayer, self).__init__()
        self.is_inverse = is_inverse

    def build(self, input_shape):
        self.diag = self.add_weight(name = 'diag',
                shape = (1, input_shape[1]),
                initializer = 'random_normal', 
                trainable = True)

    def call(self, inputs):
        if self.is_inverse:
            return inputs * tf.exp(-self.diag)
            # return tf.matmul(inputs, tf.linalg.diag(tf.math.reciprocal(tf.exp(-self.diag))))
        else:
            self.add_loss(-tf.keras.backend.sum(self.diag))
            return inputs * tf.exp(self.diag)
            # return tf.matmul(inputs, tf.linalg.diag(tf.exp(self.diag)))

    def inverse(self):
        return ScaleLayer(True)
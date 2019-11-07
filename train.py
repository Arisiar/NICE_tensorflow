import tensorflow as tf
import numpy as np

def getData():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_size = x_train.shape[1] # 28
    x_train = np.reshape(x_train, [-1, img_size * img_size])
    x_test = np.reshape(x_test, [-1, img_size * img_size])
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x, y = (x_train, y_train), (x_test, y_test)
    
    return x, y

def main():
    
    train, test = getData()
   
    sess = tf.compat.v1.Session()
    model = model.model(sess, train, 784)



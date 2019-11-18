import tensorflow as tf
import argparse
import numpy as np
import logging, os
from model import model

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# tf.compat.v1.disable_eager_execution()


def minst_old(n_batch_train, n_batch_test, size = 28):
    def make_iterator(flow):
        def iterator():
            x, y = flow.next()
            x = tf.reshape(x, [-1, size * size])
            y = tf.reshape(y, [-1])
            return x, y
        return iterator

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    train_num, test_num = x_train.shape[0], x_test.shape[0]

    x_train = np.reshape(x_train, [-1, size, size, 1])
    x_test = np.reshape(x_test, [-1, size, size, 1])

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    train_generator = ImageDataGenerator(rescale = 1./255, width_shift_range = 0.1, height_shift_range = 0.1)
    test_generator = ImageDataGenerator(rescale = 1./255)

    train_generator.fit(x_train)
    train_flow = train_generator.flow(x_train, y_train, n_batch_train)
    train_iterator = make_iterator(train_flow)

    test_generator.fit(x_test)
    test_flow = test_generator.flow(x_test, y_test, n_batch_test, shuffle = False)
    test_iterator = make_iterator(test_flow)

    return train_iterator, test_iterator, train_num, test_num 

def minst(batch, size = 28):
    def preprocessing_fn(x):
        x = tf.cast(x / 255, tf.float32) # , tf.cast(y, tf.int64)
        x = tf.reshape(x, [size * size]) # , tf.reshape(y, [-1])
        return x

    (images, labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = (tf.data.Dataset.from_tensor_slices(images)
                        .map(preprocessing_fn, num_parallel_calls = 16)
                        .shuffle(1000)
                        .batch(batch))

    return dataset

def train(args):
    dataset = minst(args.batch)

    m = model(args, dataset, dim = 784)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--epochs", dest = 'epochs', default = 1, type = int,
                            help = "Number of epochs to train on. [100]")
    parser.add_argument("--save_epochs", dest = 'save_epochs', default = 10, type = int,
                            help = "Number of epochs to save. [10]")
    parser.add_argument("--batch", dest = 'batch', default = 30, type = int,
                            help = "Size of batch to train on. [30]")
    parser.add_argument("--step", dest = 'step', default = 5, type = int,
                            help = "Step per epoch. [5]") 
    parser.add_argument("--save_path", dest = 'save_path', default = './checkpoint',
                            help = "Path to save the trained model. [./checkpoint]")    
    parser.add_argument("--lr", default = 0.001, dest = 'lr', type = float,
                            help = "Learning rate for ADAM optimizer. [0.001]") 
    parser.add_argument("--beta1", dest = 'beta1', default = 0.9, type = float,
                            help = "Beta 1 for ADAM optimizer. [0.9]") 
    parser.add_argument("--beta2", dest = 'beta2', default = 0.01, type = float,
                            help = "Beta 2 for ADAM optimizer. [0.01]") 
    args = parser.parse_args() 

    train(args)                      
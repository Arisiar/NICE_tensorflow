import tensorflow as tf
import argparse
from model import model

def minst(batch, epochs, size = 28):
    def preprocessing_fn(x):
        noise = 0.01 * tf.random.uniform(x.shape)
        x = tf.cast((x / 255) - noise, tf.float32) # , tf.cast(y, tf.int64)
        x = tf.reshape(x, [size * size]) # , tf.reshape(y, [-1])
        return x

    (images, labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = (tf.data.Dataset.from_tensor_slices(images)
                        .map(preprocessing_fn, num_parallel_calls = 16)
                        .shuffle(1000)
                        .batch(batch))

    return  dataset

def train(args):
    dataset = minst(args.batch, args.epochs)
    
    model(args, dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--epochs", dest = 'epochs', default = 1, type = int,
                            help = "Number of epochs to train on. [100]")
    parser.add_argument("--img_size", dest = 'img_size', default = 28, type = int,
                            help = "Image size of MINST. [28]")
    parser.add_argument("--test_num", dest = 'test_num', default = 10, type = int,
                            help = "Number of test images. [10]")
    parser.add_argument("--save_step", dest = 'save_step', default = 100, type = int,
                            help = "Number of save step. [100]")
    parser.add_argument("--batch", dest = 'batch', default = 128, type = int,
                            help = "Size of batch to train on. [128]")
    parser.add_argument("--save_path", dest = 'save_path', default = './checkpoint',
                            help = "Path to save the trained model. [./checkpoint]")    
    parser.add_argument("--lr", default = 1e-03, dest = 'lr', type = float,
                            help = "Learning rate for ADAM optimizer. [0.001]") 
    parser.add_argument("--beta1", dest = 'beta1', default = 0.9, type = float,
                            help = "Beta 1 for ADAM optimizer. [0.9]") 
    parser.add_argument("--beta2", dest = 'beta2', default = 1e-02, type = float,
                            help = "Beta 2 for ADAM optimizer. [0.01]") 
    parser.add_argument("--epsilon", dest = 'epsilon', default = 1e-04, type = float,
                            help = "epsilon for ADAM optimizer. [0.0001]") 
    args = parser.parse_args() 

    train(args)                      
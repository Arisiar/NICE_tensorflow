import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
def get_data(n_batch_train = 32, n_batch_test = 32, size = 28):
    def make_iterator(flow):
        x, y = flow.next()
        x = tf.reshape(x, [-1, size * size])
        y = tf.reshape(y, [-1])
        return x, y
   
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, [-1, size, size, 1])
    x_test = np.reshape(x_test, [-1, size, size, 1])

    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    train_generator = ImageDataGenerator(rescale = 1./255, width_shift_range = 0.1, height_shift_range = 0.1)
    test_generator = ImageDataGenerator(rescale = 1./255)

    train_generator.fit(x_train)
    train_iterator = make_iterator(train_generator.flow(x_train, y_train, n_batch_train))

    test_generator.fit(x_test)
    test_iterator = make_iterator(test_generator.flow(x_test, y_test, n_batch_test, shuffle = False))

    return train_iterator, test_iterator 

def main():
    train, test = get_data()
    with tf.compat.v1.Session() as sess:
        import model
        model = model.model(sess, train, test)

main()
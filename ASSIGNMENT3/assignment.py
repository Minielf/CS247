import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:

    def __init__(self, image, label, is_training, keep_prob):
        """
        A Model class contains a computational graph that classifies images
        to predictions. Each of its methods builds part of the graph
        on Model initialization. Do not modify the constructor, as doing so
        would break the autograder. You may, however, add class variables
        to use in your graph-building. e.g. learning rate, 

        image: the input image to the computational graph as a tensor
        label: the correct label of an image as a tensor
        prediction: the output prediction of the computational graph,
                    produced by self.forward_pass()
        optimize: the model's optimizing tensor produced by self.optimizer()
        loss: the model's loss produced by computing self.loss_function()
        accuracy: the model's prediction accuracy
        """
        self.image = image
        self.label = label
        self.alpha = 1e-4
        self.is_training = is_training
        self.keep_prob = keep_prob

        # TO-DO: Add any class variables you want to use.
        self.prediction = self.forward_pass()
        self.loss = self.loss_function()
        self.optimize = self.optimizer()
        self.accuracy = self.accuracy_function()

    def forward_pass(self):
        """
        Predicts a label given an image using convolution layers

        :return: the prediction as a tensor
        """
        # TO-DO: Build up the computational graph for the forward pass.
        image = tf.reshape(self.image, [-1, 28, 28, 1])
        # initialize weight and bias for conv layer1
        W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=.1))
        b1 = tf.Variable(tf.truncated_normal([32], stddev=.1))
        # conv for layer1
        conv1 = tf.nn.conv2d(image, W1, [1, 1, 1, 1], "SAME") + b1
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # initialize weight and bias for conv layer2
        W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=.1))
        b2 = tf.Variable(tf.truncated_normal([64], stddev=.1))
        # conv for layer2
        conv2 = tf.nn.conv2d(conv1, W2, [1, 1, 1, 1], "SAME") + b2
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        # get dimensions for the fully connected layer
        dimensions = conv2.get_shape().as_list()
        dim = dimensions[1] * dimensions[2] * dimensions[3]
        # initialize weight and bias for the fully connected layers
        Wfc = tf.Variable(tf.truncated_normal([dim, 1024], stddev=.1))
        bfc = tf.Variable(tf.truncated_normal([1024], stddev=.1))
        Wout = tf.Variable(tf.truncated_normal([1024, 10], stddev=.1))
        bout = tf.Variable(tf.truncated_normal([10], stddev=.1))

        fc = tf.reshape(conv2, shape=[-1, Wfc.get_shape().as_list()[0]])
        fc = tf.add(tf.matmul(fc, Wfc), bfc)
        fc = tf.nn.relu(fc)

        prbs = tf.matmul(fc, Wout) + bout
        return prbs

    def loss_function(self):
        """
        Calculates the model cross-entropy loss

        :return: the loss of the model as a tensor
        """
        # TO-DO: Add the loss function to the computational graph
        top = slim.dropout(self.prediction, keep_prob=self.keep_prob, is_training=self.is_training, scope='conv_top_dropout')
        output = slim.layers.softmax(slim.layers.flatten(top))
        loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(output) + 1e-10, axis=[1]))
        return loss

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer

        :return: the optimizer as a tensor
        """
        # TO-DO: Add the optimizer to the computational graph
        train = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        return train

    def accuracy_function(self):
        """
        Calculates the model's prediction accuracy by comparing
        predictions to correct labels – no need to modify this

        :return: the accuracy of the model as a tensor
        """
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1),
                                      tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():

    # TO-DO: import MNIST data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # TO-DO: Set-up placeholders for inputs and outputs
    batchSz = 50
    img = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    ans = tf.placeholder(dtype=tf.float32, shape=(None, 10))
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # TO-DO: initialize model and tensorflow variables
    m = Model(img, ans, is_training, keep_prob)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # TO-DO: Set-up the training step, for 2000 batches with a batch size of 50
    steps = 2000
    for i in range(steps):
        imgs, anss = mnist.train.next_batch(batchSz)
        sess.run(m.optimize, {img: imgs, ans: anss, is_training: False, keep_prob: 0.5})

    # TO-DO: run the model on test data and print the accuracy
    sumAcc = 0
    for i in range(1000):
        imgs, anss = mnist.test.next_batch(batchSz)
        sumAcc += sess.run(m.accuracy, feed_dict={img: imgs, ans: anss, is_training: False, keep_prob: 1})
    print("Test Accuracy: %r" % (sumAcc / 1000))
    return


if __name__ == '__main__':
    main()

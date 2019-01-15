import numpy as np
import os
import gzip
import random
# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    """
        This model class provides the data structures for your NN, 
        and has functions to test the three main aspects of the training.
        The data structures should not change after each step of training.
        You can add to the class, but do not change the 
        stencil provided except for the blanks and pass statements.
        Make sure that these functions work with a loop to call them multiple times,
        instead of implementing training over multiple steps in the function

        Arguments: 
        train_images - NumPy array of training images
        train_labels - NumPy array of labels
    """
    def __init__(self, train_images, train_labels):
        self.input_size, self.num_classes, self.batchSz, self.learning_rate = 784, 10, 1, 0.5
        self.train_images = train_images
        self.train_labels = train_labels
        # sets up weights and biases...
        self.W = np.zeros((self.input_size, self.num_classes))
        self.b = np.zeros((1, self.num_classes))

    def run(self):
        """
        Does the forward pass, loss calculation, and back propagation 
        for this model for one step

        Args: None
        Return: None
        """
        delta_w = np.zeros((self.input_size, self.num_classes))
        delta_b = np.zeros((1, self.num_classes))
        for i in range(self.batchSz):
            r = random.randint(0, self.train_images.shape[0] - 1)
            l = self.b + np.dot(self.train_images[r],  self.W)
            e_x = np.exp(l - np.max(l))
            p = e_x / e_x.sum()
            loss = -np.log(p)
            p[0, self.train_labels[r]] -= 1
            delta_w += -self.learning_rate * np.dot(self.train_images[r].reshape(-1, 1), p)
            delta_b += -self.learning_rate * p
        self.W += delta_w
        self.b += delta_b

    def accuracy_function(self, test_images, test_labels):
        """
        Calculates the accuracy of the model against test images and labels

        DO NOT EDIT
        Arguments
        test_images: a normalized NumPy array
        test_labels: a NumPy array of ints
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = np.argmax(scores, axis=1)
        return np.mean(predicted_classes == test_labels)


def main():
    # TO-DO: import MNIST test data
    with open('train-images-idx3-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        buf1 = bytestream.read(16 + 60000 * 28 * 28)
        train_images = np.frombuffer(buf1, dtype='uint8', offset=16).reshape(60000, 28 * 28)
        train_images = train_images[:10000] / 255.0
    with open('train-labels-idx1-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        buf2 = bytestream.read(8 + 60000)
        train_labels = np.frombuffer(buf2, dtype='uint8', offset=8)
        train_labels = train_labels[:10000]

    with open('t10k-images-idx3-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        buf3 = bytestream.read(16 + 10000 * 28 * 28)
        test_images = np.frombuffer(buf3, dtype='uint8', offset=16).reshape(10000, 28 * 28) / 255.0
    with open('t10k-labels-idx1-ubyte.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
        buf4 = bytestream.read(8 + 10000)
        test_labels = np.frombuffer(buf4, dtype='uint8', offset=8)

    # TO-DO: Create Model with training data arrays that sets up the weights and biases.
    m = Model(train_images, train_labels)

    # TO-DO: Run model for number of steps by calling run() each step
    steps = 10000
    for i in range(steps):
        m.run()

    # TO-DO: test the accuracy by calling accuracy_function with the test data
    print(m.accuracy_function(test_images, test_labels))


if __name__ == '__main__':
    main()

"""
Stencil layout for your trigram language model assignment, with embeddings.

The stencil has three main parts:
    - A class stencil for your actual trigram language model. The class is complete with helper
    methods declarations that break down what your model needs to do into reasonably-sized steps,
    which we recommend you use.

    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.

    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.
"""

import tensorflow as tf
import numpy as np
import math


class TrigramLM:
    def __init__(self, X1, X2, Y, vocab_sz):
        """
        Instantiate your TrigramLM Model, with whatever hyperparameters are necessary
        !!!! DO NOT change the parameters of this constructor !!!!

        X1, X2, and Y represent the first, second, and third words of a batch of trigrams.
        (hint: they should be placeholders that can be fed batches via your feed_dict).

        You should implement and use the "read" function to calculate the vocab size of your corpus
        before instantiating the model, as the model should be specific to your corpus.
        """
        
        # TODO: Define network parameters

        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        self.vocab_sz = vocab_sz
        self.embedding_sz = 100

        self.logits = self.forward_pass()

        # IMPORTANT - your model MUST contain two instance variables,
        # self.loss_val and self.train_op, that contain the loss computation graph 
        # (as you will define in in loss()), and training operation (as you will define in train())
        self.loss = self.loss()
        self.train_op = self.optimizer()

    def forward_pass(self):
        """
        Build the inference computation graph for the model, going from the input to the output
        logits (before final softmax activation). This is analogous to "prediction".
        """

        # TODO: Compute the logits
        W = tf.Variable(tf.random_normal([self.embedding_sz * 2, self.vocab_sz],
                                         stddev=1.0 / math.sqrt(self.embedding_sz)))
        b = tf.Variable(tf.random_normal([self.vocab_sz]))
        E = tf.Variable(tf.random_normal([self.vocab_sz, self.embedding_sz], stddev=0.1))
        embed1 = tf.nn.embedding_lookup(E, self.X1)
        embed2 = tf.nn.embedding_lookup(E, self.X2)
        both = tf.concat([embed1, embed2], 1)
        logits = tf.add(tf.matmul(both, W), b)
        return logits

    def loss(self):
        """
        Build the cross-entropy loss computation graph.
        DO 
        """

        # TODO: Perform the loss computation
        loss = tf.losses.sparse_softmax_cross_entropy(self.Y, self.logits)
        return loss

    def optimizer(self):
        """
        Build the training operation, using the cross-entropy loss and an Adam Optimizer.
        """

        # TODO: Execute the training operation
        train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        return train


def process_data(read_data, word2id):
    s = map(lambda x: x.split(), read_data)

    data = []
    for sentence in s:
        for i in range(2, len(sentence)):
            word = sentence[i]
            prev_word = [sentence[i - 2], sentence[i - 1]]
            data.append([word2id[word], [word2id[prev_word[0]], word2id[prev_word[1]]]])
    return data


def read(train_file, dev_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.
    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    """

    # TODO: Read and process text data given paths to train and development data files
    with open(train_file, 'r') as f:
        read_train_data = f.read().split("\n")
    with open(dev_file, 'r') as f:
        read_dev_data = f.read().split("\n")

    vocab = set(" ".join(read_train_data).split())
    word2id = {w: i for i, w in enumerate(list(vocab))}

    traindata = process_data(read_train_data, word2id)
    devdata = process_data(read_dev_data, word2id)
    return word2id, traindata, devdata


def get_data(data):
    X1 = [j[0] for i, j in data]
    X2 = [j[1] for i, j in data]
    Y = [i for i, j in data]
    return X1, X2, Y



def main():

    # TODO: Import and process data
    word2id, traindata, devdata = read('train.txt', 'dev.txt')

    # TODO: Set up placeholders for inputs and outputs to pass into model's constructor
    X1 = tf.placeholder(tf.int32, shape=[None])
    X2 = tf.placeholder(tf.int32, shape=[None])
    Y = tf.placeholder(tf.int32, shape=[None])

    # TODO: Initialize model and tensorflow variables
    vocab_sz = len(word2id)
    m = TrigramLM(X1, X2, Y, vocab_sz)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # TODO: Set up the training step, training with 1 epoch and with a batch size of 20
    batchSz = 20
    trainX1, trainX2, trainY = get_data(traindata)
    devX1, devX2, devY = get_data(devdata)
    los, count = 0, 0

    for start, end in zip(range(0, len(traindata) - batchSz, batchSz), range(batchSz, len(traindata), batchSz)):
        _, tmp_los = sess.run([m.train_op, m.loss],
                              feed_dict={X1: trainX1[start:end], X2: trainX2[start:end], Y: trainY[start:end]})
        los += tmp_los
        count += 1
        if count % 100 == 0:
            print('batch: {}, perplexity: {}'.format(count, np.exp(los / count)))

    # TODO: Run the model on the development set and print the final perplexity
    # Remember that perplexity is just defined as: e^(average_loss_per_input)!
    loss = sess.run(m.loss, feed_dict={X1: devX1, X2: devX2, Y: devY})
    print("The perplexity is: ", np.exp(loss))


if __name__ == "__main__":
    main()

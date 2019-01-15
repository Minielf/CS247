"""
Stencil layout for your RNN language model assignment.
The stencil has three main parts:
    - A class stencil for your language model
    - A "read" helper function to isolate the logic of parsing the raw text files. This is where
    you should build your vocabulary, and transform your input files into data that can be fed into the model.
    - A main-training-block area - this code (under "if __name__==__main__") will be run when the script is,
    so it's where you should bring everything together and execute the actual training of your model.


Q: What did the computer call its father?
A: Data!

"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Use this variable to declare your batch size. Do not rename this variable.
BATCH_SIZE = 50

# Your window size must be 20. Do not change this variable!
WINDOW_SIZE = 20


def read(train_file, test_file):
    """
    Read and parse the file, building the vectorized representations of the input and output.

    !!!!!PLEASE FOLLOW THE STENCIL. WE WILL GRADE THIS!!!!!!!

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of (train_x, train_y, test_x, test_y, vocab)

    train_x: List of word ids to use as training input
    train_y: List of word ids to use as training labels
    test_x: List of word ids to use as testing input
    test_y: List of word ids to use as testing labels
    vocab: A dict mapping from word to vocab id
    """

    # TO-DO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.
    with open(train_file, 'r') as f:
        read_train_data = f.read().split("\n")
    with open(test_file, 'r') as f:
        read_dev_data = f.read().split("\n")

    train_data = " ".join(read_train_data).split()
    dev_data = " ".join(read_dev_data).split()
    vocab = set(train_data)
    word2id = {w: i for i, w in enumerate(list(vocab))}
    print(word2id["STOP"])
    train_id = [word2id[i] for i in train_data]
    train_chunks = len(train_data) // (BATCH_SIZE * WINDOW_SIZE)
    train_id = train_id[:train_chunks * BATCH_SIZE * WINDOW_SIZE + 1]
    train_x = train_id[:-1]
    train_y = train_id[1:]

    dev_id = [word2id[i] for i in dev_data]
    dev_chunks = len(dev_data) // (BATCH_SIZE * WINDOW_SIZE)
    dev_id = dev_id[:dev_chunks * BATCH_SIZE * WINDOW_SIZE + 1]
    test_x = dev_id[:-1]
    test_y = dev_id[1:]

    return train_x, train_y, test_x, test_y, word2id



class Model:
    def __init__(self, inputs, labels, keep_prob, vocab_size):
        """
        The Model class contains the computation graph used to predict the next word in sequences of words.

        Do not delete any of these variables!

        inputs: A placeholder of input words
        label: A placeholder of next words
        keep_prob: The keep probability of dropout in the embeddings
        vocab_size: The number of unique words in the data
        """

        # Input tensors, DO NOT CHANGE
        self.inputs = inputs
        self.labels = labels
        self.keep_prob = keep_prob

        # DO NOT CHANGE
        self.vocab_size = vocab_size
        self.prediction = self.forward_pass()  # Logits for word predictions
        self.loss = self.loss_function()  # The average loss of the batch
        self.optimize = self.optimizer()  # An optimizer (e.g. ADAM)
        self.perplexity = self.perplexity_function()  # The perplexity of the model, Tensor of size 1

    def forward_pass(self):
        """
        Use self.inputs to predict self.labels.
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :return: logits: The prediction logits as a tensor
        """
        embedding_sz = 256
        rnnSz = 200
        E = tf.Variable(tf.random_normal([self.vocab_size, embedding_sz], stddev=0.1))
        W = tf.Variable(tf.random_normal([rnnSz, self.vocab_size]))
        b = tf.Variable(tf.random_normal([self.vocab_size]))
        embed = tf.nn.embedding_lookup(E, self.inputs)
        embed = tf.nn.dropout(embed, self.keep_prob)
        rnn = tf.contrib.rnn.LSTMCell(rnnSz)
        output, nextState = tf.nn.dynamic_rnn(rnn, embed, initial_state=None, dtype=tf.float32)
        output = tf.reshape(output, [-1, rnnSz])
        logits = tf.add(tf.matmul(output, W), b)
        logits = tf.reshape(logits, [BATCH_SIZE, WINDOW_SIZE, self.vocab_size])
        return logits

    def optimizer(self):
        """
        Optimizes the model loss using an Adam Optimizer
        :return: the optimizer as a tensor
        """
        return tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def loss_function(self):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :return: the loss of the model as a tensor of size 1
        """
        return tf.contrib.seq2seq.sequence_loss(self.prediction, self.labels,
                                                tf.ones_like(self.labels, dtype=tf.float32))

    def perplexity_function(self):
        """
        Calculates the model's perplexity by comparing predictions to correct labels
        :return: the perplexity of the model as a tensor of size 1
        """
        return tf.exp(self.loss)


def main():
    # Preprocess data
    train_file = "train.txt"
    dev_file = "dev.txt"
    train_x, train_y, test_x, test_y, vocab_map = read(train_file, dev_file)
    vocabSz = len(vocab_map)

    # TODO: define placeholders
    inputs = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
    labels = tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
    keep_prob = tf.placeholder(tf.float32)

    # TODO: initialize model

    m = Model(inputs, labels, keep_prob, vocabSz)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # TODO: Set-up the training step:
    # - 1) divide training set into equally sized batch chunks. We recommend 50 batches.
    # - 2) split these batch segments into windows of size WINDOW_SIZE.
    los, count = 0, 0
    for start, end in zip(range(0, len(train_x) - BATCH_SIZE * WINDOW_SIZE, BATCH_SIZE * WINDOW_SIZE),
                          range(BATCH_SIZE * WINDOW_SIZE, len(train_x), BATCH_SIZE * WINDOW_SIZE)):
        newx = np.reshape(train_x[start:end], [BATCH_SIZE, WINDOW_SIZE])
        newy = np.reshape(train_y[start:end], [BATCH_SIZE, WINDOW_SIZE])
        _, tmp_los = sess.run([m.optimize, m.loss],
                              feed_dict={inputs: newx, labels: newy, keep_prob: 0.5})
        los += tmp_los
        count += 1
        if count % 100 == 0:
            print('batch: {}, perplexity: {}'.format(count, np.exp(los / count)))

    # TODO: Run the model on the development set and print the final perplexity
    test_x = np.reshape(test_x, [-1, WINDOW_SIZE])
    test_y = np.reshape(test_y, [-1, WINDOW_SIZE])
    loss = sess.run(m.perplexity, feed_dict={inputs: test_x, labels: test_y, keep_prob: 1})
    print("TEST DATA FINAL PERPLEXITY:", loss)


if __name__ == '__main__':
    main()

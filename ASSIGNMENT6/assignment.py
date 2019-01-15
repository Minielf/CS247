import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_FR = "./processed_data/french_train.txt"
TRAIN_EN = "./processed_data/english_train.txt"
TEST_FR = "./processed_data/french_test.txt"
TEST_EN = "./processed_data/english_test.txt"

# This variable is the batch size the auto-grader will use when training your model.
BATCH_SIZE = 100
# Do not change these variables.
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 12
STOP_TOKEN = "*STOP*"


def pad_corpus(french_file_name, english_file_name):
    """
    Arguments are files of French, English sentences. All sentences are padded with "*STOP*" at
    the end to make their lengths match the window size. For English, an additional "*STOP*" is
    added to the beginning. For example, "I am hungry ." becomes
    ["*STOP*, "I", "am", "hungry", ".", "*STOP*", "*STOP*", "*STOP*",  "*STOP", "*STOP", "*STOP", "*STOP", "*STOP"]

    :param french_file_name: string, a path to a french file
    :param english_file_name: string, a path to an english file

    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English, list of French sentence lengths, list of English sentence lengths)
    """

    french_padded_sentences = []
    french_sentence_lengths = []
    with open(french_file_name, 'rt', encoding='latin') as french_file:
        for line in french_file:
            padded_french = line.split()[:FRENCH_WINDOW_SIZE]
            french_sentence_lengths.append(len(padded_french))
            padded_french += [STOP_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_french))
            french_padded_sentences.append(padded_french)

    english_padded_sentences = []
    english_sentence_lengths = []
    with open(english_file_name, "rt", encoding="latin") as english_file:
        for line in english_file:
            padded_english = line.split()[:ENGLISH_WINDOW_SIZE]
            english_sentence_lengths.append(len(padded_english))
            padded_english = [STOP_TOKEN] + padded_english + [STOP_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_english))
            english_padded_sentences.append(padded_english)

    return french_padded_sentences, english_padded_sentences, french_sentence_lengths, english_sentence_lengths


class Model:
    """
        This is a seq2seq model.

        REMINDER:

        This model class provides the data structures for your NN,
        and has functions to test the three main aspects of the training.
        The data structures should not change after each step of training.
        You can add to the class, but do not change the
        function headers or return types.
        Make sure that these functions work with a loop to call them multiple times,
        instead of implementing training over multiple steps in the function
    """

    def __init__(self, french_window_size, english_window_size, french_vocab_size, english_vocab_size):
        """
        Initialize a Seq2Seq Model with the given data.

        :param french_window_size: max len of French padded sentence, integer
        :param english_window_size: max len of English padded sentence, integer
        :param french_vocab_size: Vocab size of french, integer
        :param english_vocab_size: Vocab size of english, integer
        """

        # Initialize Placeholders
        self.french_vocab_size, self.english_vocab_size = french_vocab_size, english_vocab_size

        self.encoder_input = tf.placeholder(tf.int32, shape=[None, french_window_size], name='french_input')
        self.encoder_input_length = tf.placeholder(tf.int32, shape=[None], name='french_length')

        self.decoder_input = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_input')
        self.decoder_input_length = tf.placeholder(tf.int32, shape=[None], name='english_length')
        self.decoder_labels = tf.placeholder(tf.int32, shape=[None, english_window_size], name='english_labels')

        # Please leave these variables
        self.logits = self.forward_pass()
        self.loss = self.loss_function()
        self.train = self.back_propagation()
        self.accuracy = self.per_symbol_accuracy()

    def forward_pass(self):
        """
        Calculates the logits

        :return: A tensor of size [batch_size, english_window_size, english_vocab_size]
        """
        embedSz = 256
        rnnSz = 200
        with tf.variable_scope("enc"):
            F = tf.Variable(tf.random_normal((self.french_vocab_size, embedSz), stddev=.1))
            embs = tf.nn.embedding_lookup(F, self.encoder_input)
            cell = tf.contrib.rnn.GRUCell(rnnSz)
            encOut, encState = tf.nn.dynamic_rnn(cell, embs, dtype=tf.float32)
            wadj = tf.random_normal([FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE], stddev=0.1)
            encOT = tf.transpose(encOut, [0, 2, 1])
            decIT = tf.tensordot(encOT, wadj, [[2], [0]])
            decI = tf.transpose(decIT, [0, 2, 1])

        with tf.variable_scope("dec"):
            E = tf.Variable(tf.random_normal((self.english_vocab_size, embedSz), stddev=.1))
            embs = tf.nn.embedding_lookup(E, self.decoder_input)
            decoder_input = tf.concat([decI, embs], 2)
            cell = tf.contrib.rnn.GRUCell(rnnSz)
            decOut, _ = tf.nn.dynamic_rnn(cell, decoder_input, initial_state=encState, dtype=tf.float32)

        W = tf.Variable(tf.random_normal([rnnSz, self.english_vocab_size], stddev=.1))
        b = tf.Variable(tf.random_normal([self.english_vocab_size], stddev=.1))
        logits = tf.tensordot(decOut, W, axes=[[2], [0]]) + b
        return logits

    def loss_function(self):
        """
        Calculates the model cross-entropy loss after one forward pass

        :return: the loss of the model as a tensor (averaged over batch)
        """
        self.mask = tf.sequence_mask(self.decoder_input_length, ENGLISH_WINDOW_SIZE, dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(self.logits, self.decoder_labels, self.mask)

    def back_propagation(self):
        """
        Adds optimizer to computation graph

        :return: optimizer
        """
        return tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def per_symbol_accuracy(self):
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.logits, 2), dtype=tf.int32), self.decoder_labels)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def main():
    # Load padded corpus
    train_french, train_english, train_french_lengths, train_english_lengths = pad_corpus(TRAIN_FR, TRAIN_EN)
    test_french, test_english, test_french_lengths, test_english_lengths = pad_corpus(TEST_FR, TEST_EN)
    # 1: Build French, English Vocabularies (dictionaries mapping word types to int ids)
    train_french_set = set([train_french[i][j]
                            for i in range(len(train_french)) for j in range(len(train_french[i]))])
    french_word2id = {w: i for i, w in enumerate(list(train_french_set))}
    train_english_set = set([train_english[i][j]
                             for i in range(len(train_english)) for j in range(len(train_english[i]))])
    english_word2id = {w: i for i, w in enumerate(list(train_english_set))}

    # 2: Creates batches. Remember that the English Decoder labels need to be shifted over by 1.
    train_french = [[french_word2id[x] for x in y] for y in train_french]
    test_french = [[french_word2id[x] for x in y] for y in test_french]
    train_input_english = [[english_word2id[x] for x in y[:-1]] for y in train_english]
    train_labels_english = [[english_word2id[x] for x in y[1:]] for y in train_english]
    test_input_english = [[english_word2id[x] for x in y[:-1]] for y in test_english]
    test_labels_english = [[english_word2id[x] for x in y[1:]] for y in test_english]

    # 3. Initialize model
    m = Model(FRENCH_WINDOW_SIZE, ENGLISH_WINDOW_SIZE, len(french_word2id), len(english_word2id))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # 4: Launch Tensorflow Session
    #       -Train
    los, count = 0, 0
    for start, end in zip(range(0, len(train_french) - BATCH_SIZE, BATCH_SIZE),
                          range(BATCH_SIZE, len(train_french), BATCH_SIZE)):
        _, tmp_los = sess.run([m.train, m.loss],
                              feed_dict={m.encoder_input: train_french[start: end],
                                         m.encoder_input_length: train_french_lengths[start: end],
                                         m.decoder_input: train_input_english[start: end],
                                         m.decoder_input_length: train_english_lengths[start: end],
                                         m.decoder_labels: train_labels_english[start: end]})
        los += tmp_los
        count += 1
        if count % 100 == 0:
            print('batch: {}, perplexity: {}'.format(count, np.exp(los / count)))

    #       -Test
    los, count, total_symbol_length, sum_acc = 0, 0, 0, 0
    for start, end in zip(range(0, len(test_french) - BATCH_SIZE, BATCH_SIZE),
                          range(BATCH_SIZE, len(train_french), BATCH_SIZE)):
        acc, tmp_los = sess.run([m.accuracy, m.loss],
                                feed_dict={m.encoder_input: test_french[start: end],
                                           m.encoder_input_length: test_french_lengths[start: end],
                                           m.decoder_input: test_input_english[start: end],
                                           m.decoder_input_length: test_english_lengths[start: end],
                                           m.decoder_labels: test_labels_english[start: end]})
        symbol_length = np.sum(test_english_lengths[start: end])
        total_symbol_length += symbol_length
        los += tmp_los * symbol_length
        sum_acc += acc * symbol_length
    print('The testing accuracy is : %.3f, perplexity is : %.3f' %
          (sum_acc / total_symbol_length, np.exp(los / total_symbol_length)))


if __name__ == '__main__':
    main()

"""

Human: What do we want!?
Computer: Natural language processing!
Human: When do we want it!?
Computer: When do we want what?

"""

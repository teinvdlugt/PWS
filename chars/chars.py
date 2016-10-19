import tensorflow as tf

# [a-z], . , , , !, ?, _

input_size = 31
batch_size = 32
learning_rate = .5
lstm_size = 512
buckets = [(10, 40), (20, 40), (30, 100), (50, 100)]
dtype = tf.float16

cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

encoder_inputs = []
decoder_inputs = []
target_weights = []

for i in xrange(buckets[-1][0]):
    # i is the maximum input length
    encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                         name="encoder{0}".format(i)))
for i in xrange((buckets[-1][1]) + 1):
    # i is the maximum output length + 1 ('GO' symbol)
    decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                         name="decoder{0}".format(i)))
    target_weights.append(tf.placeholder(dtype, shape=[None],
                                         name="weight{0}".format(i)))

targets = [decoder_inputs[i + 1] for i in xrange(len(decoder_inputs) - 1)]


def seq2seq_function(_encoder_inputs, _decoder_inputs):
    return tf.nn.seq2seq.basic_rnn_seq2seq(_encoder_inputs, _decoder_inputs, cell)


outputs, losses = tf.nn.seq2seq.model_with_buckets(encoder_inputs, decoder_inputs, targets,
                                                   target_weights, buckets,
                                                   lambda x, y: seq2seq_function(x, y))

params = tf.trainable_variables()

# encoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, input_size])
# decoder_inputs = tf.placeholder(tf.int32, shape=[batch_size, input_size])

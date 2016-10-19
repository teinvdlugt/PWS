import tensorflow as tf

# [a-z], . , , , !, ?, _

input_size = 31
batch_size = 32
learning_rate = .5
lstm_size = 512
buckets = [(10, 40), (20, 40), ()]

cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)

encoder_inputs = []
decoder_inputs = []
targets = []



# encoder_inputs = tf.placeholder(float, shape=[batch_size, input_size])
# decoder_inputs = tf.placeholder(float, shape=[batch_size, input_size])

outputs, states = tf.nn.seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)

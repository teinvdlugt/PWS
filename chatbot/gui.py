import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify

from . import seq2seq_model
from . import data_utils


app = Flask(__name__)

# TensorFlow flags: you can set the values using command line parameters.
tf.app.flags.DEFINE_boolean("words", True, "True when using the word-based model, False when using chars")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.85,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_boolean("learning_rate_force_reset", False,
                            "Whether to reset the learning rate to the"
                            "parameter or default value, or to read it from the checkpoint (if available)")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")  # Originally 1024
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # Originally 3
tf.app.flags.DEFINE_integer("vocab_size", -1, "Vocabulary size.")  # TODO use 0 for default values
tf.app.flags.DEFINE_boolean("num_samples", -1, "Number of samples for the sampled softmax (0: no sampled softmax)")
tf.app.flags.DEFINE_string("data_dir", "./data/first_dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data/checkpoints-chars", "Directory to store the training checkpoints.")
tf.app.flags.DEFINE_string("train_dialogue", "PWS/data/first_dataset/train.txt", "The dialogue file used for training.")
tf.app.flags.DEFINE_string("test_dialogue", "PWS/data/first_dataset/test.txt", "The dialogue file used for testing.")
tf.app.flags.DEFINE_integer("max_read_train_data", 0,
                            "Limit on the size of training data to read into buckets (0: no limit).")
tf.app.flags.DEFINE_integer("max_read_test_data", 0,
                            "Limit on the size of test data to read into buckets (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 5000,
                            "Amount of training steps to do when executing the TF application")
tf.app.flags.DEFINE_boolean("save_pickles", False, "Whether to save the training and test data, "
                                                   "put into buckets, to disk using np.save")
tf.app.flags.DEFINE_boolean("decode", True,
                            "Set to True for interactive decoding (in stead of training).")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
FLAGS = tf.app.flags.FLAGS

_buckets_chars = [(10, 40), (30, 100), (60, 100), (100, 200)]
_buckets_words = [(5, 10), (10, 15), (20, 25), (40, 50)]


def create_model(session, forward_only):
    """Create seq2seq model and initialize or load parameters in session."""
    # Determine some parameters
    _buckets = _buckets_words if FLAGS.words else _buckets_chars
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

    # Create Seq2SeqModel object
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        num_samples=FLAGS.num_samples,
        forward_only=forward_only,
        dtype=dtype)
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MkDir(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if FLAGS.learning_rate_force_reset:
            session.run(model.learning_rate.assign(FLAGS.learning_rate))
    else:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


@app.route('/')
def index():
    return render_template('chatbot.html')


@app.route('/message', methods=['GET'])
def message():
    return str(request.form['messageContent'])


@app.route('/chat', methods=['POST'])
def chat():
    with app.app_context():
        with tf.Session() as sess:
            # Load parameters.
            model = create_model(sess, True)
            model.batch_size = 1  # We decode one sentence at a time.

            # Load vocabularies.
            vocab, rev_vocab = data_utils.get_vocabulary(FLAGS.data_dir, FLAGS.words, FLAGS.vocab_size)

            # Determine buckets
            _buckets = _buckets_words if FLAGS.words else _buckets_chars
            sentence = message()
            while sentence:
                # Get token-ids for the input sentence.
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab,
                                                             data_utils.basic_word_tokenizer)
                # Which bucket does it belong to?
                bucket_id = min([b for b in xrange(len(_buckets))
                                 if _buckets[b][0] > len(token_ids)])
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    # Print out the network's response to the input.
                return jsonify({'status': 'OK', 'answer': "".join([tf.compat.as_str(rev_vocab[output])
                                                                   for output in outputs])})


if __name__ == "__main__":
    app.run()

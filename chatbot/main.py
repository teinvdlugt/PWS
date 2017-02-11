# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train the chatbot or speak with it.

Running this program with --decode will let you speak with the neural network.
Running without --decode will start the training process or continue training
from the last checkpoint.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from flask import Flask, render_template, request, jsonify

#from . import data_utils
#from . import seq2seq_model

# Flask web interface
web_app = Flask(__name__)

# Let's set some default hyperparameter values for char- and word-chatbot respectively.
# These will be set in main(), if the FLAGS values are equal to -1.
word_default_vocab_size = 2500
char_default_vocab_size = 60
word_default_num_samples = 512
char_default_num_samples = 0

# TensorFlow flags: you can set the values using command line parameters.
tf.app.flags.DEFINE_boolean("words", False, "True when using the word-based model, False when using chars")
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
tf.app.flags.DEFINE_string("data_dir", "./data/os", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data/checkpoints-chars", "Directory to store the training checkpoints.")
tf.app.flags.DEFINE_string("train_dialogue", "PWS/data/os/train.txt", "The dialogue file used for training.")
tf.app.flags.DEFINE_string("test_dialogue", "PWS/data/os/test.txt", "The dialogue file used for testing.")
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

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
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


def train():
    """Train the chatbot."""
    # Decide which buckets to use
    _buckets = _buckets_words if FLAGS.words else _buckets_chars

    # Prepare dialogue data.
    print("Preparing dialogue data in %s" % FLAGS.data_dir)
    train_data, test_data = data_utils.prepare_dialogue_data(FLAGS.words, FLAGS.data_dir, FLAGS.vocab_size,
                                                             _buckets, FLAGS.max_read_train_data,
                                                             FLAGS.max_read_test_data, save=FLAGS.save_pickles)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Compute the sizes of the buckets.
        train_bucket_sizes = [len(train_data[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        print(" Bucket sizes: %s\n Total train size: %d" % (str(train_bucket_sizes), train_total_size))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # File to document losses.
        loss_csv_file = os.path.join(FLAGS.train_dir, "loss_eval.csv")
        if not tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.MkDir(FLAGS.train_dir)
        loss_csv_string = ""

        # This is the training loop.
        avg_step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while current_step < FLAGS.max_training_steps + 1:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_data, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            print("Step %d, loss: %f" % (current_step + 1, step_loss))

            step_time = time.time() - start_time
            avg_step_time += step_time / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # This string will later be put into loss_csv_file
            loss_csv_string += "%f,%d\n" % (step_loss, int(round(step_time * 1000)))

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                avg_step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                    print("Learning rate decayed.")
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                print("Saving checkpoint...")
                checkpoint_file = "chatbot-word.ckpt" if FLAGS.words else "chatbot-char.ckpt"
                checkpoint_path = os.path.join(FLAGS.train_dir, checkpoint_file)
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                avg_step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(test_data[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % bucket_id)
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        test_data, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                # Save the loss_csv_file (This function automatically appends, doesn't overwrite)
                file_io.write_string_to_file(loss_csv_file, loss_csv_string)


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab, rev_vocab = data_utils.get_vocabulary(FLAGS.data_dir, FLAGS.words, FLAGS.vocab_size)

        # Determine buckets
        _buckets = _buckets_words if FLAGS.words else _buckets_chars

        # Decode from standard input.
        #sys.stdout.flush()
        #sentence = sys.stdin.readline()
        chat()
        sentence = chat.message
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
            print("".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


# for testing purposes
def example_decode():
    return "Success"


def self_test():
    """Test the seq2seq model."""
    with tf.Session() as sess:
        print("Self-test for seq2seq chatbot model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99)
        sess.run(tf.global_variables_initializer())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


@web_app.route('/')
def index():
    return render_template('chatbot.html')


@web_app.route('/chat', methods=['POST'])
def chat():
    chat.message = str(request.form['messageContent'])
    # just for testing
    success = example_decode()
    return jsonify({'status': 'OK', 'answer': success})


def main(_):
    # Set word- and char-specific defaults.
    words = FLAGS.words
    if FLAGS.vocab_size == -1:
        FLAGS.__setattr__("vocab_size", word_default_vocab_size if words else char_default_vocab_size)
    if FLAGS.num_samples == -1:
        FLAGS.__setattr__("num_samples", word_default_num_samples if words else char_default_num_samples)

    # Start task according to flags.
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    #tf.app.run()
    web_app.run(debug=True)

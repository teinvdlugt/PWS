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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
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

# from six.moves import xrange  # pylint: disable=redefined-builtin
from chars import seq2seq_model
from chars import data_utils

# Hyperparameters
dtype = tf.float16  # or tf.float32
# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
buckets = [(10, 40), (20, 40), (30, 100), (50, 100)]
size = 1024  # Size of each model layer
num_layers = 1  # Number of layers in the model TODO was originally 3 -> change?
learning_rate = .5
learning_rate_decay_factor = .99
max_gradient_norm = 5.0  # Clip gradients to this norm
batch_size = 32  # Batch size used during training
max_train_data_size = 0
steps_per_checkpoint = 200

train_dir = "/tmp"  # TODO change to a permanent directory in the git repo?
data_dir = "/tmp"

# tf.app.flags.DEFINE_boolean("decode", False,
#                             "Set to True for interactive decoding.")
# tf.app.flags.DEFINE_boolean("self_test", False,
#                             "Run a self-test if this is set to True.")
# tf.app.flags.DEFINE_boolean("use_fp16", False,
#                             "Train using fp16 instead of fp32.")
#
# FLAGS = tf.app.flags.FLAGS

pad_index = 0
go_index = 1
eos_index = 2
special_vocab = ['#', ' ', '~']
vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
         'u', 'v', 'w', 'x', 'y', 'z', '.', ',', '!', '?', ' ']
total_vocab = [special_vocab + vocab]


def read_data(dataset_path, max_size=None):
    """Read dialogue from file, make input and output pairs and put into buckets.

    Args:
      dataset_path: path to a file containing a dialog, consecutive lines being
        said by different people in response to each other
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with open(dataset_path, "r") as f:
        input_sentence = f.readline()
        output_sentence = f.readline()
        count = 0
        while input_sentence and output_sentence:
            count += 1
            if count == max_size:
                break

            input_chars = []
            for i in input_sentence:
                try:
                    input_chars.append(vocab.index(i) + len(special_vocab))
                except ValueError:
                    continue
            output_chars = []
            for i in output_sentence:
                try:
                    output_chars.append(vocab.index(i) + len(special_vocab))
                except ValueError:
                    continue
            output_chars.append(eos_index)

            for bucket_id in xrange(len(buckets)):
                bucket_max_input_size = buckets[bucket_id][0]
                bucket_max_output_size = buckets[bucket_id][1]
                if len(input_chars) < bucket_max_input_size and len(output_chars) < bucket_max_output_size:
                    data_set[bucket_id].append([input_chars, output_chars])
                    break

            input_sentence = f.readline()
            output_sentence = f.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
        buckets,
        size,
        num_layers,
        max_gradient_norm,
        batch_size,
        learning_rate,
        learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Creating model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train the model!"""
    # Prepare dialogue data.
    print("Preparing data in %s" % data_dir)
    # en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
    #     FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)
    dataset_path, eval_dataset_path = data_utils.prepare_data(blahblah)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (num_layers, size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % max_train_data_size)
        dev_set = read_data(eval_dataset_path)
        training_set = read_data(dataset_path, max_train_data_size)
        train_bucket_sizes = [len(training_set[b]) for b in xrange(len(buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size of i-th training bucket. Used to pick random buckets for training.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(  # TODO review get_batch method
                training_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,  # TODO review step method
                                         target_weights, bucket_id, False)
            # To get averages when a checkpoint has arrived, divide by steps_per_checkpoint
            step_time += (time.time() - start_time) / steps_per_checkpoint
            loss += step_loss / steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                print("global step %d learning rate %.4f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(train_dir, "chars.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                step_time, loss = 0.0, 0.0

                # Run evaluations on development set and print their perplexity.
                for bucket_id in xrange(len(buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % bucket_id)
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                        "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()


def decode():
    """Propagate forward and create a response to an input sentence"""
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess,
                             True)  # forward_only is True, because we don't need to backpropagate
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)  # TODO Create function
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(buckets))
                             if buckets[b][0] > len(token_ids)])
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                # Creating dictionary, not list, because there's only one bucket_id, maybe != 0
                {bucket_id: [(token_ids, [])]},
                bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:  # TODO rewrite according to data_utils
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out the response sentence corresponding to outputs.
            print(total_vocab[output] for output in outputs)

            # Read next input line
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural conversational model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel([(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, dtype=tf.float16)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


def main(_):
    # if FLAGS.self_test:
    #     self_test()
    # elif FLAGS.decode:
    #     decode()
    # else:
    #     train()
    pass


if __name__ == "__main__":
    # tf.app.run()
    pass

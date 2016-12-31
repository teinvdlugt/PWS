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

from . import data_utils
from . import seq2seq_model

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(10, 40), (30, 100), (60, 100), (100, 200)]


def create_model(session, forward_only, FLAGS):
    """Create seq2seq model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MkDir(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train(FLAGS):
    """Train the chatbot."""
    # Prepare dialogue data.
    print("Preparing dialogue data in %s" % FLAGS.data_dir)
    train_data, test_data = data_utils.prepare_dialogue_data(FLAGS.data_dir, FLAGS.vocab_size, _buckets,
                                                             FLAGS.max_read_train_data, FLAGS.max_read_test_data,
                                                             save=FLAGS.save_pickles)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, FLAGS)

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
        loss_graph_file = os.path.join(FLAGS.train_dir, "loss_eval.csv")
        if not tf.gfile.Exists(FLAGS.train_dir):
            tf.gfile.MkDir(FLAGS.train_dir)

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

            save_loss_and_time(loss_graph_file, step_loss, step_time)

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
                checkpoint_path = os.path.join(FLAGS.train_dir, "chatbot.ckpt")
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
                sys.stdout.flush()


def save_loss_and_time(filename, loss, step_time):
    _file = tf.gfile.Open(filename, mode='a')
    _file.write("%f,%d\n" % (loss, int(round(step_time * 1000))))
    _file.close()


def decode(FLAGS):
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True, FLAGS)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab_path = os.path.join(FLAGS.data_dir, "chars_vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
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


def self_test(FLAGS):
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


def main(FLAGS):
    if FLAGS.self_test:
        self_test(FLAGS)
    elif FLAGS.decode:
        decode(FLAGS)
    else:
        train(FLAGS)


if __name__ == "__main__":
    tf.app.run()
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

import json
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from . import data_utils
from . import seq2seq_model

# Let's set some default hyperparameter values for char- and word-chatbot respectively.
# These will be set in main(), if the FLAGS values are equal to -1.
word_default_vocab_size = 3500
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
tf.app.flags.DEFINE_integer("vocab_size", -1, "Vocabulary size.")
tf.app.flags.DEFINE_boolean("num_samples", -1, "Number of samples for the sampled softmax (0: no sampled softmax)")
tf.app.flags.DEFINE_string("data_dir", "./data/os", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./data/checkpoints-chars", "Directory to store the training checkpoints.")
tf.app.flags.DEFINE_string("train_dialogue", "PWS/data/os/train.txt", "The dialogue file used for training.")
tf.app.flags.DEFINE_string("test_dialogue", "PWS/data/os/test.txt", "The dialogue file used for testing.")
tf.app.flags.DEFINE_boolean("word_embeddings", False, "Whether to use preset word embeddings.")
tf.app.flags.DEFINE_integer("max_read_train_data", 0,
                            "Limit on the size of training data to read into buckets (0: no limit).")
tf.app.flags.DEFINE_boolean("read_again", False, "")
tf.app.flags.DEFINE_integer("max_read_test_data", 0,
                            "Limit on the size of test data to read into buckets (0: no limit).")
tf.app.flags.DEFINE_integer("start_read_train_data", 0,
                            "Start reading from the training data from this line on.")
tf.app.flags.DEFINE_integer("start_read_test_data", 0,
                            "Start reading from the test data from this line on.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 5000,
                            "Amount of training steps to do when executing the TF application")
tf.app.flags.DEFINE_boolean("save_pickles", False, "Whether to save the training and test data, "
                                                   "put into buckets, to disk using np.save")
tf.app.flags.DEFINE_boolean("decode", False,
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


def create_model(forward_only, word_embeddings_non_trainable=False):
    """Create seq2seq model and initialize or load parameters in session."""
    # Determine some parameters
    _buckets = _buckets_words if FLAGS.words else [_buckets_chars[1]]  # TODO undo
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
        word_embeddings_non_trainable=word_embeddings_non_trainable,
        dtype=dtype)
    return model


def init_model(session, model, embeddings_file=None):
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MkDir(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        if FLAGS.learning_rate_force_reset:
            session.run(model.learning_rate.assign(FLAGS.learning_rate))

    else:
        if FLAGS.decode:
            input("You sure you want to talk to an untrained chatbot? Press Ctrl-C to stop, Return to continue ")
            print("Fine.")

        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())

        if embeddings_file is not None:
            print("Reading the word embeddings from the word2vec file")
            init_word_embeddings(session, embeddings_file)


def init_word_embeddings(session, embeddings_file):
    """Replace the random initialized word-embedding arrays in session with word2vec embeddings
    and make them non-trainable"""
    # Create word embedding array from word2vec file
    vocab_size = FLAGS.vocab_size
    embeddings = []
    with tf.gfile.Open(embeddings_file) as f:
        i = 0
        while i < vocab_size:
            numbers = f.readline().split()
            if len(numbers) > 0:
                embeddings.append([float(n) for n in numbers])
                i += 1
            else:
                break  # Last line of embeddings file is empty

    # Eliminate the random word embeddings and introduce word2vec to the realm of variable scopes.
    # The victims will be:
    # "embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding"
    # "embedding_attention_seq2seq/embedding_attention_decoder/embedding"
    np_embeddings = np.array(embeddings)
    with variable_scope.variable_scope("embedding_attention_seq2seq/RNN/EmbeddingWrapper", reuse=True) as scope:
        embedding = variable_scope.get_variable("embedding")
        session.run(embedding.assign(np_embeddings))
    with variable_scope.variable_scope("embedding_attention_seq2seq/embedding_attention_decoder", reuse=True) as scope:
        embedding = variable_scope.get_variable("embedding")
        session.run(embedding.assign(np_embeddings))


def train():
    """Train the chatbot."""
    # Decide which buckets to use
    _buckets = _buckets_words if FLAGS.words else [_buckets_chars[1]]  # TODO undo

    # For Distributed TensorFlow
    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_info = env.get('cluster')
    cluster_spec = tf.train.ClusterSpec(cluster_info)
    task_info = env.get('task')
    job_name, task_index = task_info['type'], task_info['index']

    device_fn = tf.train.replica_device_setter(
        cluster=cluster_spec,
        worker_device='/job:%s/task:%d' % (job_name, task_index))

    print("Start job:%s, index:%d" % (job_name, task_index))

    server = tf.train.Server(cluster_spec,
                             job_name=job_name, task_index=task_index)

    # Start a parameter server node
    if job_name == 'ps':
        server.join()

    # Start a master/worker node
    if job_name == 'master' or job_name == 'worker':
        is_chief = (job_name == 'master')

        with tf.Graph().as_default() as graph:
            with tf.device(device_fn):

                # Prepare the data
                if FLAGS.word_embeddings:
                    # Get the dialogue data manually.
                    vocab_dir = os.path.join(FLAGS.data_dir, "word2vec")
                    train_ids = os.path.join(vocab_dir, "train_ids%d" % FLAGS.vocab_size)
                    test_ids = os.path.join(vocab_dir, "test_ids%d" % FLAGS.vocab_size)
                    train_data = data_utils.read_data(train_ids, _buckets, FLAGS.max_read_train_data,
                                                      FLAGS.start_read_train_data)
                    test_data = data_utils.read_data(test_ids, _buckets, FLAGS.max_read_test_data,
                                                     FLAGS.start_read_test_data)
                    embeddings_file = os.path.join(vocab_dir, "vocab%d_embeddings" % FLAGS.vocab_size)
                else:
                    # Get dialogue data using the functions in data_utils.py
                    print("Preparing dialogue data in %s" % FLAGS.data_dir)
                    train_data, test_data = data_utils.prepare_dialogue_data(
                        FLAGS.words, FLAGS.data_dir, FLAGS.vocab_size, _buckets,
                        FLAGS.max_read_train_data, FLAGS.max_read_test_data,
                        FLAGS.start_read_train_data, FLAGS.start_read_test_data,
                        read_again=FLAGS.read_again, save=FLAGS.save_pickles)
                    embeddings_file = None

                # Create model.
                print("(%s,%d) Creating %d layers of %d units." % (job_name, task_index, FLAGS.num_layers, FLAGS.size))
                model = create_model(False, embeddings_file is not None)

                # Stuff that used to be in create_model
                if not tf.gfile.Exists(FLAGS.train_dir):
                    tf.gfile.MkDir(FLAGS.train_dir)
                init_op = tf.global_variables_initializer()

                def init_fn(session):
                    if embeddings_file is not None:
                        print("Reading the word embeddings from the word2vec file")
                        init_word_embeddings(session, embeddings_file)

                sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.train_dir, init_op=init_op, init_fn=init_fn,
                                         saver=model.saver, summary_op=None,
                                         global_step=model.global_step, save_model_secs=0)

                with sv.managed_session(server.target) as sess:
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
                    print("(%s,%d) Training begins!" % (job_name, task_index))
                    while model.global_step.eval(sess) < FLAGS.max_training_steps + 1 and not sv.should_stop():
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

                        step_time = time.time() - start_time
                        avg_step_time += step_time / FLAGS.steps_per_checkpoint
                        loss += step_loss / FLAGS.steps_per_checkpoint
                        current_step += 1

                        # This string will later be put into loss_csv_file
                        loss_csv_string += "%f,%d\n" % (step_loss, int(round(step_time * 1000)))

                        print("(%s,%d) Global step %d, loss: %.4f, time: %d" % (job_name, task_index,
                                                                                model.global_step.eval(sess), step_loss,
                                                                                step_time * 1000))

                        # Once in a while, we save checkpoint, print statistics, and run evals.
                        if model.global_step.eval(sess) % FLAGS.steps_per_checkpoint == 0 and is_chief:
                            # Print statistics for the previous epoch.
                            perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                            print("step %d, global step %d, learning rate %.4f, step-time %.2f, average loss %.4f, "
                                  "perplexity %.2f" % (
                                      current_step, model.global_step.eval(sess), model.learning_rate.eval(sess),
                                      avg_step_time, loss, perplexity))

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

                            # # Run evals on development set and print their perplexity.
                            # for bucket_id in xrange(len(_buckets)):
                            #     if len(test_data[bucket_id]) == 0:
                            #         print("  eval: empty bucket %d" % bucket_id)
                            #         continue
                            #     encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            #         test_data, bucket_id)
                            #     _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                            #                                  target_weights, bucket_id, True)
                            #     eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                            #         "inf")
                            #     print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

                            # Save the loss_csv_file
                            mode = "a" if tf.gfile.Exists(loss_csv_file) else "w"
                            with tf.gfile.Open(loss_csv_file, mode=mode) as f:
                                f.write(loss_csv_string)
                sv.stop()


def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        vocab, rev_vocab = data_utils.get_vocabulary(FLAGS.data_dir, FLAGS.words,
                                                     FLAGS.word_embeddings, FLAGS.vocab_size)

        # Determine buckets
        _buckets = _buckets_words if FLAGS.words else _buckets_chars

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
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
            join = " " if FLAGS.words else ""
            print(join.join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


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


def main(_):
    # Set word- and char-specific defaults.
    words = FLAGS.words
    if FLAGS.vocab_size == -1:
        FLAGS.__setattr__("vocab_size", word_default_vocab_size if words else char_default_vocab_size)
    if FLAGS.num_samples == -1:
        FLAGS.__setattr__("num_samples", word_default_num_samples if words else char_default_num_samples)

    if FLAGS.words:
        data_utils._START_VOCAB = data_utils.START_VOCAB_WORD

    # Check compatibility with word2vec file
    if FLAGS.word_embeddings:
        # For now, assume the embedding size is 300. If variable, reprogram.
        print("Setting LSTM size to 300 to conform to the word2vec file")
        FLAGS.__setattr__("size", 300)

    # The names of the embeddings are:
    # "embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding"
    # "embedding_attention_seq2seq/embedding_attention_decoder/embedding"

    # FLAGS.__setattr__("train_dir", "/tmp/train")
    # session = tf.Session()
    # model = create_model(session, True)
    # with variable_scope.variable_scope("embedding_attention_seq2seq/RNN/EmbeddingWrapper", reuse=True) as scope:
    #     embedding = variable_scope.get_variable("embedding")
    #     print(embedding)
    #     array = embedding.eval(session)
    #     print(array)
    #     new_array = np.zeros((array.shape[0], array.shape[1]))
    #     session.run(embedding.assign(new_array))
    #     print(embedding.eval(session))
    #     print(tf.trainable_variables())
    #     array = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    #     print([var.name for var in array])
    #     array.remove(embedding)
    #     print([var.name for var in tf.trainable_variables()])
    #
    # exit(0)

    # Start task according to flags.
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()

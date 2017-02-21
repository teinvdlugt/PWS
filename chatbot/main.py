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
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
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
tf.app.flags.DEFINE_string("tensorboard_logdir", None, "Directory to store TensorBoard summaries. "
                                                       "If None (default), it becomes the same as train_dir.")
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
tf.app.flags.DEFINE_integer("steps_per_eval", 10,
                            "After how many training steps an evaluation will be done")
tf.app.flags.DEFINE_integer("max_training_steps", 100000,
                            "Amount of training steps to do IN TOTAL, so including the steps "
                            "already done in previous training sessions")
tf.app.flags.DEFINE_boolean("save_pickles", False, "Whether to save the training and test data, "
                                                   "put into buckets, to disk using np.save")
tf.app.flags.DEFINE_boolean("adagrad", False, "Use AdaGrad as optimization algorithm")
tf.app.flags.DEFINE_boolean("adadelta", False, "Use AdaDelta as optimization algorithm")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding (in stead of training).")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_boolean("distributed", False, "Run in distributed mode or not")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
BUCKETS_CHARS = [(10, 40), (30, 100), (60, 100), (100, 200)]
BUCKETS_WORDS = [(5, 10), (10, 15), (20, 25), (40, 50)]
buckets = BUCKETS_WORDS if FLAGS.words else BUCKETS_CHARS


def create_model(forward_only):
    """Create a Seq2SeqModel object.

    Args:
        forward_only: e.g. if False, gradients for SGD will be created too
    Returns:
        A Seq2SeqModel object.
    """
    # Determine data type
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    word_embeddings_non_trainable = FLAGS.word_embeddings

    # Create Seq2SeqModel object
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.vocab_size,
        buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        num_samples=FLAGS.num_samples,
        forward_only=forward_only,
        word2vec=word_embeddings_non_trainable,
        dtype=dtype,
        adagrad=FLAGS.adagrad,
        adadelta=FLAGS.adadelta)
    return model


def init_model(session, model):
    """Load the variables from a checkpoint, or initialize them using tf.global_variables_initializer()"""
    # If there is a checkpoint, load it
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MkDir(FLAGS.train_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)

    # Else initialize the variables
    else:
        if FLAGS.decode:
            input("You sure you want to talk to an untrained chatbot? Press Ctrl-C to stop, Return to continue ")
            print("Fine.")

        print("Creating model with fresh parameters.")
        session.run(tf.global_variables_initializer())


def after_init(session, model, embeddings_file):
    """Optionally loads word embeddings from the embeddings file.
    Resets the learning rate if FLAGS.learning_rate_force_reset is set.

    This function is not included in init_model() because of Distributed TensorFlow.
    The Supervisor of Distributed TensorFlow does initialization itself, after which
    this function is called.

    NOTE: Using this function is only useful when creating a model which is going
    to be trained.
    """
    if embeddings_file is not None:
        print("Reading the word embeddings from the word2vec file")
        init_word_embeddings(session, model, embeddings_file)
    if FLAGS.learning_rate_force_reset:
        session.run(model.learning_rate_assign_op,
                    feed_dict={model.learning_rate_placeholder: FLAGS.learning_rate})


def init_word_embeddings(session, model, embeddings_file):
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
    feed_dict = {model.word2vec_placeholder: np_embeddings}
    session.run(model.word2vec_assign_encoder_op, feed_dict=feed_dict)
    session.run(model.word2vec_assign_decoder_op, feed_dict=feed_dict)


def calculate_perplexity(loss):
    """Simply mathematical equation to calculate word perplexity from loss.
    See this paper for more info on word perplexity: https://arxiv.org/abs/1507.04808"""
    return math.exp(float(loss)) if loss < 300 else float("inf")


def train_distributed():
    """Train model with Distributed TensorFlow."""
    # Distributed stuff learnt from this repo: https://github.com/GoogleCloudPlatform/cloudml-dist-
    # mnist-example/blob/master/trainer/task.py

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

        with tf.Graph().as_default() as graph:  # TODO necessary?
            with tf.device(device_fn):
                # Prepare the data
                train_data, test_data, embeddings_file = prepare_data()

                # Create the model
                print("(%s,%d) Creating %d layers of %d units." %
                      (job_name, task_index, FLAGS.num_layers, FLAGS.size))
                model = create_model(False)

                # Create train_dir
                if is_chief:
                    if not tf.gfile.Exists(FLAGS.train_dir):
                        tf.gfile.MkDir(FLAGS.train_dir)

                # TensorBoard summaries
                (test_loss, test_perplexity, bucket_loss_placeholders,
                 bucket_perplexity_placeholders, summary, summary_writer) = create_summary_objects(graph)

                # Create supervisor
                init_op = tf.global_variables_initializer()

                # Create Supervisor. Disabling checkpoints and summaries, because we do that manually
                sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.train_dir, init_op=init_op,
                                         init_fn=lambda session: after_init(session, model, embeddings_file),
                                         saver=model.saver, global_step=model.global_step,
                                         save_model_secs=0, save_summaries_secs=0, summary_op=None,
                                         summary_writer=None)

                with sv.managed_session(server.target) as sess:
                    train(sess, model, train_data, test_data, summary, summary_writer, test_loss,
                          test_perplexity, bucket_loss_placeholders, bucket_perplexity_placeholders,
                          is_chief, job_name, task_index, sv.should_stop)
                sv.stop()


def train_not_distributed():
    """Train a model in non-distributed TensorFlow mode."""
    with tf.Graph().as_default() as graph:
        # Prepare the data
        train_data, test_data, embeddings_file = prepare_data()

        # Create model
        model = create_model(False)

        # Create summaries and SummaryWriter
        (test_loss, test_perplexity, bucket_loss_placeholders,
         bucket_perplexity_placeholders, summary, summary_writer) = create_summary_objects(graph)

        with tf.Session() as sess:
            init_model(sess, model)
            after_init(sess, model, embeddings_file)

            train(sess, model, train_data, test_data, summary, summary_writer, test_loss,
                  test_perplexity, bucket_loss_placeholders, bucket_perplexity_placeholders)


def prepare_data():
    """Prepare the data using functions from data_utils.

    Returns:
        A tuple of 3 elements:
            (1) Training data in array form
            (2) Test data in array form
            (3) Path to a file containing the word embeddings which should be loaded
                into the model, if FLAGS.word_embeddings is set
    """
    if FLAGS.word_embeddings:
        # Get the dialogue data manually.
        vocab_dir = os.path.join(FLAGS.data_dir, "word2vec")
        train_ids = os.path.join(vocab_dir, "train_ids%d" % FLAGS.vocab_size)
        test_ids = os.path.join(vocab_dir, "test_ids%d" % FLAGS.vocab_size)
        train_data = data_utils.read_data(train_ids, buckets, FLAGS.max_read_train_data,
                                          FLAGS.start_read_train_data)
        test_data = data_utils.read_data(test_ids, buckets, FLAGS.max_read_test_data,
                                         FLAGS.start_read_test_data)
        embeddings_file = os.path.join(vocab_dir, "vocab%d_embeddings" % FLAGS.vocab_size)
    else:
        # Get dialogue data using the functions in data_utils.py
        print("Preparing dialogue data in %s" % FLAGS.data_dir)
        train_data, test_data = data_utils.prepare_dialogue_data(
            FLAGS.words, FLAGS.data_dir, FLAGS.vocab_size, buckets,
            FLAGS.max_read_train_data, FLAGS.max_read_test_data,
            FLAGS.start_read_train_data, FLAGS.start_read_test_data,
            read_again=FLAGS.read_again, save=FLAGS.save_pickles)
        embeddings_file = None
    return train_data, test_data, embeddings_file


def create_summary_objects(graph=None):
    """Creates summary ops, placeholders for them and a summary writer.

    Args:
        graph: The optional graph to pass to the summary_writer to visualize it in TensorBoard
    Returns:
        A tuple containing six elements:
            (1) A placeholder for the overall loss of the network
            (2) A placeholder for the overall perplexity of the network
            (3) An array containing placeholders for the loss of each bucket
            (4) An array containing placeholders for the perplexity of each bucket
            (5) A Summary op, created by merging all the summaries
            (6) A SummaryWriter object, to save the summaries with
    """
    test_loss = tf.placeholder(tf.float32, [])
    test_perplexity = tf.placeholder(tf.float32, [])
    tf.summary.scalar('Loss_overall', test_loss)
    tf.summary.scalar('Perplexity_overall', test_perplexity)
    # Do it for each individual bucket as well
    bucket_loss_placeholders = []
    bucket_perplexity_placeholders = []
    for bucket_id in range(0, len(buckets)):
        bucket_loss = tf.placeholder(tf.float32, [])
        bucket_perplexity = tf.placeholder(tf.float32, [])
        tf.summary.scalar('Loss_bucket%d' % bucket_id, bucket_loss)
        tf.summary.scalar('Perplexity_bucket%d' % bucket_id, bucket_perplexity)
        bucket_loss_placeholders.append(bucket_loss)
        bucket_perplexity_placeholders.append(bucket_perplexity)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.tensorboard_logdir, graph)
    return (test_loss, test_perplexity, bucket_loss_placeholders,
            bucket_perplexity_placeholders, summary, summary_writer)


def train(sess, model, train_data, test_data, summary, summary_writer, test_loss, test_perplexity,
          bucket_loss_placeholders, bucket_perplexity_placeholders, is_chief=True, job_name=None,
          task_index=0, should_stop_fn=lambda: False):
    """Train the model.

    Args:
        sess: A tensorflow `Session` object to be trained
        model: The Seq2SeqModel to be trained
        train_data: The training data in array form
        test_data: The test data in array form
        summary: Summary op used to save summaries
        summary_writer: `SummaryWriter` object
        test_loss: Placeholder for the loss
        test_perplexity: Placeholder for the perplexity
        bucket_loss_placeholders: List of Placeholders for loss of each bucket
        bucket_perplexity_placeholders: List of Placeholders for perplexity of each bucket
        is_chief: Optional, whether this thread is the master (if Distributed TF)
        job_name: Optional, the name of this job ('master' or 'worker') (if Distributed TF)
        task_index: Optional, the index of the task (if Distributed TF)
        should_stop_fn: Optional, a function indicating to stop or proceed training
         (if Distributed TF; sv.should_stop())
    """

    # Compute the sizes of the buckets.
    train_bucket_sizes = [len(train_data[b]) for b in xrange(len(buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    print(" Bucket sizes: %s\n Total train size: %d" %
          (str(train_bucket_sizes), train_total_size))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    global_step = model.global_step.eval(sess)
    # These variables will only be used when is_chief
    num_evals = global_step // FLAGS.steps_per_eval
    previous_losses = []
    avg_loss = None
    num_checkpoints = (global_step // FLAGS.steps_per_checkpoint) + 1  # Will be used by the first worker

    # This is the training loop.
    print("(%s,%d) Training begins!" % (job_name, task_index))
    while model.global_step.eval(sess) < FLAGS.max_training_steps + 1 and not should_stop_fn():
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_data, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, False)

        # Evaluation time! When distributed, the first worker does this (the master is busy
        # saving checkpoints). When not distributed (job_name is None), this is also done.
        if ((not is_chief and task_index == 0) or job_name is None) \
                and model.global_step.eval(sess) > FLAGS.steps_per_eval * num_evals:

            # Run evals on test set and calculate their perplexity.
            feed_dict = {}  # Feed dict for the summary writer
            bucket_losses = []
            bucket_perplexities = []
            for bucket_id in xrange(len(buckets)):
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_data, bucket_id)
                _, eval_loss, _ = model.step(sess, encoder_inputs,
                                             decoder_inputs, target_weights, bucket_id, True)
                eval_ppx = calculate_perplexity(eval_loss)
                bucket_losses.append(eval_loss)
                bucket_perplexities.append(eval_ppx)
                feed_dict[bucket_loss_placeholders[bucket_id]] = eval_loss
                feed_dict[bucket_perplexity_placeholders[bucket_id]] = eval_ppx
            avg_loss = sum([bucket_losses[i] * train_bucket_sizes[i] / train_total_size
                            for i in range(0, len(buckets))])
            avg_perplexity = calculate_perplexity(avg_loss)
            feed_dict[test_loss] = avg_loss
            feed_dict[test_perplexity] = avg_perplexity

            # Save the summaries
            computed_summary, global_step = sess.run([summary, model.global_step], feed_dict=feed_dict)
            summary_writer.add_summary(computed_summary, global_step)

            # Print evals to screen
            print("Summary saved. Step: %d, loss: %.4f, perplexity: %.4f" % (global_step, avg_loss, avg_perplexity))

            previous_losses.append(avg_loss)
            num_evals += 1

        # Checkpoint time! When distributed, this is done by the chief.
        # When not distributed (job_name is None), this is also done.
        if (is_chief or job_name is None) \
                and model.global_step.eval(sess) > FLAGS.steps_per_checkpoint * num_checkpoints:

            # If SGD, decrease learning rate if no improvement was seen over last 3 checkpoint times.
            last_n_evals = 3 * FLAGS.steps_per_checkpoint / FLAGS.steps_per_eval
            if not FLAGS.adagrad and not FLAGS.adadelta and \
                    avg_loss and len(previous_losses) >= last_n_evals \
                    and avg_loss > max(previous_losses[-last_n_evals:]):
                sess.run(model.learning_rate_decay_op)
                print("Learning rate decayed.")

            # Save checkpoint.
            print("Saving checkpoint...")
            checkpoint_file = "chatbot-word.ckpt" if FLAGS.words else "chatbot-char.ckpt"
            checkpoint_path = os.path.join(FLAGS.train_dir, checkpoint_file)
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print("Checkpoint saved!")

            num_checkpoints += 1


def decode():
    """Have a conversation with the chatbot.
    This will create a (forward-only) Seq2SeqModel and initialize it using the checkpoint
    file in FLAGS.train_dir, if available. This doesn't work with Distributed TensorFlow.
    """
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(True)
        model.batch_size = 1  # We decode one sentence at a time.
        init_model(sess, model)

        # Load vocabularies.
        vocab, rev_vocab = data_utils.get_vocabulary(FLAGS.data_dir, FLAGS.words,
                                                     FLAGS.word_embeddings, FLAGS.vocab_size)

        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab,
                                                         data_utils.basic_word_tokenizer)
            # Which bucket does it belong to?
            bucket_id = min([b for b in xrange(len(buckets))
                             if buckets[b][0] > len(token_ids)])
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
    """Test the seq2seq model using fake data."""
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
    """Executed by tf.app.run() and main function of this file."""
    # Set FLAGS defaults.
    words = FLAGS.words
    if FLAGS.vocab_size == -1:
        FLAGS.__setattr__("vocab_size", word_default_vocab_size if words else char_default_vocab_size)
    if FLAGS.num_samples == -1:
        FLAGS.__setattr__("num_samples", word_default_num_samples if words else char_default_num_samples)
    if FLAGS.tensorboard_logdir is None:
        FLAGS.__setattr__("tensorboard_logdir", FLAGS.train_dir)

    if FLAGS.words:
        data_utils._START_VOCAB = data_utils.START_VOCAB_WORD

    # Check compatibility with word2vec file
    if FLAGS.word_embeddings:
        # For now, assume the embedding size is 300. If variable, reprogram.
        print("Setting LSTM size to 300 to conform to the word2vec file")
        FLAGS.__setattr__("size", 300)

    # Start task according to flags.
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train_distributed() if FLAGS.distributed else train_not_distributed()


if __name__ == "__main__":
    tf.app.run()

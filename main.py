from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_bool("words", False, "True when using the word-based model, False when using chars")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")  # Originally 1024
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")  # Originally 3
tf.app.flags.DEFINE_integer("vocab_size", 60, "Vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "PWS/data/os", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "PWS/data/checkpoints-chars", "Directory to store the training checkpoints.")
tf.app.flags.DEFINE_string("train_dialogue", "PWS/data/os/train.txt", "The dialogue file used for training.")
tf.app.flags.DEFINE_string("test_dialogue", "PWS/data/os/test.txt", "The dialogue file used for testing.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.words:
        from .embedded_chars import chars
        chars.main(FLAGS)


if __name__ == "__main__":
    tf.app.run()
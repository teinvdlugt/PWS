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

"""Utilities for tokenizing and creating vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile

from data import opensubtitles_util

# Special vocabulary symbols - we always put them at the start.
_PAD = b"#"
_GO = b">"
_EOS = b"<"
_UNK = b"~"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_character_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens, being characters in this case."""
    return [c for c in sentence]


def create_vocabulary(vocabulary_path, data_file, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one utterance per line. Each utterance is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Vocabulary file will be written in binary format.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_file: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_file))
        vocab = {}
        with gfile.GFile(data_file, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 10000 == 0:
                    print("  processing line %d" % counter)
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_character_tokenizer(line)
                # Remove newline
                tokens = tokens[:-1]
                for c in tokens:
                    char = _DIGIT_RE.sub(b"0", c) if normalize_digits else c
                    if char in vocab:
                        vocab[char] += 1
                    elif char not in _START_VOCAB:
                        vocab[char] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size - 1]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
      e
      o
    will result in a vocabulary {"e": 0, "o": 1}, and this function will
    also return the reversed-vocabulary ["e", "o"].

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.rstrip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "hello" may become tokenized into
    ["h", "e", "l", "l", "o"] and with vocabulary {"h": 1, "e": 2,
    "l": 4, "o": 7"} this function will return [1, 2, 4, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
        This shouldn't contain a newline at the end.
      vocabulary: a dictionary mapping tokens to integers.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_character_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      tokenizer: a function to use to tokenize each sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    # Remove newline
                    line = line[:-1]
                    counter += 1
                    if counter % 10000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_dialogue_data(data_dir, vocab_size, tokenizer=None):
    """From the dialogue files, create vocabularies and tokenize data in data_dir.

    Args:
        data_dir: directory in which the data and vocab will be stored.
        vocab_size: maximum size of the vocab to create and use.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.

    Returns:
        A tuple of 3 elements:
            (1) path to the token-ids for training dataset,
            (2) path to the token-ids for testing dataset,
            (3) path to the vocabulary file.
    """
    train_file, test_file = opensubtitles_util.get_data(data_dir)

    # Create vocab file
    vocab_path = os.path.join(data_dir, "chars_vocab%d" % vocab_size)
    create_vocabulary(vocab_path, test_file, vocab_size, tokenizer)

    # Create token ids for the training data
    train_ids_path = os.path.join(data_dir, "chars_train_ids%d" % vocab_size)
    data_to_token_ids(train_file, train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    test_ids_path = os.path.join(data_dir, "chars_test_ids%d" % vocab_size)
    data_to_token_ids(test_file, test_ids_path, vocab_path, tokenizer)

    return train_ids_path, test_ids_path, vocab_path

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
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from chatbot.data_utils import opensubtitles_util

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


def maybe_create_vocabulary(vocabulary_path, data_file, max_vocabulary_size,
                            tokenizer=None, all_lowercase=True, normalize_digits=True):
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
      all_lowercase: Boolean; if true, all characters will be made lowercase.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_file))
        vocab = {}
        with gfile.GFile(data_file, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)

                if all_lowercase:
                    line = line.lower()

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
                          tokenizer=None, all_lowercase=True, normalize_digits=True):
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
      all_lowercase: Boolean; if true, sentence will be converted to lowercase first.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """

    if all_lowercase:
        sentence = sentence.lower()

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_character_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def maybe_data_to_token_ids(data_path, target_path, vocabulary_path,
                            tokenizer=None, all_lowercase=True, normalize_digits=True):
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
      all_lowercase: Boolean; if true, all text will be converted to lowercase.
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
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, all_lowercase, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def read_data(dialogue_file, buckets, max_lines=None):
    """Read data from a dialogue file and put it into buckets.

    Args:
        dialogue_file: a file containing text converted to token-ids.
        buckets: an array containing the sizes of the buckets, in which to put the data
        max_lines: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (input, output) pairs read from the provided data file that fit
        into the n-th bucket, i.e., such that len(input) < _buckets[n][0] and
        len(output) < _buckets[n][1]; input and output are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(dialogue_file, 'r') as f:
        input_sentence = f.readline()
        output_sentence = f.readline()
        count = 0
        while input_sentence and output_sentence and (not max_lines or count < max_lines):
            count += 1
            if count % 100000 == 0:
                print("  reading data line %d" % count)
                sys.stdout.flush()

            input_sentence_ids = [int(x) for x in input_sentence.split()]
            output_sentence_ids = [int(x) for x in output_sentence.split()]
            output_sentence_ids.append(EOS_ID)

            for bucket_id, (input_size, output_size) in enumerate(buckets):
                if len(input_sentence_ids) < input_size and len(output_sentence_ids) < output_size:
                    data_set[bucket_id].append([input_sentence_ids, output_sentence_ids])
                    break

            input_sentence = f.readline()
            output_sentence = f.readline()
    return data_set


def get_encoded_data(data_dir, vocab_size, tokenizer=None):
    """Get the paths to the files containing the training and test data in id-form.
    Make those files, in the case that they are not already available, using the plain text data.
    Download those plain text data files if needed.

    By 'encoded', I mean that the data doesn't consist of human-readable characters and words, but of
    numbers which represent the index of those characters or words in the vocabulary.

    Args:
        data_dir: The directory where the data (plain text and id-form) should be or is stored.
        vocab_size: The maximum size of the vocabulary, used when creating a new vocabulary is necessary.
        tokenizer: The tokenizer to tokenize the plain text, before creating a vocabulary and putting the data
         into id-form. If None, basic_character_tokenizer is used.

    Returns:
        A tuple containing the paths to the 1) encoded training data
    """
    train_ids_path = os.path.join(data_dir, "chars_train_ids%d" % vocab_size)
    test_ids_path = os.path.join(data_dir, "chars_test_ids%d" % vocab_size)
    vocab_path = os.path.join(data_dir, "chars_vocab%d" % vocab_size)

    if not (os.path.exists(train_ids_path) and os.path.exists(test_ids_path)):
        if vocab_size == 60:
            # I have already put a tokenized version of the dataset online with vocab=60, so better download that
            print("Downloading already vocabularized data files with vocab_size=60")
            train_ids_path, test_ids_path, vocab_path = opensubtitles_util.get_encoded_data(data_dir)
        else:
            print("Downloading plain text data set")
            train_file, test_file = opensubtitles_util.get_data(data_dir)
            print("Tokenizing and vocabularizing data sets")
            maybe_create_vocabulary(vocab_path, test_file, vocab_size, tokenizer)
            maybe_data_to_token_ids(train_file, train_ids_path, vocab_path, tokenizer)
            maybe_data_to_token_ids(test_file, test_ids_path, vocab_path, tokenizer)

    return train_ids_path, test_ids_path


def prepare_dialogue_data(data_dir, vocab_size, buckets, max_read_train_data=0, max_read_test_data=0,
                          read_again=False, save=True, tokenizer=None):
    """From the dialogue files, create vocabularies and tokenize data in data_dir.

    Args:
        data_dir: directory in which the data and vocab will be stored.
        buckets: an array containing the sizes of the buckets, in which to put the data
        max_read_train_data: maximum amount of lines of training data to be read into buckets,
         if data is going to be put into buckets again (not if the data is already in
         buckets and read from a np.save file)
        max_read_test_data: maximum amount of lines of test data to be read into buckets,
         if data is going to be put into buckets again (not if the data is already in
         buckets and read from a np.save file)
        read_again: Whether to read the data into buckets again (True) or to load from an np.save file,
         if available
        save: True if you want to save the read-again data and thereby replace the old np.save file
        vocab_size: maximum size of the vocab to create and use.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.

    Returns:
        A tuple of 2 elements:
            (1) (numpy-)array containing the training data in buckets;
            (2) (numpy-)array containing the test data in buckets.
    """
    train_ids_npsaved_path = os.path.join(data_dir, "chars_train_ids%d_array.npy" % vocab_size)
    test_ids_npsaved_path = os.path.join(data_dir, "chars_test_ids%d_array.npy" % vocab_size)

    # Get train data array
    if read_again or not os.path.exists(train_ids_npsaved_path):
        print(train_ids_npsaved_path)
        train_ids_path, _ = get_encoded_data(data_dir, vocab_size, tokenizer)
        print("Reading training data into buckets, limit: %d" % max_read_train_data)
        train_ids_array = read_data(train_ids_path, buckets, max_read_train_data)
        if save:
            print("Saving training data arrays to numpy files")
            np.save(train_ids_npsaved_path, train_ids_array)
    else:
        print("Loading training data arrays from numpy files")
        train_ids_array = np.load(train_ids_npsaved_path)

    # Get test data array
    if read_again or not os.path.exists(test_ids_npsaved_path):
        _, test_ids_path = get_encoded_data(data_dir, vocab_size, tokenizer)
        print("Reading test data into buckets, limit: %d" % max_read_test_data)
        test_ids_array = read_data(test_ids_path, buckets, max_read_test_data)
        if save:
            print("Saving test data arrays to numpy files")
            np.save(test_ids_npsaved_path, test_ids_array)
    else:
        print("Loading test data arrays from numpy files")
        test_ids_array = np.load(test_ids_npsaved_path)

    return train_ids_array, test_ids_array

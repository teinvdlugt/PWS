import re

import numpy as np
import tensorflow as tf
from .. import data_utils

vocab_size = 3500
vocab_text_path = "./data/os/test.txt"
max_read_text_for_vocab = 0  # Zero for no limit
vocab_path = "./data/os/word2vec/vocab%d" % vocab_size
new_embedding_path = "./data/os/word2vec/embeddings%d" % vocab_size
word2vec_file_path = "./data/word_embeddings/word2vec10000"
train_path = "./data/os/train.txt"
test_path = "./data/os/test.txt"
train_ids_path = "./data/os/word2vec/train_ids%d" % vocab_size
test_ids_path = "./data/os/word2vec/test_ids%d" % vocab_size

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_START_VOCAB = ["_PAD", "_GO", "_EOS", "_UNK"]


def word2vec_tokenizer(sentence):
    """Tokenizes a sentence conforming to the word2vec method."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        if re.match(r"\d\d", space_separated_fragment):
            re.sub(r"\d", "#", space_separated_fragment)
            words.append(space_separated_fragment)
            continue

        tokens = _WORD_SPLIT.split(space_separated_fragment)
        if '\'' in tokens:
            index = tokens.index('\'')
            if index != len(tokens) - 1 and tokens[index + 1] in ("re", "ve", "m", "ll", "d"):
                piece_of_text = '\'' + tokens[index + 1]
                tokens.pop(index)
                tokens.pop(index)
                tokens.insert(index + 1, piece_of_text)

        words.extend(tokens)
    # Do some cleaning up of punctuation marks, because word2vec doesn't support them
    words = [w for w in words if w not in ('?', '!', ':', ';', '"', '\'', '.', ',', '[', ']', '(', ')')]
    return [w for w in words if w]


def maybe_create_vocab_and_embeddings_from_word2vec():
    if not tf.gfile.Exists(vocab_path):
        print("Creating vocabulary from %s and %s" % (vocab_text_path, word2vec_file_path))

        print("Getting the vocab and embeddings from the word2vec file")
        word2vec = {}
        with tf.gfile.Open(word2vec_file_path) as f:
            for line in f:
                split = line.split(" ")
                word2vec[split[0]] = split[1:]

        # The array to store our selection of the embeddings from word2vec_embeddings in:
        embeddings = {}

        vocab = {}
        with tf.gfile.Open(vocab_text_path, mode='rb') as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                if counter >= max_read_text_for_vocab and not max_read_text_for_vocab == 0:
                    print("Maximum exceeded")
                    break

                line = tf.compat.as_bytes(line)
                tokens = word2vec_tokenizer(line)
                # Remove newline
                tokens = tokens[:-1]
                for t in tokens:
                    if t in vocab:
                        vocab[t] += 1
                    elif t not in _START_VOCAB:
                        # Add the token to the vocab, if it is present in the word2vec file.
                        try:
                            embeddings[t] = word2vec[t]
                            vocab[t] = 1
                        except KeyError:
                            # The token was not in the word2vec vocab.
                            pass
        # Create embeddings for _START_VOCAB
        _PAD_EMBEDDING = np.zeros(300)
        _GO_EMBEDDING = np.ones(300)
        _EOS_EMBEDDING = np.empty(300)
        _EOS_EMBEDDING.fill(-1)
        _UNK_EMBEDDING = np.empty(300)
        _UNK_EMBEDDING.fill(.5)

        sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
        sorted_embeddings = []
        for x in sorted_vocab:
            sorted_embeddings.append(embeddings[x])
        vocab_list = _START_VOCAB + sorted_vocab
        embedding_list = [_PAD_EMBEDDING, _GO_EMBEDDING, _EOS_EMBEDDING, _UNK_EMBEDDING] + sorted_embeddings

        # Chop to vocab size
        if len(vocab_list) > vocab_size:
            vocab_list = vocab_list[:vocab_size]
            embedding_list = embedding_list[:vocab_size]

        # Write vocab and embedding files
        with tf.gfile.Open(vocab_path, mode="wb") as vocab_f:
            for w in vocab_list:
                vocab_f.write(w + b"\n")
        with tf.gfile.Open(vocab_path + "_embeddings", mode="wb") as embedding_f:
            for embedding in embedding_list:
                str_to_write = ""
                for i in embedding:
                    str_to_write += str(i) + " "
                embedding_f.write(str_to_write.strip() + "\n")


maybe_create_vocab_and_embeddings_from_word2vec()
data_utils.maybe_data_to_token_ids(train_path, train_ids_path, vocab_path, word2vec_tokenizer, False, False)
data_utils.maybe_data_to_token_ids(test_path, test_ids_path, vocab_path, word2vec_tokenizer, False, False)

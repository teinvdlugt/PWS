# Create a vocab file and pickle file from the word2vec file

import tensorflow as tf
import numpy as np

word2vec_file = "./data/word_embeddings/word2vec10000"
vocab_file = "./data/word_embeddings/word2vec_vocab3500"
pickle_file = "./data/word_embeddings/word2vec_array3500"

array = []

with tf.gfile.Open(word2vec_file, 'r') as f:
    # First line in word2vec file is metadata
    f.readline()
    with tf.gfile.Open(vocab_file, 'w') as vocab_f:
        i = 0
        while i < 3500:
            line_array = f.readline().split(" ")
            vocab_f.write(line_array[0] + "\n")
            array.append(line_array[1:])
            i += 1
            if i % 1000 == 0:
                print("At line %d" % i)
np.save(pickle_file, array)

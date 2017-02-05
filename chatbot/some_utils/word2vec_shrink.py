# Create a smaller version of word2vec, only the 10000 most-used words.
import tensorflow as tf

vocab_size = 50000
word2vec_file = "./data/word_embeddings/word2vec"
small_file = "./data/word_embeddings/word2vec%d" % vocab_size

with tf.gfile.Open(word2vec_file, 'r') as f:
    with tf.gfile.Open(small_file, 'w') as new_f:
        # First line contains "{vocab_size} {embedding_size}"
        embedding_size = int(f.readline().split(" ")[1])
        new_f.write("%d %d\n" % (vocab_size, embedding_size))

        # After that, each line contains one word and its embedding
        i = 0
        while i < vocab_size:
            new_f.write(f.readline())
            i += 1

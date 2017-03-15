import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from . import data_utils
from . import seq2seq_model
from .some_utils import word2vec_create

app = Flask(__name__, static_url_path='/static')


class CheckpointConfig:
    def __init__(self, ckpt_path, vocab_path, vocab_size, size,
                 num_layers, num_samples, buckets, tokenizer, join):
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.vocab_size = vocab_size
        self.size = size
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.buckets = buckets
        self.tokenizer = tokenizer
        self.join = join


checkpoint1 = CheckpointConfig("./gui-demo/checkpoints1", "./gui-demo/char-vocab60",
                               60, 256, 2, 0, [(10, 40), (30, 100), (60, 100), (100, 200)],
                               data_utils.basic_character_tokenizer, "")
checkpoint2 = CheckpointConfig("./gui-demo/checkpoints2", "./gui-demo/char-vocab60",
                               60, 256, 2, 0, [(10, 40), (30, 100), (60, 100), (100, 200)],
                               data_utils.basic_character_tokenizer, "")
checkpoint3 = CheckpointConfig("./gui-demo/checkpoints3", "./gui-demo/word2vec-vocab7000",
                               7000, 300, 3, 512, [(5, 10), (10, 15), (20, 25), (40, 50)],
                               word2vec_create.word2vec_tokenizer, " ")

checkpoint = checkpoint1  # Choose here which checkpoint to use

session = None
model = None
vocab = None
rev_vocab = None


def init_session_model_vocab():
    # Create session
    _session = tf.Session()

    # Create model
    _model = seq2seq_model.Seq2SeqModel(checkpoint.vocab_size, checkpoint.buckets,
                                        checkpoint.size, checkpoint.num_layers, 0,
                                        1,  # We decode one sentence at a time.
                                        0, 0, num_samples=checkpoint.num_samples, forward_only=True)

    # Load the checkpoint file
    ckpt = tf.train.get_checkpoint_state(checkpoint.ckpt_path)
    print(ckpt is None)
    print(ckpt.model_checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        _model.saver.restore(_session, ckpt.model_checkpoint_path)

    # Load vocabularies.
    _vocab, _rev_vocab = data_utils.initialize_vocabulary(checkpoint.vocab_path)

    return _session, _model, _vocab, _rev_vocab


def answer_message(message):
    # Init model if it hasn't already been initialized
    global session, model, vocab, rev_vocab, checkpoint
    if session is None or model is None:
        session, model, vocab, rev_vocab = init_session_model_vocab()

    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(message), vocab,
                                                 checkpoint.tokenizer)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(checkpoint.buckets))
                     if checkpoint.buckets[b][0] > len(token_ids)])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

    return checkpoint.join.join([tf.compat.as_str(rev_vocab[output]) for output in outputs])


@app.route('/', methods=['GET', 'POST'])
def index():
    global session, model, vocab, rev_vocab
    if request.method == 'GET':
        if session:
            return render_template('chatbot.html')
        else:
            return render_template('landing.html')
    elif request.method == 'POST':
        if request.form['submit'] == 'Start chatting':
            session, model, vocab, rev_vocab = init_session_model_vocab()
            return render_template('chatbot.html')


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    message = str(request.form['messageContent'])
    response = answer_message(message)
    return jsonify({'status': 'OK', 'answer': response})


if __name__ == "__main__":
    app.run()

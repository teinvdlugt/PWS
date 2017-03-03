import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

from . import data_utils
from . import seq2seq_model

app = Flask(__name__, static_url_path='/static')

# Deze dingen kunnen we aanpasbaar maken in de GUI
CHECKPOINT_PATH = "./gui"
# Deze variabelen moeten in overeenstemming zijn met het checkpoint bestand
VOCAB_PATH = "./gui/vocab60"
VOCAB_SIZE = 60
SIZE = 256
NUM_LAYERS = 2
NUM_SAMPLES = 0
buckets = [(10, 40), (30, 100), (60, 100), (100, 200)]

session = None
model = None
vocab = None
rev_vocab = None


def init_session_model_vocab(checkpoint_path, vocab_path):
    # Create session
    _session = tf.Session()

    # Create model
    _model = seq2seq_model.Seq2SeqModel(VOCAB_SIZE, buckets, SIZE, NUM_LAYERS, 0,
                                        1,  # We decode one sentence at a time.
                                        0, 0, NUM_SAMPLES, forward_only=True)

    # Load the checkpoint file
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        _model.saver.restore(_session, ckpt.model_checkpoint_path)

    # Load vocabularies.
    _vocab, _rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    loaded = True

    return _session, _model, _vocab, _rev_vocab


def answer_message(message):
    # Init model if it hasn't already been initialized
    global session, model, vocab, rev_vocab, buckets
    if session is None or model is None:
        session, model, vocab, rev_vocab = init_session_model_vocab(CHECKPOINT_PATH, VOCAB_PATH)

    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(message), vocab,
                                                 data_utils.basic_word_tokenizer)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(buckets))
                     if buckets[b][0] > len(token_ids)])

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

    return "".join([tf.compat.as_str(rev_vocab[output]) for output in outputs])


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
            session, model, vocab, rev_vocab = init_session_model_vocab(CHECKPOINT_PATH, VOCAB_PATH)
            return render_template('chatbot.html')


@app.route('/chat', methods=['POST', 'GET'])
def chat():
    message = str(request.form['messageContent'])
    response = answer_message(message)
    return jsonify({'status': 'OK', 'answer': response})


if __name__ == "__main__":
    app.run()

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import main


app = Flask(__name__)

sess = tf.Session()
sess, model, vocab, rev_vocab = main.init_session(sess)


@app.route('/')
def index():
    return render_template('chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({'status': 'OK', 'answer': main.decode_message(sess, model, vocab, rev_vocab,
                                                                  str(request.form['messageContent']))})

if __name__ == "__main__":
    app.run()

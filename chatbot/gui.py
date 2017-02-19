from flask import Flask, render_template, request, jsonify
from . import main
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('chatbot.html')


@app.route('/message', methods=['GET'])
def message():
    return str(request.form['messageContent'])


@app.route('/chat', methods=['POST'])
def chat():
    with app.app_context():
        with tf.Session():
            return jsonify({'status': 'OK', 'answer': main.decode()})


if __name__ == "__main__":
    app.run()

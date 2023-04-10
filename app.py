from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

# load the saved model
model = tf.keras.models.load_model('model_gru.h5')

# create a Flask app
app = Flask(__name__)

# define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# define a route for the analyze function
@app.route('/analyze', methods=['POST'])
def analyze():
    # get the text from the input field
    text = request.form['text']

    # split the input text into individual sentences
    sentences = sent_tokenize(text)

    # preprocess the sentences
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # make a prediction using the model
    predictions = model.predict(padded_sequences)
    print("Input Text: ", text)
    print("Predictions: ", predictions)

    # convert the predictions to a human-readable format
    predictions = []
    results = []
    for prediction in predictions:
        if prediction > 0.5:
            results.append("Positive")
        else:
            results.append("Negative")

    # return the results to the user
    return render_template('result.html', results=results)
    return render_template('result.html', predictions=predictions)



# run the app
if __name__ == '__main__':
    app.run(debug=True)

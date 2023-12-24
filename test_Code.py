import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image #pip install pillow

# Loading the trained model
model = load_model('model/sentiment_analysis.h5')

# User entry here
sample_text = input("Enter the Text for Sentiment Analysis: ")

# Tokenizing and preprocessing the input
max_features = 10000
maxlen = 600 

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts([sample_text])

sequence = tokenizer.texts_to_sequences([sample_text])
input_data = pad_sequences(sequence, maxlen=maxlen)

# Making a prediction
probabilities = model.predict(input_data)[0][0]

image_path_negative_sentiment = 'Images\\negative.png'
image_path_positive_sentiment = 'Images\\positive.png'

img_negative_sentiment = Image.open(image_path_negative_sentiment)
img_postivie_sentiment = Image.open(image_path_positive_sentiment)

# Displaying the result
if probabilities <= 0.5:
    sentiment = 'Negative'
else:
    sentiment = 'Positive'

print("Sample Text:", sample_text)
if sentiment == 'Negative':
    print("Predicted Sentiment:", sentiment)
    print("Predicted Probability:", probabilities)
    img_negative_sentiment.show()
else:
    print("Predicted Sentiment:", sentiment)
    print("Predicted Probability:", probabilities)
    img_postivie_sentiment.show()

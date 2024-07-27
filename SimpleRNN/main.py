import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

model=load_model('best_model.h5')

# Decode an encoded review back to its original text form.
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, "?") for i in encoded_review])

def preprocess_review(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=100)
    return padded_review


# Streamlit

import streamlit as st
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify its sentiment:")
    
review_text = st.text_area("Review Text", "")
    
if st.button('Classify Review'):
    if review_text:
            # Preprocess the review and make prediction
            processed_review = preprocess_review(review_text)
            try:
                prediction = model.predict(processed_review)
                sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
                st.write(f"The review sentiment is: {sentiment}")
                st.write(f"The Probability is: {prediction[0][0]}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
            st.write("Please enter a review.")    
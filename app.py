import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()

def transform_text(text):
  # ... your text preprocessing code ...

# Load the fitted vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
  tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
  model = pickle.load(f)

# Set background color
st.markdown("<style>body {background-color:pink}</style>", unsafe_allow_html=True)

# Header and instructions
st.header('SMS Spam Classifier')
st.write('Enter an SMS message below to check if it is spam:')

# Text area for input
input_sms = st.text_area('SMS')

# Predict button
if st.button('RESULT'):
  if input_sms:
    # Preprocess
    transformed_sms = transform_text(input_sms)

    if transformed_sms:
      # Option A (Assuming similar data)
      # vector_input = tfidf.transform([transformed_sms])

      # Option B (If unsure about data similarity)
      vector_input = tfidf.transform([transformed_sms])

      # Check if the model is fitted before making predictions
      if hasattr(model, 'predict'):
        # Ensure the model is a classification model
        if not hasattr(model, 'classes_'):
          st.error("The loaded model seems not to be a classification model. Please ensure you're loading the correct model.")
          return

        # Predict
        result = model.predict(vector_input)[0]
        
        # Display prediction
        st.subheader('Prediction')
        if result == 1:
          st.success('Spam')
        else:
          st.warning('Not spam')
      else:
        st.error('Model is not fitted yet. Please fit the model before making predictions.')
  else:
    st.warning('Please enter an SMS message.')
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
ps = PorterStemmer()

def transform_text(text):
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum():
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)

        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))

        return " ".join(y)
    except Exception as e:
        st.error(f"Error in transform_text: {e}")
        return ""

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

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
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # Check if the model is fitted before making predictions
            if hasattr(model, 'predict'):
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

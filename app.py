import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os
nltk.data.path.append(os.path.expanduser('~/nltk_data'))

ps = PorterStemmer()

def transform_text(text):
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

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app
st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Email/SMS Spam Classifier")
st.image("https://static3.depositphotos.com/1000363/104/i/450/depositphotos_1045328-stock-photo-spam-warning.jpg", width=400)

st.write("""
### Enter the message below to check if it's spam or not.
This tool uses a machine learning model to classify the message as **Spam** or **Not Spam**.
""")

input_sms = st.text_area("Enter the message", placeholder="Type your message here...")

if st.button('Predict'):
    if input_sms:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("🚨 Spam")
            st.warning("This message is classified as spam.")
        else:
            st.header("✅ Not Spam")
            st.success("This message is classified as not spam.")
    else:
        st.error("Please enter a message to classify.")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
    }
    .stTextArea textarea {
        border: 2px solid #4CAF50;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

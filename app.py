import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ── Download NLTK data ────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

load_nltk()

# ── Load model & vectorizer ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, tfidf = load_model()

# ── Text preprocessing (same as notebook) ────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text  = re.sub(r'<.*?>', ' ', text)
    text  = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text  = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

def predict(text):
    cleaned = clean_text(text)
    vector  = tfidf.transform([cleaned])
    label   = model.predict(vector)[0]
    proba   = model.predict_proba(vector)[0]
    return label, proba

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("Email Spam Detector")
st.markdown("*Prepared By:- Rohit Kasaudhan*")
st.markdown("*Powered by Multinomial Naive Bayes & TF-IDF*")
st.markdown("")

# Text input
text_input = st.text_area(
    "Enter your email or SMS message:",
    height=150,
    placeholder="Paste your message here..."
)

# Predict button
if st.button(" Detect Spam", use_container_width=True, type="primary"):
    if text_input.strip():
        label, proba = predict(text_input)

        st.markdown("---")
        if label == "spam":
            st.error("**SPAM DETECTED**")
            confidence = proba[list(model.classes_).index('spam')] * 100
        else:
            st.success("**LEGITIMATE MESSAGE (Ham)**")
            confidence = proba[list(model.classes_).index('ham')] * 100

        st.markdown(f"**Confidence:** `{confidence:.1f}%`")
        st.progress(int(confidence))

        with st.expander(" View cleaned text"):
            st.code(clean_text(text_input))
    else:
        st.warning("Please enter a message to check.")


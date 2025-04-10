import streamlit as st
import joblib
import string

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Same preprocessing as in training
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set([
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'and', 'but', 'or',
    'this', 'that', 'is', 'was', 'are', 'were', 'be', 'been', 'to', 'with',
    'as', 'by', 'it', 'its', 'from'
])
  # Same stop word list
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("🎬 IMDb Sentiment Classifier")
st.markdown("Enter a movie review below and the model will predict its sentiment!")

user_input = st.text_area("Enter your movie review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = preprocess(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "😊 Positive" if prediction == 1 else "☹️ Negative"
        st.success(f"Predicted Sentiment: **{sentiment}**")

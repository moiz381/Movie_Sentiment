import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define your stop words
stop_words = set([
    'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'and', 'but', 'or', 
    'this', 'that', 'is', 'was', 'are', 'were', 'be', 'been', 'to', 'with', 
    'as', 'by', 'it', 'its', 'from'
])

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Clean reviews
df['cleaned_review'] = df['review'].apply(preprocess)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['label'], test_size=0.2, random_state=42
)

# ✅ Fit vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# ✅ Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ✅ Save to disk
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully.")

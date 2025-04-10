# Movie_Sentiment
# 🎬 Movie Review Sentiment Analysis

A machine learning web app that predicts whether a given movie review is **Positive** or **Negative**. Built using **Logistic Regression**, **TF-IDF Vectorization**, and deployed using **Streamlit**.

---

## 📌 Overview

This project uses the IMDb movie review dataset to train a sentiment classification model. The model takes a user’s review as input and predicts its sentiment in real-time via a Streamlit web app.

- ✅ Model: Logistic Regression
- 🧠 NLP: TF-IDF Vectorizer
- 📊 Accuracy: ~88.5%
- 🌐 Deployed: Streamlit Cloud

---

## 📁 Project Structure

📦 Movie-Review-Sentiment-Analysis/ ├── app.py # Streamlit web app ├── model_training.py # Script to train and save the model ├── model.pkl # Trained Logistic Regression model ├── vectorizer.pkl # Fitted TF-IDF vectorizer ├── requirements.txt # Dependencies ├── Movie Review Sentiment Analysis.ipynb # Notebook with training & evaluation └── README.md # This file

---

## 🧠 Dataset

**IMDB Large Movie Review Dataset**

- 50,000 reviews split into 25k for training and 25k for testing.
- Binary labels: **positive (1)** or **negative (0)**.
- You can download it from (https://www.kaggle.com/datasets/mantri7/imdb-movie-reviews-dataset).

> ⚠️ **Note**: The dataset is not included in the repo due to size. Please download `IMDB Dataset.csv` and place it in the root directory before training.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
cd movie-review-sentiment-analysis

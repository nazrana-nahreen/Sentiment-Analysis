# Sentiment-Analysis
Goal: Classify movie reviews as Positive or Negative using NLP techniques.  Tools: Python, scikit-learn, NLTK (for preprocessing), or Hugging Face (optional).  Dataset: IMDb dataset (built-in with Keras or from Kaggle).
# IMDb Sentiment Analysis (Movie Reviews)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.25-green?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

## 🔥 Project Overview
This project implements a **Sentiment Analysis system** to classify **IMDb movie reviews** as **Positive** or **Negative** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system can automatically read movie reviews, extract meaningful features from text, and predict whether the review expresses a positive or negative sentiment.

---

## 🛠 Tools & Technologies
- Python 3.10+
- TensorFlow (for LSTM/Embeddings approach)
- scikit-learn (for Logistic Regression approach)
- NLTK (for text preprocessing)
- NumPy

---

## 🧠 How It Works

### **1. Dataset**
- **IMDb Movie Reviews** dataset (built-in with TensorFlow/Keras)
- Contains **50,000 reviews**, equally split between positive and negative.
- Reviews are sequences of integers representing words (top 10,000 most frequent words).

### **2. Preprocessing**
- Normalize review length using **padding** (`pad_sequences`) so all sequences have the same length.
- Optional: Convert integers back to words for visualization.
- Remove irrelevant tokens (stopwords can be removed if using raw text).

### **3. Feature Extraction**
- **Logistic Regression Approach:**  
  - Convert text sequences to Bag-of-Words vectors using `CountVectorizer`.
- **LSTM Approach:**  
  - Use `Embedding` layer to convert integers to dense vectors.
  - `LSTM` layer captures sequential patterns in the text.

### **4. Model Training**
- **Logistic Regression:** trains on Bag-of-Words features to separate positive vs negative reviews.
- **LSTM Neural Network:** trains on sequences of word embeddings to capture sentiment patterns.

### **5. Prediction**
- The trained model can classify new reviews as:
  - `0` → Negative
  - `1` → Positive
- Outputs probability for confidence score (if using LSTM).

### **6. Evaluation**
- Model accuracy is measured on a **test dataset**.
- Logistic Regression: ~85% accuracy  
- LSTM: Can achieve higher accuracy depending on hyperparameters.

---

## 📊 Results
- **Test Accuracy:** ~85% (Logistic Regression)
- **Example Prediction:**  
  - Review: *"The movie was amazing and I loved it"* → Predicted: Positive  
  - Review: *"Terrible movie. Waste of time"* → Predicted: Negative

---

## 🚀 How to Run
1. Clone this repository:
```bash
git clone https://github.com/YourUsername/IMDb-Sentiment-Analysis.git
📎 References

IMDb Dataset

Sentiment Analysis with TensorFlow

Scikit-learn Logistic Regression Guide

IMDB Sentiment Analysis with Deep Learning
A comprehensive sentiment analysis project using IMDB movie reviews dataset with multiple neural network architectures (RNN, LSTM, GRU) for comparison and evaluation.
🎯 Project Overview
This project implements sentiment analysis on IMDB movie reviews using four different neural network architectures:

Simple RNN (Recurrent Neural Network)
LSTM (Long Short-Term Memory)
GRU (Gated Recurrent Unit)
Advanced Multi-layer LSTM

The goal is to automatically classify movie reviews as positive or negative and compare the performance of different deep learning models.
📊 Dataset

Source: IMDB Movie Reviews Dataset (via Keras)
Size: 50,000 movie reviews
Classes: Binary classification (Positive/Negative)
Split: 60% Training, 20% Validation, 20% Testing
Vocabulary: Top 10,000 most frequent words

🛠️ Features
Data Preprocessing

Text cleaning (HTML removal, special characters, case normalization)
Stopword removal
Stemming using Porter Stemmer
Tokenization and sequence padding

Model Architectures

Simple RNN Model

Embedding layer (128 dimensions)
Simple RNN layer (64 units)
Dense layers with dropout
Expected accuracy: ~83%


LSTM Model

Embedding layer (128 dimensions)
LSTM layer (64 units) with dropout
Dense layers with regularization
Expected accuracy: ~87%


GRU Model

Embedding layer (128 dimensions)
GRU layer (64 units) with dropout
Dense layers with regularization
Expected accuracy: ~86%


Advanced LSTM Model

Embedding layer (128 dimensions)
Two LSTM layers (128 and 64 units)
Multiple dense layers with dropout
Expected accuracy: ~88%



Evaluation & Visualization

Accuracy comparison across models
Training history plots (loss and accuracy curves)
Model complexity comparison
Classification reports and confusion matrices
Sample predictions with confidence scores

🚀 Installation
Prerequisites
bashPython 3.7+
Required Libraries
bashpip install pandas numpy matplotlib seaborn scikit-learn tensorflow nltk
NLTK Data Download
The script automatically downloads required NLTK data (stopwords) on first run.

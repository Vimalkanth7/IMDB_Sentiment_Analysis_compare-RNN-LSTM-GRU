# IMDB Sentiment Analysis using LSTM, GRU, and RNN Models
# This script demonstrates sentiment analysis on IMDB movie reviews using different neural network architectures

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK stopwords already downloaded")

print("=== IMDB Sentiment Analysis with Deep Learning ===")
print("Libraries imported successfully!")

# ================================
# 1. DATA LOADING AND EXPLORATION
# ================================

def load_imdb_data():
    """
    Load IMDB dataset. If Kaggle dataset is not available, 
    we'll use the built-in Keras IMDB dataset as a fallback.
    """
    try:
        # Try to load from Kaggle dataset if available
        # Uncomment and modify path if you have the Kaggle dataset
        # df = pd.read_csv('IMDB Dataset.csv')
        # return df
        
        # Fallback: Use Keras built-in IMDB dataset
        print("Loading IMDB dataset from Keras...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
        
        # Get word index for decoding
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = {value: key for key, value in word_index.items()}
        
        # Decode sequences back to text
        def decode_review(encoded_review):
            return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
        
        # Convert to DataFrame format
        train_reviews = [decode_review(review) for review in x_train]
        test_reviews = [decode_review(review) for review in x_test]
        
        # Combine train and test for preprocessing
        all_reviews = train_reviews + test_reviews
        all_labels = np.concatenate([y_train, y_test])
        
        df = pd.DataFrame({
            'review': all_reviews,
            'sentiment': ['positive' if label == 1 else 'negative' for label in all_labels]
        })
        
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Load the data
df = load_imdb_data()

if df is not None:
    # Basic exploration
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
    print(f"\nSample Reviews:")
    print(df.head(2))

# ================================
# 2. DATA PREPROCESSING
# ================================

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords and apply stemming
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def preprocess_dataframe(self, df):
        """Preprocess the entire dataframe"""
        print("Preprocessing text data...")
        df_processed = df.copy()
        df_processed['clean_review'] = df_processed['review'].apply(self.clean_text)
        
        # Remove empty reviews
        df_processed = df_processed[df_processed['clean_review'].str.len() > 0]
        
        # Convert sentiment to binary
        df_processed['label'] = (df_processed['sentiment'] == 'positive').astype(int)
        
        print(f"Preprocessing complete! Final shape: {df_processed.shape}")
        return df_processed

# Preprocess the data
if df is not None:
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataframe(df)
    
    print(f"\nPreprocessed Data Sample:")
    print(df_clean[['clean_review', 'label']].head(2))

# ================================
# 3. TEXT TOKENIZATION AND EMBEDDING
# ================================

def prepare_sequences(texts, labels, max_features=10000, max_len=200):
    """Tokenize and pad sequences"""
    print("Tokenizing and preparing sequences...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = np.array(labels)
    
    print(f"Sequences prepared! Shape: {X.shape}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    
    return X, y, tokenizer

if df is not None:
    # Prepare sequences
    MAX_FEATURES = 10000  # Maximum number of words in vocabulary
    MAX_LEN = 200         # Maximum sequence length
    
    X, y, tokenizer = prepare_sequences(
        df_clean['clean_review'].values,
        df_clean['label'].values,
        MAX_FEATURES,
        MAX_LEN
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\nData Split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

# ================================
# 4. MODEL ARCHITECTURES
# ================================

def create_rnn_model(max_features, max_len, embedding_dim=128, units=64):
    """Create Simple RNN model"""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        SimpleRNN(units, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_model(max_features, max_len, embedding_dim=128, units=64):
    """Create LSTM model"""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        LSTM(units, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_gru_model(max_features, max_len, embedding_dim=128, units=64):
    """Create GRU model"""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        GRU(units, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_advanced_lstm_model(max_features, max_len, embedding_dim=128):
    """Create advanced LSTM model with multiple layers"""
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ================================
# 5. TRAINING AND EVALUATION
# ================================

def train_and_evaluate_model(model, model_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate a model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} Model")
    print(f"{'='*50}")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=0.0001
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return history, test_accuracy, y_pred

# ================================
# 6. MAIN EXECUTION
# ================================

if df is not None:
    # Initialize models
    models = {
        'Simple RNN': create_rnn_model(MAX_FEATURES, MAX_LEN),
        'LSTM': create_lstm_model(MAX_FEATURES, MAX_LEN),
        'GRU': create_gru_model(MAX_FEATURES, MAX_LEN),
        'Advanced LSTM': create_advanced_lstm_model(MAX_FEATURES, MAX_LEN)
    }
    
    # Train and evaluate all models
    results = {}
    histories = {}
    
    for model_name, model in models.items():
        print(f"\nModel Architecture for {model_name}:")
        model.summary()
        
        history, accuracy, predictions = train_and_evaluate_model(
            model, model_name, X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        results[model_name] = accuracy
        histories[model_name] = history
    
    # ================================
    # 7. RESULTS COMPARISON
    # ================================
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nModel Performance Ranking:")
    print("-" * 40)
    for i, (model_name, accuracy) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}: {accuracy:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    accuracies = list(results.values())
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Training history plots
    colors = ['blue', 'green', 'red', 'orange']
    
    # Training loss
    plt.subplot(2, 2, 2)
    for i, (model_name, history) in enumerate(histories.items()):
        plt.plot(history.history['loss'], color=colors[i], label=f'{model_name} Train')
        plt.plot(history.history['val_loss'], color=colors[i], linestyle='--', label=f'{model_name} Val')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Training accuracy
    plt.subplot(2, 2, 3)
    for i, (model_name, history) in enumerate(histories.items()):
        plt.plot(history.history['accuracy'], color=colors[i], label=f'{model_name} Train')
        plt.plot(history.history['val_accuracy'], color=colors[i], linestyle='--', label=f'{model_name} Val')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Model complexity comparison
    plt.subplot(2, 2, 4)
    model_params = []
    for model in models.values():
        model_params.append(model.count_params())
    
    bars = plt.bar(model_names, model_params, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Complexity (Parameters)')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar, params in zip(bars, model_params):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_params)*0.01,
                f'{params:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # ================================
    # 8. SAMPLE PREDICTIONS
    # ================================
    
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    
    # Get the best model
    best_model_name = sorted_results[0][0]
    best_model = models[best_model_name]
    
    # Sample predictions
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    print(f"\nSample predictions using {best_model_name}:")
    print("-" * 50)
    
    for idx in sample_indices:
        # Get original text (reverse preprocessing for display)
        original_idx = df_clean.index[X_train.shape[0] + X_val.shape[0] + idx]
        original_text = df_clean.loc[original_idx, 'review'][:200] + "..."
        
        # Get prediction
        pred_proba = best_model.predict(X_test[idx:idx+1], verbose=0)[0][0]
        pred_label = "Positive" if pred_proba > 0.5 else "Negative"
        actual_label = "Positive" if y_test[idx] == 1 else "Negative"
        
        print(f"\nReview: {original_text}")
        print(f"Actual: {actual_label} | Predicted: {pred_label} (Confidence: {pred_proba:.3f})")
        print("-" * 50)
    
    print(f"\n🎉 Sentiment Analysis Complete!")
    print(f"Best performing model: {best_model_name} with {sorted_results[0][1]:.4f} accuracy")

else:
    print("Could not load dataset. Please ensure you have the IMDB dataset available.")

# ================================
# 9. HELPER FUNCTIONS FOR NEW PREDICTIONS
# ================================

def predict_sentiment(text, model, tokenizer, max_len=200):
    """Predict sentiment for new text"""
    # Preprocess the text
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

# Example usage (uncomment to test with custom text):
# if 'best_model' in locals():
#     sample_text = "This movie was absolutely fantastic! Great acting and storyline."
#     sentiment, confidence = predict_sentiment(sample_text, best_model, tokenizer)
#     print(f"\nCustom Prediction:")
#     print(f"Text: {sample_text}")
#     print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")

print("\n" + "="*60)
print("SCRIPT EXECUTION COMPLETED")
print("="*60)
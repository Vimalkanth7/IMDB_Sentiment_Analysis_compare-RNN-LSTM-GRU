# IMDB Sentiment Analysis: Complete Code Explanation

## Overview
This code performs **sentiment analysis** on movie reviews from the IMDB dataset. Sentiment analysis means determining whether a review is positive or negative. We use deep learning models (RNN, LSTM, GRU) to automatically classify reviews.

---

## Step-by-Step Code Breakdown

### 1. **Library Imports and Setup**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense, Dropout
```

**What this does:**
- **pandas**: For handling data in table format (like Excel sheets)
- **numpy**: For mathematical operations on arrays
- **matplotlib/seaborn**: For creating graphs and visualizations
- **sklearn**: For splitting data and evaluating model performance
- **tensorflow/keras**: For building and training neural networks

**Why we need these:**
Each library has a specific purpose in our machine learning pipeline.

---

### 2. **Data Loading Function**

```python
def load_imdb_data():
    # Loads IMDB movie review dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
```

**What this does:**
- Downloads the IMDB dataset (50,000 movie reviews)
- Each review is already converted to numbers (we'll explain this later)
- `num_words=10000` means we only use the 10,000 most common words

**Data Structure:**
- `x_train`: Training reviews (as numbers)
- `y_train`: Training labels (0=negative, 1=positive)
- `x_test`: Test reviews (as numbers)
- `y_test`: Test labels

**Example of raw data:**
```
Review (as numbers): [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 5244, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 5952, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 7486, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 5535, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 5345, 19, 178, 32]

Label: 1 (positive)
```

**Converting numbers back to words:**
```python
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
```

This converts the numbers back to readable text like: "this film was just brilliant casting location scenery story direction everyone's really suited..."

---

### 3. **Text Preprocessing Class**

```python
class TextPreprocessor:
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
```

**What text preprocessing does:**

**Before preprocessing:**
```
"This movie was ABSOLUTELY fantastic! <br/>The acting was great... 9/10 would recommend!!!"
```

**After preprocessing:**
```
"movi absolut fantast act great would recommend"
```

**Steps explained:**
1. **Lowercase**: "This" → "this" (consistency)
2. **Remove HTML**: `<br/>` → removed
3. **Remove punctuation**: "!!!" → removed
4. **Remove stopwords**: "was", "the" → removed (common words that don't add meaning)
5. **Stemming**: "fantastic" → "fantast", "acting" → "act" (reduce words to root form)

**Why do this?**
- Reduces vocabulary size
- Focuses on meaningful words
- Improves model performance

---

### 4. **Tokenization and Sequence Preparation**

```python
def prepare_sequences(texts, labels, max_features=10000, max_len=200):
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
```

**What tokenization does:**

**Step 1: Create vocabulary**
```
Word → Number mapping:
"the" → 1
"and" → 2
"movie" → 3
"good" → 4
...
```

**Step 2: Convert text to numbers**
```
"this movie was good" → [45, 3, 12, 4]
```

**Step 3: Pad sequences**
All reviews must have the same length for neural networks:
```
Original: [45, 3, 12, 4]
Padded:   [45, 3, 12, 4, 0, 0, 0, 0, 0, 0, ...] (length 200)
```

**Why padding?**
Neural networks need fixed-size inputs. Short reviews get zeros added, long reviews get truncated.

---

### 5. **Data Splitting**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
```

**Data split visualization:**
```
Original Dataset (50,000 reviews)
├── Training Set (32,000 reviews) - 64%
├── Validation Set (8,000 reviews) - 16%
└── Test Set (10,000 reviews) - 20%
```

**Purpose of each set:**
- **Training**: Teach the model patterns
- **Validation**: Monitor performance during training
- **Test**: Final evaluation (model never sees this during training)

---

### 6. **Neural Network Models**

#### **A. Embedding Layer**
```python
Embedding(max_features, embedding_dim, input_length=max_len)
```

**What embedding does:**
Converts word numbers to dense vectors that capture meaning:

```
Word: "good" (number: 4)
Embedding: [0.2, -0.1, 0.8, -0.3, 0.5, ...]  (128 dimensions)

Word: "excellent" (number: 156)
Embedding: [0.3, -0.2, 0.9, -0.2, 0.6, ...]  (similar to "good")
```

**Why embeddings?**
- Words with similar meanings get similar vectors
- Captures semantic relationships
- Learnable during training

#### **B. Simple RNN Model**
```python
def create_rnn_model(max_features, max_len, embedding_dim=128, units=64):
    model = Sequential([
        Embedding(max_features, embedding_dim, input_length=max_len),
        SimpleRNN(units, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
```

**RNN Architecture Flow:**
```
Input: [word1, word2, word3, word4, ...]
   ↓
Embedding: [[0.2, -0.1, ...], [0.3, 0.5, ...], ...]
   ↓
RNN: Processes sequence step by step
   word1 → hidden_state1
   word2 + hidden_state1 → hidden_state2
   word3 + hidden_state2 → hidden_state3
   ...
   ↓
Dense Layer: 64 neurons with ReLU activation
   ↓
Output Layer: 1 neuron with sigmoid (0-1 probability)
```

#### **C. LSTM Model**
```python
LSTM(units, dropout=0.2, recurrent_dropout=0.2)
```

**LSTM vs RNN:**
- **RNN Problem**: Forgets long-term dependencies
- **LSTM Solution**: Has memory cells to remember important information

**LSTM Memory Mechanism:**
```
Input: "The movie started slow but the ending was fantastic"

RNN might forget "slow" when processing "fantastic"
LSTM remembers both "slow" and "fantastic" for final decision
```

#### **D. GRU Model**
```python
GRU(units, dropout=0.2, recurrent_dropout=0.2)
```

**GRU vs LSTM:**
- **LSTM**: More complex, 3 gates (forget, input, output)
- **GRU**: Simpler, 2 gates (reset, update)
- **Performance**: Often similar, GRU trains faster

---

### 7. **Training Process**

```python
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)
```

**Training Parameters Explained:**

**Batch Size (128):**
- Don't process all 32,000 reviews at once
- Process 128 reviews, update weights, repeat
- Faster and more stable training

**Epochs (10):**
- One epoch = seeing all training data once
- 10 epochs = model sees each review 10 times
- More epochs can improve learning but risk overfitting

**Callbacks:**
- **Early Stopping**: Stop if validation performance doesn't improve
- **Reduce LR**: Lower learning rate if stuck

**Training Loop Visualization:**
```
Epoch 1:
├── Batch 1 (reviews 1-128): loss = 0.693, accuracy = 0.52
├── Batch 2 (reviews 129-256): loss = 0.681, accuracy = 0.55
├── ...
├── Batch 250 (reviews 31,873-32,000): loss = 0.423, accuracy = 0.78
└── Validation: loss = 0.445, accuracy = 0.76

Epoch 2:
├── (repeat with shuffled data)
└── Validation: loss = 0.398, accuracy = 0.82

...
```

---

### 8. **Model Evaluation**

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()
```

**Evaluation Process:**

**Step 1: Get Predictions**
```
Review: "This movie was terrible"
Model Output: 0.15 (15% positive)
Prediction: Negative (< 0.5 threshold)
```

**Step 2: Calculate Metrics**
```
Accuracy = Correct Predictions / Total Predictions
Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
```

**Step 3: Confusion Matrix**
```
                Predicted
Actual    Negative  Positive
Negative    4200      300
Positive     200     4300

Accuracy = (4200 + 4300) / 9000 = 94.4%
```

---

### 9. **Key Concepts Summary**

#### **Data Flow Through the System:**
```
Raw Text → Preprocessing → Tokenization → Padding → Embedding → RNN/LSTM/GRU → Classification
```

#### **What Each Model Learns:**
- **Patterns**: "not good" = negative, "excellent" = positive
- **Context**: "not bad" (positive despite "bad")
- **Sequences**: Order of words matters

#### **Why Deep Learning?**
- Traditional methods count words
- Deep learning understands context and meaning
- Handles complex language patterns

#### **Model Comparison:**
- **Simple RNN**: Fast but limited memory
- **LSTM**: Better memory, more complex
- **GRU**: Balance of speed and performance
- **Advanced LSTM**: Multiple layers for complex patterns

---

### 10. **Practical Application**

When you run this code, it will:

1. **Load** 50,000 movie reviews
2. **Clean** the text (remove noise)
3. **Convert** text to numbers
4. **Train** four different models
5. **Compare** their performance
6. **Show** which model works best
7. **Demonstrate** predictions on new reviews

The final output tells you which neural network architecture is most effective for sentiment analysis on movie reviews, typically achieving 85-90% accuracy.

---

### 11. **Common Results**

**Typical Performance:**
- Simple RNN: ~83% accuracy
- LSTM: ~87% accuracy  
- GRU: ~86% accuracy
- Advanced LSTM: ~88% accuracy

**Why LSTM/GRU perform better:**
They can remember important information from earlier in the review, like "I expected this movie to be terrible, but it was actually amazing!"
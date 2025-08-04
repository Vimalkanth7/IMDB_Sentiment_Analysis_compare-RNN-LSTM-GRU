<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Process Flow</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        
        .title {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .flow-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
            align-items: center;
        }
        
        .step {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 20px;
            border-radius: 15px;
            width: 80%;
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
            position: relative;
        }
        
        .step:hover {
            transform: translateY(-5px);
        }
        
        .step-number {
            position: absolute;
            top: -10px;
            left: 20px;
            background: #ff6b6b;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        
        .arrow {
            font-size: 2em;
            color: #666;
            margin: 10px 0;
        }
        
        .data-example {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            font-family: monospace;
            font-size: 0.9em;
            color: #333;
        }
        
        .model-comparison {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .model-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        
        .model-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .accuracy {
            font-size: 2em;
            font-weight: bold;
            color: #ffd700;
        }
        
        .details-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .section-title {
            color: #333;
            font-size: 1.3em;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
        }
        
        .neural-network {
            display: flex;
            align-items: center;
            justify-content: space-around;
            padding: 20px;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 10px;
            color: white;
            margin: 10px 0;
        }
        
        .layer {
            text-align: center;
            padding: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 8px;
            min-width: 100px;
        }
        
        .layer-arrow {
            font-size: 1.5em;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">🎬 IMDB Sentiment Analysis Process</h1>
        
        <div class="flow-container">
            <div class="step">
                <div class="step-number">1</div>
                <strong>Data Loading</strong><br>
                Load 50,000 IMDB movie reviews from Keras dataset
                <div class="data-example">
                    Example: "This movie was absolutely fantastic!" → [1, 14, 22, 16, 43, 530, 973...]
                </div>
            </div>
            
            <div class="arrow">↓</div>
            
            <div class="step">
                <div class="step-number">2</div>
                <strong>Text Preprocessing</strong><br>
                Clean and normalize the text data
                <div class="data-example">
                    Before: "This Movie Was GREAT!!! &lt;br/&gt; 10/10"<br>
                    After: "movi great"
                </div>
            </div>
            
            <div class="arrow">↓</div>
            
            <div class="step">
                <div class="step-number">3</div>
                <strong>Tokenization</strong><br>
                Convert words to numbers and create vocabulary
                <div class="data-example">
                    Vocabulary: {"the": 1, "movie": 2, "good": 3, "bad": 4, ...}<br>
                    "good movie" → [3, 2]
                </div>
            </div>
            
            <div class="arrow">↓</div>
            
            <div class="step">
                <div class="step-number">4</div>
                <strong>Sequence Padding</strong><br>
                Make all reviews the same length (200 words)
                <div class="data-example">
                    Short: [3, 2] → [3, 2, 0, 0, 0, ..., 0] (padded with zeros)<br>
                    Long: [1, 2, 3, ..., 250 words] → [1, 2, 3, ..., 200] (truncated)
                </div>
            </div>
            
            <div class="arrow">↓</div>
            
            <div class="step">
                <div class="step-number">5</div>
                <strong>Data Splitting</strong><br>
                Divide data into Training (64%), Validation (16%), Test (20%)
                <div class="data-example">
                    Training: 32,000 reviews (learn patterns)<br>
                    Validation: 8,000 reviews (monitor performance)<br>
                    Test: 10,000 reviews (final evaluation)
                </div>
            </div>
            
            <div class="arrow">↓</div>
            
            <div class="step">
                <div class="step-number">6</div>
                <strong>Neural Network Architecture</strong><br>
                Build four different models for comparison
            </div>
        </div>
        
        <div class="details-section">
            <div class="section-title">Neural Network Flow</div>
            <div class="neural-network">
                <div class="layer">
                    <strong>Input</strong><br>
                    [3, 2, 45, 0, 0...]<br>
                    <small>Padded sequence</small>
                </div>
                <div class="layer-arrow">→</div>
                <div class="layer">
                    <strong>Embedding</strong><br>
                    Convert to vectors<br>
                    <small>128 dimensions</small>
                </div>
                <div class="layer-arrow">→</div>
                <div class="layer">
                    <strong>RNN/LSTM/GRU</strong><br>
                    Process sequence<br>
                    <small>Remember context</small>
                </div>
                <div class="layer-arrow">→</div>
                <div class="layer">
                    <strong>Dense</strong><br>
                    64 neurons<br>
                    <small>Feature extraction</small>
                </div>
                <div class="layer-arrow">→</div>
                <div class="layer">
                    <strong>Output</strong><br>
                    0.85<br>
                    <small>85% positive</small>
                </div>
            </div>
        </div>
        
        <div class="details-section">
            <div class="section-title">Model Comparison</div>
            <div class="model-comparison">
                <div class="model-card">
                    <div class="model-title">Simple RNN</div>
                    <div class="accuracy">~83%</div>
                    <p>Basic recurrent network. Fast but limited memory for long sequences.</p>
                </div>
                <div class="model-card">
                    <div class="model-title">LSTM</div>
                    <div class="accuracy">~87%</div>
                    <p>Long Short-Term Memory. Remembers important information throughout the review.</p>
                </div>
                <div class="model-card">
                    <div class="model-title">GRU</div>
                    <div class="accuracy">~86%</div>
                    <p>Gated Recurrent Unit. Simpler than LSTM but still handles long sequences well.</p>
                </div>
                <div class="model-card">
                    <div class="model-title">Advanced LSTM</div>
                    <div class="accuracy">~88%</div>
                    <p>Multi-layer LSTM. Most complex, captures intricate patterns in text.</p>
                </div>
            </div>
        </div>
        
        <div class="details-section">
            <div class="section-title">Training Process</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>What happens during training:</h4>
                    <ul>
                        <li><strong>Forward Pass:</strong> Model predicts sentiment</li>
                        <li><strong>Loss Calculation:</strong> Compare with actual labels</li>
                        <li><strong>Backpropagation:</strong> Adjust weights to reduce error</li>
                        <li><strong>Repeat:</strong> Process batches of 128 reviews</li>
                    </ul>
                </div>
                <div>
                    <h4>Training monitoring:</h4>
                    <ul>
                        <li><strong>Epochs:</strong> Complete passes through data</li>
                        <li><strong>Validation:</strong> Check performance on unseen data</li>
                        <li><strong>Early Stopping:</strong> Prevent overfitting</li>
                        <li><strong>Learning Rate:</strong> Control step size for weight updates</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="details-section">
            <div class="section-title">Key Concepts Explained</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <h4>🔤 Word Embeddings</h4>
                    <p>Convert words to numerical vectors that capture meaning. Similar words get similar vectors.</p>
                    
                    <h4>🧠 Memory in Neural Networks</h4>
                    <p>RNNs process sequences step by step. LSTMs and GRUs can remember important information from earlier parts.</p>
                </div>
                <div>
                    <h4>📊 Evaluation Metrics</h4>
                    <p>Accuracy: Percentage of correct predictions<br>
                    Precision: Of predicted positives, how many were actually positive<br>
                    Recall: Of actual positives, how many were correctly identified</p>
                    
                    <h4>🎯 Why This Works</h4>
                    <p>Deep learning models learn complex patterns in language that traditional methods miss, like context, sarcasm, and nuanced expressions.</p>
                </div>
            </div>
        </div>
        
        <div class="step" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); margin-top: 30px;">
            <strong>🎉 Final Result</strong><br>
            A trained model that can predict sentiment of new movie reviews with ~85-90% accuracy!
            <div class="data-example" style="margin-top: 15px; color: #333;">
                Example Prediction:<br>
                Input: "This movie was boring and predictable"<br>
                Output: Negative (92% confidence)
            </div>
        </div>
    </div>
</body>
</html>
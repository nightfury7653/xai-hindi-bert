# Implementation Guide: Phase 1 Complete Breakdown

## What We've Built

I've created a **complete, production-ready implementation** of Phase 1 for your Hindi Sentiment Analysis project. Here's what's been set up:

### Project Structure

```
NLP Project/
├── config.py                          # ⚙️  All configuration in one place
├── requirements.txt                    # 📦 Python dependencies
├── README.md                          # 📖 Project overview
├── QUICK_START.md                     # 🚀 Quick start guide
├── IMPLEMENTATION_GUIDE.md            # 📚 This file
├── run_phase1.py                      # ▶️  Executable Phase 1 script
│
├── src/                               # 💻 Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data handling utilities
│   ├── model.py                       # BERT classifier implementation
│   ├── explainability.py              # (Phase 3)
│   └── visualization.py               # (Phase 6)
│
├── notebooks/                         # 📓 Jupyter notebooks
│   ├── phase1_model_training.ipynb    # Interactive Phase 1
│   ├── phase2_attention_analysis.ipynb # (Next phase)
│   └── ...
│
├── data/                              # 📊 Datasets
│   ├── raw/                          # Original data
│   ├── processed/                    # Cleaned & split data
│   └── counterfactuals/              # (Phase 5)
│
├── models/                            # 🤖 Saved models
└── outputs/                           # 📈 Visualizations & results
```

---

## Module Breakdown

### 1. `config.py` - Central Configuration

**Purpose:** All hyperparameters and settings in one place.

**Key Features:**
- Model selection (IndicBERT, mBERT, HindiBERT)
- Training hyperparameters
- Path management
- Device configuration (CPU/GPU)
- Label mappings

**How to modify:**
```python
# Change model
MODEL_NAME = MODEL_OPTIONS['mbert']  # Switch to multilingual BERT

# Adjust training
BATCH_SIZE = 8          # Reduce for less memory
NUM_EPOCHS = 5          # Train longer
LEARNING_RATE = 3e-5    # Change learning rate
```

---

### 2. `src/data_preprocessing.py` - Data Utilities

**Contains:**

#### Class: `HindiTextCleaner`
- Cleans Hindi text
- Removes extra whitespace
- Optionally removes emojis/special characters
- Handles invalid entries

```python
cleaner = HindiTextCleaner(
    remove_emojis=False,        # Keep emojis (they carry sentiment!)
    remove_special_chars=False  # Keep Hindi punctuation
)
df_clean = cleaner.clean_dataframe(df, text_column='text')
```

#### Class: `SentimentDataset` 
- PyTorch Dataset wrapper
- Tokenizes text on-the-fly
- Returns input_ids, attention_mask, labels
- Handles padding/truncation automatically

```python
dataset = SentimentDataset(
    texts=['यह अच्छी है', 'यह बुरी है'],
    labels=[2, 0],  # positive, negative
    tokenizer=tokenizer,
    max_length=128
)
```

#### Functions:
- `create_sample_dataset()`: Generate demo data
- `prepare_data_splits()`: Create train/val/test splits (stratified)
- `print_dataset_stats()`: Display dataset information
- `load_dataset()`: Load from CSV

---

### 3. `src/model.py` - BERT Classifier

**Contains:**

#### Class: `BERTSentimentClassifier`

**Architecture:**
```
Input Text
    ↓
Tokenizer (converts to IDs)
    ↓
BERT Base Model (12 layers, 768 hidden dim)
    ↓
[CLS] Token Representation
    ↓
Dropout Layer (prevents overfitting)
    ↓
Linear Classification Head (768 → 3)
    ↓
Logits (scores for each class)
    ↓
Softmax → Probabilities
```

**Key Parameters:**
```python
model = BERTSentimentClassifier(
    model_name='ai4bharat/indic-bert',
    num_labels=3,                    # Positive, Neutral, Negative
    hidden_dropout=0.1,              # Dropout rate
    output_attentions=False,         # Enable in Phase 2
    output_hidden_states=False       # Enable in Phase 4
)
```

#### Functions:

**`train_epoch()`**
- Trains model for one epoch
- Handles gradient clipping
- Updates learning rate scheduler
- Returns average loss

**`evaluate()`**
- Evaluates on validation/test set
- No gradient computation (faster)
- Returns loss, accuracy, predictions, true labels

**`predict_text()`**
- Predicts sentiment for single text
- Returns label, probabilities, class name

```python
pred_label, probs, pred_class = predict_text(
    model=model,
    text="यह फिल्म शानदार है",
    tokenizer=tokenizer,
    device=device
)
print(f"Prediction: {pred_class} (confidence: {probs[pred_label]:.2%})")
```

**`save_model()` & `load_model()`**
- Save/load trained model weights
- Save/load tokenizer configuration

---

### 4. `run_phase1.py` - Complete Workflow Script

**What it does:**

1. **Loads/creates dataset** (900 samples by default)
2. **Cleans text data** (removes invalid entries)
3. **Creates splits** (80% train, 10% val, 10% test)
4. **Initializes tokenizer** (downloads if needed)
5. **Creates PyTorch datasets and loaders**
6. **Initializes BERT model** (downloads pretrained weights)
7. **Sets up optimizer and scheduler**
8. **Trains for N epochs** (default: 3)
9. **Evaluates on test set**
10. **Tests with sample sentences**
11. **Saves visualizations** (training curves, confusion matrix)

**How to run:**
```bash
cd "/home/nightfury653/Documents/NLP Project"
python run_phase1.py
```

**Expected runtime:**
- First run: 30-40 minutes (includes downloading)
- Subsequent runs: 10-20 minutes

**Output:**
```
models/
├── model.pt              # Trained weights
├── config.json           # Model config
├── tokenizer_config.json # Tokenizer config
└── vocab.txt             # Vocabulary

outputs/
├── phase1_training_curves.png     # Loss & accuracy plots
└── phase1_confusion_matrix.png    # Confusion matrix heatmap
```

---

## How the Training Works (Detailed)

### Training Loop Explained

```python
for epoch in range(NUM_EPOCHS):
    for batch in train_loader:
        # 1. Get batch data
        input_ids = batch['input_ids']          # [batch_size, 128]
        attention_mask = batch['attention_mask'] # [batch_size, 128]
        labels = batch['labels']                 # [batch_size]
        
        # 2. Forward pass
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        # 3. Backward pass
        loss.backward()  # Compute gradients
        
        # 4. Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 5. Update weights
        optimizer.step()
        scheduler.step()
        
        # 6. Reset gradients
        optimizer.zero_grad()
```

### What Happens in Forward Pass

```python
def forward(input_ids, attention_mask, labels):
    # 1. Pass through BERT
    #    Input: [batch_size, 128] token IDs
    #    Output: [batch_size, 128, 768] hidden states
    bert_outputs = self.bert(input_ids, attention_mask)
    
    # 2. Get [CLS] token (first token, represents whole sentence)
    #    Shape: [batch_size, 768]
    cls_output = bert_outputs.last_hidden_state[:, 0, :]
    
    # 3. Apply dropout (randomly zeros some values)
    #    Prevents overfitting
    cls_output = self.dropout(cls_output)
    
    # 4. Classification head (linear layer)
    #    Shape: [batch_size, 3]
    logits = self.classifier(cls_output)
    
    # 5. Calculate loss
    #    CrossEntropyLoss = combines softmax + negative log likelihood
    loss = CrossEntropyLoss(logits, labels)
    
    return {'logits': logits, 'loss': loss}
```

### Why These Hyperparameters?

**Learning Rate: 2e-5**
- BERT is pretrained → already knows Hindi
- Too high = destroy pretrained knowledge
- Too low = very slow learning
- 2e-5 is empirically proven for BERT fine-tuning

**Batch Size: 16**
- Larger = faster training, more stable gradients
- Smaller = less memory, more noise (can help generalization)
- 16 is sweet spot for BERT on consumer GPUs

**Epochs: 3**
- BERT fine-tuning converges quickly
- More epochs risk overfitting on small datasets
- For larger datasets (10K+), can train 4-5 epochs

**Weight Decay: 0.01**
- Regularization technique
- Penalizes large weights
- Prevents overfitting

**Warmup Steps: 100**
- Gradually increase learning rate at start
- Prevents early instability
- Standard practice for transformers

---

## Understanding the Output

### Training Logs

```
Epoch 1/3
Training: 100%|████████| 45/45 [02:15<00:00, 3.01s/it, loss=0.8523]
Evaluating: 100%|████████| 6/6 [00:08<00:00]

Results:
   Train Loss:    0.8523
   Val Loss:      0.7234
   Val Accuracy:  0.6889 (68.89%)
   ✅ New best model saved!
```

**What this means:**
- **45 batches**: 720 training samples / 16 batch_size = 45
- **3.01s/it**: Each batch takes ~3 seconds
- **Train Loss 0.85**: Model's uncertainty is decreasing
- **Val Accuracy 68.89%**: Correctly predicts 69 out of 100 samples

### Good vs Bad Training

**✅ Healthy Training:**
```
Epoch 1: Train=1.05, Val=0.98, Acc=62%
Epoch 2: Train=0.72, Val=0.68, Acc=71%
Epoch 3: Train=0.51, Val=0.54, Acc=78%
```
- Both losses decreasing
- Val loss follows train loss
- Accuracy improving

**❌ Overfitting:**
```
Epoch 1: Train=1.05, Val=0.98, Acc=62%
Epoch 2: Train=0.52, Val=0.89, Acc=68%
Epoch 3: Train=0.21, Val=1.12, Acc=65%
```
- Train loss keeps decreasing
- Val loss INCREASES
- Val accuracy plateaus or drops

**Solutions:**
- Add more data
- Increase dropout rate
- Reduce epochs
- Add data augmentation

---

## Evaluation Metrics Explained

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions
```

Simple but can be misleading with imbalanced classes.

**Example:**
- Dataset: 90 positive, 10 negative
- Model always predicts positive
- Accuracy: 90% (but useless!)

### Precision
```
Precision = True Positives / (True Positives + False Positives)
```

"Of all positive predictions, how many were actually positive?"

**High precision = Few false alarms**

### Recall
```
Recall = True Positives / (True Positives + False Negatives)
```

"Of all actual positives, how many did we find?"

**High recall = Few missed cases**

### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Harmonic mean - balances precision and recall.

**Best metric for imbalanced classes.**

### Confusion Matrix

```
                Predicted
              Neg  Neu  Pos
Actual  Neg   25   3    2     (Actual negatives)
        Neu    5   18   7     (Actual neutrals)
        Pos    1   4    25    (Actual positives)
```

**Reading it:**
- **Diagonal** (25, 18, 25): Correct predictions
- **Row sums** (30, 30, 30): Actual class counts
- **Column sums**: Predicted class counts

**Common patterns:**
- Neutrals often confused with pos/neg
- Pos/Neg confusion is serious (opposite sentiments!)

---

## How to Use Your Trained Model

### Option 1: Direct Prediction

```python
from transformers import AutoTokenizer
import torch
from src.model import load_model

# Load model
model, tokenizer = load_model(
    model_name='ai4bharat/indic-bert',
    path='models/',
    device=torch.device('cpu')
)

# Predict
from src.model import predict_text
pred_label, probs, pred_class = predict_text(
    model=model,
    text="यह फिल्म बेहतरीन है",
    tokenizer=tokenizer,
    device=torch.device('cpu')
)

print(f"Sentiment: {pred_class}")
print(f"Confidence: {probs[pred_label]:.2%}")
```

### Option 2: Batch Prediction

```python
texts = [
    "मुझे यह पसंद है",
    "यह बहुत बुरा है",
    "ठीक है"
]

for text in texts:
    _, _, pred_class = predict_text(model, text, tokenizer, device)
    print(f"{text} → {pred_class}")
```

### Option 3: Integrate into Application

```python
class SentimentAnalyzer:
    def __init__(self, model_path):
        self.model, self.tokenizer = load_model(
            'ai4bharat/indic-bert',
            model_path,
            device='cpu'
        )
    
    def analyze(self, text):
        _, probs, pred_class = predict_text(
            self.model, text, self.tokenizer, 'cpu'
        )
        return {
            'sentiment': pred_class,
            'confidence': float(probs.max()),
            'scores': {
                'negative': float(probs[0]),
                'neutral': float(probs[1]),
                'positive': float(probs[2])
            }
        }

# Use it
analyzer = SentimentAnalyzer('models/')
result = analyzer.analyze("यह शानदार है")
print(result)
# Output: {'sentiment': 'positive', 'confidence': 0.92, 'scores': {...}}
```

---

## Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
# In config.py
BATCH_SIZE = 8  # or even 4
```

2. **Use gradient accumulation:**
```python
# Simulate larger batch size
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch_size = 16
```

3. **Use CPU (slower but works):**
```python
# In config.py
DEVICE = torch.device('cpu')
```

### Issue 2: Poor Accuracy (<60%)

**Causes & Solutions:**

**1. Insufficient Data**
- Need 5K+ samples minimum
- Solution: Collect more data

**2. Imbalanced Classes**
```
positive: 800 samples
neutral: 50 samples
negative: 150 samples
```
- Solution: Balance by oversampling minority classes

**3. Noisy Labels**
- Incorrect annotations
- Solution: Review and clean labels

**4. Wrong Hyperparameters**
- Learning rate too high/low
- Solution: Try 1e-5, 2e-5, 3e-5, 5e-5

### Issue 3: Model Not Learning

**Symptoms:**
- Loss stuck
- Accuracy not improving

**Causes:**

**1. Learning rate too low:**
```python
LEARNING_RATE = 2e-5  # Try 3e-5 or 5e-5
```

**2. Frozen BERT layers:**
```python
# Make sure all parameters are trainable
for param in model.parameters():
    param.requires_grad = True
```

**3. Wrong loss function:**
- CrossEntropyLoss expects class indices (0, 1, 2), not one-hot vectors

### Issue 4: Tokenizer Issues

**Error:**
```
Token indices sequence length is longer than the specified maximum sequence length
```

**Solution:**
```python
# Increase max_length in config.py
MAX_LENGTH = 256  # or 512 for longer texts
```

---

## Next Steps: Phase 2-7

### Phase 2: Attention Visualization (1 day)

**Goal:** Understand what BERT focuses on

**Tasks:**
1. Enable attention output in model
2. Extract attention weights
3. Visualize with BertViz
4. Analyze patterns across layers

**Key question:** Which Hindi words get most attention?

### Phase 3: SHAP/LIME Explanations (1-1.5 days)

**Goal:** Explain individual predictions

**Tasks:**
1. Implement SHAP for transformers
2. Add LIME approximations
3. Create word importance heatmaps
4. Compare explanation methods

**Key question:** Why did model predict this sentiment?

### Phase 4: Gradient Methods (1 day)

**Goal:** Neural-level importance

**Tasks:**
1. Implement Integrated Gradients
2. Compare with SHAP outputs
3. Find influential tokens

### Phase 5: Counterfactual Analysis (0.5 day)

**Goal:** Test robustness

**Tasks:**
1. Generate counterfactuals
2. Test negation handling
3. Identify biases

### Phase 6: Interactive Interface (1 day)

**Goal:** User-friendly demo

**Tasks:**
1. Build Gradio/Streamlit app
2. Show predictions + explanations
3. Deploy locally

### Phase 7: Documentation (1 day)

**Goal:** Professional report

**Tasks:**
1. Write findings
2. Create presentation
3. Document insights

---

## Best Practices

### 1. Always Version Your Experiments

```python
# Save with version number
save_model(model, tokenizer, 'models/v1_baseline/')
save_model(model, tokenizer, 'models/v2_balanced_data/')
```

### 2. Log Everything

```python
import json

experiment_log = {
    'model': config.MODEL_NAME,
    'batch_size': config.BATCH_SIZE,
    'learning_rate': config.LEARNING_RATE,
    'train_samples': len(train_df),
    'test_accuracy': test_accuracy,
    'timestamp': datetime.now().isoformat()
}

with open('outputs/experiment_log.json', 'w') as f:
    json.dump(experiment_log, f, indent=2)
```

### 3. Start Simple, Add Complexity

1. ✅ Get basic model working
2. ✅ Evaluate thoroughly
3. ✅ Add explainability
4. ✅ Optimize performance

### 4. Validate Assumptions

- Check data distribution
- Verify label quality
- Test edge cases
- Monitor for bias

---

## Summary

You now have:

✅ **Complete Phase 1 implementation**
✅ **Production-ready code structure**
✅ **Modular, reusable components**
✅ **Comprehensive documentation**
✅ **Ready for Phase 2-7 explainability**

**Time to run it!**

```bash
python run_phase1.py
```

Good luck! 🚀


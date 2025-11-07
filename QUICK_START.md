# Quick Start Guide: Explainable Hindi Sentiment Analysis

Welcome! This guide will help you start implementing the project step by step.

## Project Overview

You're building an **Explainable AI system** for Hindi sentiment analysis with 7 phases:
1. **Phase 1**: Train BERT sentiment classifier
2. **Phase 2**: Visualize attention patterns  
3. **Phase 3**: Add SHAP/LIME explanations
4. **Phase 4**: Gradient-based methods
5. **Phase 5**: Counterfactual analysis
6. **Phase 6**: Interactive visualization
7. **Phase 7**: Documentation

---

## Installation

### Step 1: Install Dependencies

```bash
cd "/home/nightfury653/Documents/NLP Project"
pip install -r requirements.txt
```

**Note:** This will take 5-10 minutes as it downloads large libraries.

---

## Phase 1: Training Your Model (START HERE!)

### Understanding the Architecture

**What we've built:**
```
Input Hindi Text → Tokenizer → BERT Model → Classification Head → Sentiment Prediction
                                     ↓
                           (Attention Patterns)
                                  We'll visualize these in Phase 2
```

### Step-by-Step Execution

#### 1. Start Jupyter Notebook

```bash
cd "/home/nightfury653/Documents/NLP Project"
jupyter notebook
```

#### 2. Navigate to:
```
notebooks/phase1_model_training.ipynb
```

#### 3. Run Each Cell in Order

The notebook is structured as:
- **Cell 1-2**: Setup and imports
- **Cell 3-4**: Load/create dataset
- **Cell 5-6**: Data preprocessing
- **Cell 7-8**: Train/val/test splits
- **Cell 9-10**: Initialize BERT model
- **Cell 11-12**: Training loop
- **Cell 13-14**: Evaluation and visualization

---

## Key Concepts Explained

### 1. **Why BERT for Hindi?**

BERT (Bidirectional Encoder Representations from Transformers):
- Pretrained on massive Hindi text corpus
- Understands context from both directions
- Already knows Hindi grammar and semantics
- We just fine-tune it for sentiment

**Models available:**
- `IndicBERT`: Best for Indian languages (recommended)
- `mBERT`: Multilingual, supports 104 languages
- `HindiBERT`: Hindi-specific

### 2. **Data Requirements**

**Minimum for demo:**
- 900-1000 samples (300 per class)
- Balanced distribution

**Recommended for real project:**
- 5,000-10,000+ samples
- Diverse sources (social media, reviews, news)
- Balanced classes

**Dataset structure:**
```csv
text,label
"यह फिल्म बहुत अच्छी है",positive
"मैं घर जा रहा हूं",neutral
"यह बहुत बुरा है",negative
```

### 3. **Training Process**

```
For each epoch:
    For each batch:
        1. Convert Hindi text to token IDs
        2. Feed tokens through BERT
        3. Get sentiment prediction
        4. Calculate loss (how wrong we are)
        5. Update model weights to improve
```

**Hyperparameters explained:**
- `learning_rate=2e-5`: Small because BERT is pretrained
- `batch_size=16`: Balance between speed and memory
- `epochs=3`: Usually sufficient for fine-tuning
- `max_length=128`: Maximum tokens per sentence

### 4. **Evaluation Metrics**

**Accuracy:**
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Precision:** Of all positive predictions, how many were actually positive?
**Recall:** Of all actual positive cases, how many did we find?
**F1-Score:** Harmonic mean of precision and recall

---

## Expected Timeline & Output

### Phase 1 Timeline:
- **Setup**: 30 minutes
- **Understanding code**: 1 hour
- **Training**: 30-60 minutes (depends on hardware)
- **Evaluation & testing**: 30 minutes
- **Total**: ~2.5-3 hours

### What You'll Have After Phase 1:

✅ Trained model saved in `models/`
✅ Tokenizer saved
✅ Training curves visualization
✅ Confusion matrix
✅ Classification report
✅ Ability to predict new Hindi sentences

---

## Common Issues & Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
pip install transformers torch pandas numpy scikit-learn matplotlib seaborn
```

### Issue 2: CUDA Out of Memory

**Problem:** GPU memory insufficient

**Solutions:**
1. Reduce batch_size in `config.py`:
   ```python
   BATCH_SIZE = 8  # or even 4
   ```

2. Use CPU (slower but works):
   ```python
   DEVICE = torch.device('cpu')
   ```

### Issue 3: Model Download Fails

**Problem:** Network issues downloading BERT

**Solution:**
```python
# Download manually first
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('ai4bharat/indic-bert')
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert')
```

### Issue 4: Low Accuracy (<60%)

**Causes:**
- Insufficient data
- Imbalanced classes
- Noisy labels
- Wrong hyperparameters

**Solutions:**
- Collect more data (aim for 5K+ samples)
- Balance classes
- Increase epochs to 4-5
- Clean dataset annotations

---

## Testing Your Model

After training, test with these Hindi sentences:

```python
test_sentences = [
    "यह फिल्म बहुत शानदार है",        # Positive
    "मुझे यह पसंद नहीं आया",           # Negative (with negation)
    "आज मौसम ठीक है",                   # Neutral
    "सेवा बिल्कुल घटिया थी",            # Strongly negative
    "यह किताब अच्छी है लेकिन महंगी है"  # Mixed sentiment
]
```

The model should correctly identify:
- Clear positive/negative sentiments
- Neutral statements
- Handle negation words ("नहीं")

---

## Configuration Reference

All settings are in `config.py`. Key parameters:

```python
# Model selection
MODEL_NAME = 'ai4bharat/indic-bert'  # Change if needed

# Training parameters
BATCH_SIZE = 16          # Reduce if memory issues
NUM_EPOCHS = 3           # Increase for better accuracy
LEARNING_RATE = 2e-5     # Standard for BERT fine-tuning
MAX_LENGTH = 128         # Increase for longer texts

# Paths
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
OUTPUT_DIR = 'outputs/'
```

---

## File Structure Explanation

```
NLP Project/
│
├── config.py                    # All settings in one place
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── QUICK_START.md              # This file
│
├── src/                         # Source code modules
│   ├── data_preprocessing.py   # Data loading and cleaning
│   ├── model.py                # BERT classifier
│   ├── explainability.py       # SHAP, LIME (Phase 3)
│   └── visualization.py        # Plotting functions (Phase 6)
│
├── notebooks/                   # Interactive notebooks
│   ├── phase1_model_training.ipynb
│   ├── phase2_attention_analysis.ipynb
│   ├── phase3_local_explainability.ipynb
│   └── ... (more phases)
│
├── data/                        # Dataset files
│   ├── raw/                    # Original data
│   ├── processed/              # Train/val/test splits
│   └── counterfactuals/        # Generated test cases
│
├── models/                      # Saved model checkpoints
└── outputs/                     # Visualizations and results
```

---

## Next Steps After Phase 1

Once your model is trained and showing good accuracy:

### Phase 2: Attention Visualization
- **Goal**: See what words BERT focuses on
- **Time**: 1 day
- **File**: `notebooks/phase2_attention_analysis.ipynb`

**You'll answer:**
- Which Hindi words get most attention?
- How does attention change across layers?
- Does attention align with human intuition?

### Phase 3: Word Importance (SHAP/LIME)
- **Goal**: Explain individual predictions
- **Time**: 1-1.5 days
- **Libraries**: SHAP, LIME

**You'll answer:**
- Why did model predict this sentiment?
- Which words contributed most?
- Can we trust the model's reasoning?

---

## Key Takeaways

### What Makes This Project Special:

1. **Explainability**: Not just "what" but "why"
2. **Multi-layered**: Multiple explanation methods
3. **Low-resource language**: Hindi NLP is challenging
4. **Production-ready**: Modular, well-documented code

### Skills You'll Learn:

- Transformer architectures (BERT)
- Transfer learning and fine-tuning
- Explainable AI techniques
- Hindi NLP challenges
- Model evaluation and error analysis
- Visualization best practices

---

## Getting Help

### Resources:

1. **Transformers Library**: https://huggingface.co/docs/transformers
2. **BERT Paper**: https://arxiv.org/abs/1810.04805
3. **SHAP Documentation**: https://shap.readthedocs.io
4. **Hindi NLP**: https://indicnlp.ai4bharat.org

### Debugging Tips:

1. **Start small**: Use 100 samples first to verify pipeline
2. **Check shapes**: Print tensor shapes to catch dimension errors
3. **Gradual complexity**: Get basic model working before adding explainability
4. **Save often**: Checkpoint models after each phase

---

## Ready to Start?

### Checklist:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset prepared (or using sample data)
- [ ] Reviewed configuration in `config.py`
- [ ] Jupyter notebook running
- [ ] Read this guide

### First Command:

```bash
cd "/home/nightfury653/Documents/NLP Project"
jupyter notebook notebooks/phase1_model_training.ipynb
```

---

## Questions to Consider During Training:

1. Is training loss decreasing steadily?
2. Is validation accuracy improving?
3. Are train and val losses close? (if not = overfitting)
4. Which sentiment class is hardest?
5. What patterns do you see in errors?

These observations will guide Phase 2-3 explainability analysis!

---

**Good luck with your implementation! 🚀**

Remember: Understanding comes from doing. Run each cell, read the outputs, and experiment!


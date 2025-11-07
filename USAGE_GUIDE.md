# Usage Guide

## Quick Start (5 Minutes)

### Step 1: Setup Environment

```bash
# Navigate to project
cd "/home/nightfury653/Documents/NLP Project"

# Activate virtual environment
source venv/bin/activate
```

### Step 2: Verify Model

```bash
# Check if model exists
ls -lh models/model.pt
```

If model doesn't exist, train it:
```bash
python run_phase1.py  # Takes ~10-15 minutes
```

### Step 3: Launch Interactive Interface

```bash
python run_interactive.py
```

Open browser to: **http://localhost:7860**

---

## Detailed Workflows

### Workflow 1: Complete Pipeline Analysis

Run all analysis phases in sequence:

```bash
# Activate environment
source venv/bin/activate

# Run all phases
python run_phase2.py  # Attention Analysis
python run_phase3.py  # SHAP/LIME
python run_phase4.py  # Gradient Methods
python run_phase5.py  # Counterfactuals

# View results
ls -R outputs/
```

**Time**: ~5-10 minutes total  
**Output**: All visualizations in `outputs/` directory

### Workflow 2: Single Text Analysis

Analyze a single text with all methods:

```python
# save as analyze_text.py
import torch
from transformers import AutoTokenizer
from pathlib import Path

from src.model import BERTSentimentClassifier
from src.attention_analysis import AttentionAnalyzer
from src.shap_lime_explainer import SHAP_LIME_Explainer
from src.gradient_explainer import GradientExplainer
from src.counterfactual_analyzer import CounterfactualAnalyzer
from config import MODEL_NAME, NUM_LABELS, DEVICE

# Load model
model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load('models/model.pt', map_location=DEVICE))
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Your text
text = "यह फिल्म बहुत अच्छी थी।"

# Basic prediction
cf_analyzer = CounterfactualAnalyzer(model, tokenizer)
pred = cf_analyzer.predict(text)
print(f"Prediction: {pred['predicted_label']} ({pred['confidence']:.2%})")

# Attention analysis
attention = AttentionAnalyzer(model, tokenizer)
att_result = attention.get_token_importance(text, merge_subwords=True)
print("\nTop tokens (Attention):")
for word, score in zip(att_result['merged_words'][:5], 
                       att_result['merged_scores'][:5]):
    print(f"  {word}: {score:.3f}")

# SHAP analysis
shap_lime = SHAP_LIME_Explainer(model, tokenizer)
shap_result = shap_lime.explain_with_shap(text, merge_subwords=True)
print("\nTop tokens (SHAP):")
for word, score in zip(shap_result['merged_words'][:5],
                       shap_result['merged_scores'][:5]):
    print(f"  {word}: {score:.3f}")
```

Run with:
```bash
python analyze_text.py
```

### Workflow 3: Batch Analysis

Analyze multiple texts:

```python
# save as batch_analyze.py
import torch
from transformers import AutoTokenizer
from src.model import BERTSentimentClassifier
from src.counterfactual_analyzer import CounterfactualAnalyzer
from config import MODEL_NAME, NUM_LABELS, DEVICE

# Load model
model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load('models/model.pt', map_location=DEVICE))
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
analyzer = CounterfactualAnalyzer(model, tokenizer)

# Texts to analyze
texts = [
    "यह फिल्म बहुत अच्छी थी।",
    "खाना ठीक-ठाक था।",
    "सेवा बहुत खराब थी।"
]

# Analyze all
results = []
for text in texts:
    pred = analyzer.predict(text)
    results.append({
        'text': text,
        'sentiment': pred['predicted_label'],
        'confidence': pred['confidence']
    })

# Display results
print("\nBatch Analysis Results:")
print("=" * 70)
for i, result in enumerate(results, 1):
    print(f"\n{i}. {result['text']}")
    print(f"   → {result['sentiment']} ({result['confidence']:.2%})")
```

Run with:
```bash
python batch_analyze.py
```

---

## Interactive Interface Usage

### Starting the Interface

```bash
# From project root
source venv/bin/activate
python run_interactive.py
```

### Using the Interface

1. **Enter Text**: Type or paste Hindi text in the input box
2. **Click "Analyze Sentiment"**: Wait 5-10 seconds
3. **View Results**:
   - Prediction with confidence scores
   - Attention visualization
   - SHAP explanation
   - LIME explanation
   - Gradient saliency
   - Counterfactual examples

### Example Inputs

**Positive**:
```
यह फिल्म बहुत अच्छी थी, मुझे बेहद पसंद आई।
```

**Neutral**:
```
खाना ठीक-ठाक था, कुछ खास नहीं।
```

**Negative**:
```
सेवा बहुत खराब थी, बिल्कुल समय की बर्बादी।
```

### Interface Features

- **Real-time Analysis**: Instant predictions
- **Multiple Methods**: All 5 explainability methods
- **Visual Output**: High-quality plots
- **Example Gallery**: Click examples to try
- **Responsive**: Works on desktop and tablet

### Sharing the Interface

To create a public URL (accessible from anywhere):

1. Edit `run_interactive.py`
2. Change `share=False` to `share=True`
3. Restart the interface
4. Get a public `https://xxx.gradio.app` URL

⚠️ **Note**: Public URLs expire after 72 hours

---

## Command-Line Analysis

### Phase 2: Attention Analysis

```bash
python run_phase2.py
```

**Outputs** (in `outputs/phase2/`):
- `sample_1_attention_heatmap.png`
- `sample_1_attention_summary.png`
- `sample_1_token_importance.png`
- `sample_1_attention_flow.png`
- (Similar files for samples 2 and 3)

**What to look for**:
- Which words have highest attention
- How attention flows between tokens
- Layer-wise attention patterns

### Phase 3: SHAP/LIME Analysis

```bash
python run_phase3.py
```

**Outputs** (in `outputs/phase3/`):
- `sample_1_shap.png`
- `sample_1_lime.png`
- `sample_1_comparison.png`
- (Similar files for samples 2 and 3)

**What to look for**:
- Positive vs. negative contributions
- Agreement between SHAP and LIME
- Word-level importance scores

### Phase 4: Gradient Analysis

```bash
python run_phase4.py
```

**Outputs** (in `outputs/phase4/`):
- `sample_1_saliency.png`
- `sample_1_integrated_gradients.png`
- `sample_1_grad_x_input.png`
- `sample_1_comparison.png`
- (Similar files for samples 2 and 3)

**What to look for**:
- Consistency across methods
- Gradient-based importance
- Top influential tokens

### Phase 5: Counterfactual Analysis

```bash
python run_phase5.py
```

**Outputs** (in `outputs/phase5/`):
- `sample_1_counterfactuals.png`
- `sample_1_probability_comparison.png`

**What to look for**:
- What changes flip predictions
- Model sensitivity to words
- Decision boundaries

---

## Viewing Results

### View All Outputs

```bash
# List all generated visualizations
find outputs/ -name "*.png" | sort

# Count total files
find outputs/ -name "*.png" | wc -l
```

### Open Specific Visualization

```bash
# On Linux with default image viewer
xdg-open outputs/phase2/sample_1_attention_heatmap.png

# Or use any image viewer
eog outputs/phase2/sample_1_attention_heatmap.png
```

### Create Summary Report

```bash
# Generate file listing
ls -lh outputs/phase2/*.png
ls -lh outputs/phase3/*.png
ls -lh outputs/phase4/*.png
ls -lh outputs/phase5/*.png
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution**:
```bash
# Train the model first
python run_phase1.py
```

### Issue: "CUDA out of memory"

**Solution**: Edit `config.py` and reduce batch size:
```python
BATCH_SIZE = 2  # Or even 1
```

### Issue: Hindi text shows as boxes

**Solution**: Install Hindi fonts:
```bash
sudo apt-get install fonts-noto fonts-noto-core
fc-cache -fv  # Rebuild font cache
```

### Issue: "Module not found"

**Solution**: Activate virtual environment:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Interface won't start

**Solution**: Check if port 7860 is already in use:
```bash
# Find process on port 7860
lsof -i :7860

# Kill it if needed
kill -9 <PID>

# Or use different port
# Edit run_interactive.py and change server_port
```

---

## Performance Tips

### Speed Up Analysis

1. **Use GPU**: Ensure CUDA is available
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. **Reduce Samples**: Process fewer examples
   ```python
   # In run_phase*.py, modify test_samples list
   test_samples = test_samples[:1]  # Only first sample
   ```

3. **Skip Heavy Methods**: Comment out slow analyses
   ```python
   # In run scripts, comment out:
   # explainer.compute_integrated_gradients(...)  # Slowest
   ```

### Save Memory

1. **Smaller Batch Size**: Edit `config.py`
   ```python
   BATCH_SIZE = 1
   ```

2. **Clear Cache**: Between runs
   ```python
   torch.cuda.empty_cache()
   ```

3. **Use CPU**: If GPU memory insufficient
   ```python
   # In config.py
   DEVICE = 'cpu'
   ```

---

## Advanced Usage

### Custom Dataset

To use your own data:

1. **Prepare data** in format:
   ```
   text,label
   "यह अच्छा है",positive
   "यह बुरा है",negative
   ```

2. **Modify** `src/data_preprocessing.py`:
   ```python
   # Update create_synthetic_dataset() to load your CSV
   ```

3. **Retrain**:
   ```bash
   python run_phase1.py
   ```

### Export Results

Save analysis results to file:

```python
import json

# After analysis
results = {
    'text': text,
    'prediction': pred,
    'attention': att_result,
    'shap': shap_result
}

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

### Integrate into Application

Use as library in your code:

```python
from your_app import get_user_input
from src.model import BERTSentimentClassifier
from src.counterfactual_analyzer import CounterfactualAnalyzer
import torch

# Load once at startup
model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load('models/model.pt'))
model.eval()

analyzer = CounterfactualAnalyzer(model, tokenizer)

# Use in application
user_text = get_user_input()
result = analyzer.predict(user_text)

# Process result
if result['predicted_label'] == 'Positive':
    # Handle positive sentiment
    pass
elif result['predicted_label'] == 'Negative':
    # Handle negative sentiment
    pass
```

---

## Best Practices

### 1. Always Use Virtual Environment
```bash
source venv/bin/activate  # First!
```

### 2. Check GPU Status
```bash
nvidia-smi  # Before running
```

### 3. Save Important Results
```bash
# Backup outputs
cp -r outputs/ outputs_backup_$(date +%Y%m%d)/
```

### 4. Monitor Resources
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch CPU/RAM
htop
```

### 5. Version Control
```bash
# Save your modifications
git init
git add .
git commit -m "Custom changes"
```

---

## Getting Help

1. **Check Documentation**:
   - README.md (overview)
   - FINAL_REPORT.md (technical details)
   - This file (usage guide)

2. **Review Code Comments**: All functions are documented

3. **Test Examples**: Run provided sample scripts

4. **Check Logs**: Review `training.log` for errors

---

## Summary

**For Quick Analysis**: Use interactive interface
```bash
python run_interactive.py
```

**For Complete Pipeline**: Run all phases
```bash
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py
```

**For Custom Integration**: Import modules in your code

---

**Last Updated**: November 7, 2025  
**Project Status**: Production Ready ✅


# Quick Reference Card

## 🎯 3-Minute Quick Start

```bash
# 1. Install (one time, 5-10 minutes)
cd "/home/nightfury653/Documents/NLP Project"
pip install -r requirements.txt

# 2. Run Phase 1 (15-30 minutes)
python run_phase1.py

# Done! Check outputs/ folder for results
```

---

## 📚 Which File Should I Read?

### Just Starting? → **START_HERE.md**
- Complete beginner's guide
- What has been built
- Step-by-step instructions

### Want Quick Setup? → **QUICK_START.md**
- Installation guide
- Common issues & solutions
- Configuration reference

### Need Technical Details? → **IMPLEMENTATION_GUIDE.md**
- How each module works
- Code explanations
- Architecture deep dive

### Want Overview? → **PROJECT_SUMMARY.md**
- What's been delivered
- Expected performance
- Next phases overview

### Need This Now? → **QUICK_REFERENCE.md**
- This file - quick commands
- File purposes
- Common tasks

---

## 📁 File Purposes at a Glance

| File | Purpose | When to Use |
|------|---------|-------------|
| `config.py` | All settings | Change hyperparameters |
| `run_phase1.py` | Complete workflow | Run Phase 1 |
| `src/data_preprocessing.py` | Data utilities | Understand data handling |
| `src/model.py` | BERT classifier | Understand model |
| `notebooks/phase1_*.ipynb` | Interactive | Learn step-by-step |
| `requirements.txt` | Dependencies | First time setup |

---

## ⚙️ Common Configuration Changes

### In `config.py`:

```python
# Use CPU instead of GPU
DEVICE = torch.device('cpu')

# Reduce memory usage
BATCH_SIZE = 8  # or 4

# Train longer
NUM_EPOCHS = 5

# Different model
MODEL_NAME = MODEL_OPTIONS['mbert']

# Longer texts
MAX_LENGTH = 256
```

---

## 🎯 Common Tasks

### Task: Use My Own Dataset

**Step 1:** Prepare CSV:
```csv
text,label
"यह अच्छी है",positive
"यह बुरी है",negative
```

**Step 2:** Edit `run_phase1.py` line ~70:
```python
df = pd.read_csv('data/raw/my_data.csv')
```

**Step 3:** Run:
```bash
python run_phase1.py
```

### Task: Make Predictions

```python
from src.model import load_model, predict_text
import torch

model, tokenizer = load_model(
    'ai4bharat/indic-bert',
    'models/',
    device=torch.device('cpu')
)

_, _, sentiment = predict_text(
    model,
    "यह फिल्म शानदार है",
    tokenizer,
    torch.device('cpu')
)
print(sentiment)  # 'positive'
```

### Task: Check Accuracy

After running `run_phase1.py`, check terminal output:
```
Test Accuracy: 0.7456 (74.56%)
```

### Task: View Training Curves

Open: `outputs/phase1_training_curves.png`

### Task: See Confusion Matrix

Open: `outputs/phase1_confusion_matrix.png`

### Task: Modify Hyperparameters

Edit `config.py`, then rerun:
```bash
python run_phase1.py
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `CUDA out of memory` | Set `BATCH_SIZE = 4` in config.py |
| Training too slow | Normal on CPU, use GPU if available |
| Low accuracy (<60%) | Need more/better data |
| Model file not found | Run `python run_phase1.py` first |

---

## 📊 What Good Results Look Like

### Training Output
```
✅ Good:
Epoch 1: Train=0.95, Val=0.89, Acc=65%
Epoch 2: Train=0.68, Val=0.62, Acc=73%
Epoch 3: Train=0.51, Val=0.48, Acc=78%

❌ Bad (Overfitting):
Epoch 1: Train=0.95, Val=0.89, Acc=65%
Epoch 2: Train=0.42, Val=0.95, Acc=62%
Epoch 3: Train=0.18, Val=1.23, Acc=58%
```

### Accuracy Benchmarks
- 60-70%: Baseline (ok for demo)
- 70-80%: Good
- 80-85%: Very good
- 85%+: Excellent

---

## 🎓 Learning Path

### Beginner Path (3 hours)
1. Read START_HERE.md (15 min)
2. Run `python run_phase1.py` (30 min)
3. Open `notebooks/phase1_*.ipynb` (2 hours)
4. Experiment with predictions (15 min)

### Advanced Path (1 hour)
1. Skim QUICK_START.md (5 min)
2. Run Phase 1 (30 min)
3. Read IMPLEMENTATION_GUIDE.md (20 min)
4. Modify hyperparameters and rerun (varies)

---

## 🚀 Phase 2-7 Quick Overview

| Phase | Goal | Time | Status |
|-------|------|------|--------|
| **1** | **Train model** | **2-3h** | **✅ READY** |
| 2 | Attention viz | 1 day | 📝 Pending |
| 3 | SHAP/LIME | 1-1.5 days | 📝 Pending |
| 4 | Gradients | 1 day | 📝 Pending |
| 5 | Counterfactuals | 0.5 day | 📝 Pending |
| 6 | Interface | 1 day | 📝 Pending |
| 7 | Documentation | 1 day | 📝 Pending |

**Total: ~7 days for complete project**

---

## 💡 Pro Tips

### Tip 1: Start Small
```bash
# First run with sample data (fast)
python run_phase1.py

# Then with your data (slower but better)
# Edit run_phase1.py to load your CSV
```

### Tip 2: Save Experiments
```bash
# Version your models
cp -r models/ models_v1_baseline/
cp -r models/ models_v2_more_data/
```

### Tip 3: Log Everything
```python
# Add to your experiments
with open('experiment_log.txt', 'a') as f:
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
```

### Tip 4: Use Notebooks for Learning
```bash
# Better for understanding
jupyter notebook notebooks/phase1_model_training.ipynb
```

### Tip 5: Check Outputs Folder
```bash
ls outputs/
# phase1_training_curves.png
# phase1_confusion_matrix.png
```

---

## 🎯 Success Checklist

After running Phase 1, you should have:

- [ ] No errors during execution
- [ ] `models/model.pt` exists
- [ ] `outputs/` contains 2 PNG files
- [ ] Test accuracy printed in terminal
- [ ] Can run predictions on new text
- [ ] Understanding of how it works

**If all checked → Phase 1 complete! 🎉**

---

## 📞 Need Help?

### Error During Installation?
→ Check **QUICK_START.md** Section "Troubleshooting"

### Error During Training?
→ Check **IMPLEMENTATION_GUIDE.md** Section "Troubleshooting Common Issues"

### Want to Understand Code Better?
→ Read **IMPLEMENTATION_GUIDE.md** Section "Module Breakdown"

### Ready for Phase 2?
→ Open **START_HERE.md** Section "What's Next"

---

## 🎬 The One Command You Need

```bash
python run_phase1.py
```

That's it! Everything else is optional reading and customization.

---

## 📈 Interpreting Your Results

### Terminal Output
```
Test Accuracy: 0.7456 (74.56%)  ← Main metric
```

### Training Curves
- **Left plot**: Loss should decrease
- **Right plot**: Accuracy should increase

### Confusion Matrix
- **Diagonal** (dark blue): Correct predictions
- **Off-diagonal** (light): Mistakes
- **Neutral** often confused (normal)

### Classification Report
```
           precision  recall  f1-score
Negative      0.78     0.80     0.79  ← Per-class metrics
Neutral       0.65     0.60     0.62  ← Usually lowest
Positive      0.81     0.83     0.82  ← Usually highest
```

---

## 🏃‍♂️ Let's Go!

**Everything is ready. Just run:**

```bash
cd "/home/nightfury653/Documents/NLP Project"
python run_phase1.py
```

**Then check:**
- Terminal for accuracy
- `outputs/` for visualizations
- `models/` for saved model

**Finally:**
- Read outputs carefully
- Understand what worked/didn't
- Get ready for Phase 2!

---

**🎉 You've got this! Time to build your explainable Hindi sentiment analyzer!**


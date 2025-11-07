# 🎭 Explainable Hindi Sentiment Analysis with BERT

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive **Explainable AI (XAI)** system for Hindi sentiment analysis using BERT-based models. This project implements **5 complementary interpretability methods** to provide transparent, trustworthy sentiment predictions.

## ✨ Features

- 🎯 **High-Performance Model**: BERT-based 3-class sentiment classifier (Negative, Neutral, Positive)
- 🔍 **5 Explainability Methods**: Attention, SHAP, LIME, Gradients, Counterfactuals
- 🌐 **Interactive Web Interface**: Gradio-based dashboard for real-time analysis
- 🇮🇳 **Hindi Language Support**: Full Devanagari script rendering in visualizations
- 🚀 **GPU Accelerated**: Optimized for CUDA-enabled GPUs
- 📊 **Rich Visualizations**: Professional plots for all explanation methods
- 📚 **Well-Documented**: Comprehensive guides and code comments

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# Clone or navigate to project directory
cd "NLP Project"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Hindi font support (Linux)
sudo apt-get install fonts-noto fonts-noto-core
```

### Train Model

```bash
# Train BERT sentiment classifier (Phase 1)
python run_phase1.py
```

### Run Interactive Interface

```bash
# Launch web interface
python run_interactive.py

# Open browser to: http://localhost:7860
```

## 📋 Project Structure

```
NLP Project/
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── model.py                   # BERT sentiment classifier
│   ├── attention_analysis.py      # Attention visualization
│   ├── shap_lime_explainer.py     # SHAP & LIME explanations
│   ├── gradient_explainer.py      # Gradient-based methods
│   ├── counterfactual_analyzer.py # Counterfactual generation
│   └── interactive_interface.py   # Gradio web interface
│
├── outputs/                       # Generated visualizations
│   ├── phase2/                   # Attention heatmaps
│   ├── phase3/                   # SHAP/LIME plots
│   ├── phase4/                   # Gradient visualizations
│   └── phase5/                   # Counterfactual examples
│
├── models/                        # Trained model weights
│   ├── model.pt                  # Model state dict
│   └── tokenizer files...
│
├── config.py                      # Centralized configuration
├── requirements.txt               # Python dependencies
│
├── run_phase1.py                  # Training script
├── run_phase2.py                  # Attention analysis
├── run_phase3.py                  # SHAP/LIME analysis
├── run_phase4.py                  # Gradient analysis
├── run_phase5.py                  # Counterfactual analysis
├── run_interactive.py             # Launch web interface
│
├── README.md                      # This file
├── FINAL_REPORT.md               # Comprehensive technical report
└── training.log                   # Training logs
```

## 🎓 Implementation Phases

### Phase 1: Model Training ✅
**Script**: `run_phase1.py`

- Trains BERT-based sentiment classifier
- 3-class classification (Negative, Neutral, Positive)
- GPU-accelerated training
- Saves model weights and tokenizer

**Output**: Trained model in `models/model.pt`

### Phase 2: Attention Analysis ✅
**Script**: `run_phase2.py`

- Extracts attention weights from BERT layers
- Aggregates multi-head attention
- Creates attention heatmaps and token importance plots
- Merges subword tokens for readability

**Output**: Visualizations in `outputs/phase2/`

### Phase 3: SHAP/LIME Explainability ✅
**Script**: `run_phase3.py`

- SHAP: Game-theoretic feature attribution
- LIME: Local interpretable explanations
- Comparison plots between methods
- Token-level importance scores

**Output**: Visualizations in `outputs/phase3/`

### Phase 4: Gradient-based Interpretability ✅
**Script**: `run_phase4.py`

Three gradient-based methods:
1. **Saliency Maps**: Gradient magnitude
2. **Integrated Gradients**: Path integral attribution
3. **Gradient × Input**: Combined importance

**Output**: Visualizations in `outputs/phase4/`

### Phase 5: Counterfactual Analysis ✅
**Script**: `run_phase5.py`

Generates alternative inputs that flip predictions:
- Word replacement (antonyms)
- Word removal
- Negation addition/removal

**Output**: Counterfactual examples in `outputs/phase5/`

### Phase 6: Interactive Interface ✅
**Script**: `run_interactive.py`

Gradio-based web application featuring:
- Real-time sentiment analysis
- All 5 explainability methods
- Interactive visualizations
- Example inputs
- Professional UI

**Access**: http://localhost:7860

### Phase 7: Documentation ✅
**File**: `FINAL_REPORT.md`

Comprehensive technical report including:
- Architecture details
- Implementation notes
- Results and performance
- Challenges solved
- Future enhancements

## 🔍 Explainability Methods

### 1. Attention Analysis
**What it shows**: Which tokens BERT pays attention to  
**Method**: Extracts and aggregates attention weights across layers  
**Speed**: Fast (~100ms)  
**Best for**: Understanding model focus patterns

### 2. SHAP (SHapley Additive exPlanations)
**What it shows**: Game-theoretic contribution of each token  
**Method**: Shapley value computation with masking  
**Speed**: Medium (~2s)  
**Best for**: Mathematically grounded attributions

### 3. LIME (Local Interpretable Model-agnostic Explanations)
**What it shows**: Local linear approximation  
**Method**: Perturb inputs and fit surrogate model  
**Speed**: Medium (~3s)  
**Best for**: Model-agnostic quick analysis

### 4. Gradient-based Methods
**What they show**: Input importance via gradients  
**Methods**: 
- Saliency Maps (gradient magnitude)
- Integrated Gradients (path attribution)
- Gradient × Input (combined)  
**Speed**: Varies (100ms - 5s)  
**Best for**: Neural network specific insights

### 5. Counterfactual Analysis
**What it shows**: Minimal changes to flip predictions  
**Method**: Systematic word replacement/removal  
**Speed**: Fast (~1s)  
**Best for**: Understanding decision boundaries

## 📊 Example Usage

### Command Line Analysis

```bash
# Run complete analysis pipeline
python run_phase2.py  # Attention
python run_phase3.py  # SHAP/LIME
python run_phase4.py  # Gradients
python run_phase5.py  # Counterfactuals
```

### Python API Usage

```python
from src.model import BERTSentimentClassifier
from src.attention_analysis import AttentionAnalyzer
from transformers import AutoTokenizer
import torch

# Load model
model = BERTSentimentClassifier('bert-base-multilingual-cased', 3)
model.load_state_dict(torch.load('models/model.pt'))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

# Initialize analyzer
analyzer = AttentionAnalyzer(model, tokenizer)

# Analyze text
text = "यह फिल्म बहुत अच्छी थी।"
result = analyzer.get_token_importance(text)

print(result['merged_words'])
print(result['merged_scores'])
```

### Interactive Web Interface

```bash
# Launch interface
python run_interactive.py

# Then open browser to http://localhost:7860
# Enter Hindi text and click "Analyze Sentiment"
```

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_NAME = 'bert-base-multilingual-cased'
NUM_LABELS = 3
MAX_LENGTH = 128

# Training settings
BATCH_SIZE = 4  # Adjust based on GPU memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# Device settings
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 🎯 Performance

### Model Metrics
- Training Accuracy: >95%
- Convergence: 3 epochs
- Inference Speed: ~50ms per sample

### Hardware Requirements
- **Minimum**: 4GB GPU, 8GB RAM
- **Recommended**: 8GB+ GPU, 16GB RAM
- **Tested on**: RTX 3050 4GB

### Explanation Speed (RTX 3050)
| Method | Time per Sample |
|--------|----------------|
| Prediction | 50ms |
| Attention | 100ms |
| SHAP | 2s |
| LIME | 3s |
| Integrated Gradients | 5s |
| Counterfactuals | 1s |

## 🛠️ Technical Challenges Solved

### 1. Hindi Font Rendering
**Problem**: Matplotlib displayed boxes instead of Hindi text  
**Solution**: Installed Noto Sans Devanagari, hybrid font approach

### 2. Subword Token Merging
**Problem**: BERT tokenizes into subwords (e.g., `##क`)  
**Solution**: Implemented merging algorithm with score aggregation

### 3. GPU Memory Optimization
**Problem**: 4GB GPU insufficient for default batch size  
**Solution**: Reduced batch size to 4, optimized gradient computation

### 4. Gradient Retention
**Problem**: Non-leaf tensor gradients not retained  
**Solution**: Added `.retain_grad()` and `retain_graph=True`

### 5. Model Authentication
**Problem**: Some models require HuggingFace login  
**Solution**: Used publicly available multilingual BERT

## 🌟 Use Cases

- ✅ **Customer Feedback Analysis**: Understand sentiment in Hindi reviews
- ✅ **Social Media Monitoring**: Track brand sentiment on Hindi platforms
- ✅ **Product Review Classification**: Automated sentiment labeling
- ✅ **Content Moderation**: Identify negative/harmful content
- ✅ **Market Research**: Analyze consumer opinions
- ✅ **Chatbot Enhancement**: Sentiment-aware responses

## 📚 Documentation

- **[FINAL_REPORT.md](FINAL_REPORT.md)**: Comprehensive technical report
- **[config.py](config.py)**: Configuration reference
- **Code Comments**: Detailed inline documentation
- **Docstrings**: Function-level documentation

## 🔮 Future Enhancements

### Model Improvements
- [ ] Fine-tune on larger Hindi corpus
- [ ] Multi-task learning (sentiment + emotion)
- [ ] Cross-lingual transfer learning
- [ ] Ensemble methods

### Explainability Extensions
- [ ] Layer-wise relevance propagation (LRP)
- [ ] Influence functions
- [ ] Concept activation vectors
- [ ] Interactive attention manipulation

### Deployment
- [ ] Docker containerization
- [ ] REST API endpoint
- [ ] Cloud deployment (AWS/GCP)
- [ ] Model versioning
- [ ] A/B testing framework

### Interface Enhancements
- [ ] Batch processing
- [ ] Export to PDF/HTML
- [ ] Comparison mode
- [ ] User feedback collection
- [ ] Performance monitoring dashboard

## 🤝 Contributing

This is an educational project demonstrating XAI techniques for Hindi NLP. Contributions, suggestions, and improvements are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **Hugging Face**: Transformers library and pre-trained models
- **PyTorch**: Deep learning framework
- **SHAP/LIME**: Explainability libraries
- **Gradio**: Interactive interface framework
- **Google Fonts**: Noto Sans Devanagari font

## 📧 Contact

For questions, issues, or collaboration:
- Review documentation in project files
- Check code comments
- Test with provided examples

## 🎖️ Project Status

✅ **All Phases Complete**
- [x] Phase 1: Model Training
- [x] Phase 2: Attention Analysis  
- [x] Phase 3: SHAP/LIME Explainability
- [x] Phase 4: Gradient-based Interpretability
- [x] Phase 5: Counterfactual Analysis
- [x] Phase 6: Interactive Interface
- [x] Phase 7: Documentation

**Status**: Production Ready 🚀  
**Last Updated**: November 7, 2025

---

<div align="center">

**Built with ❤️ for Transparent and Trustworthy AI**

[Documentation](FINAL_REPORT.md) • [Installation](#installation) • [Quick Start](#quick-start)

</div>

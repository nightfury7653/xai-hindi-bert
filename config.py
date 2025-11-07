"""
Configuration file for the Hindi Sentiment Analysis project.
All hyperparameters and settings in one place for easy modification.
"""

import torch

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection - Choose one of these pre-trained models
MODEL_OPTIONS = {
    'indicbert': 'ai4bharat/indic-bert',           # Requires HuggingFace login (gated)
    'mbert': 'bert-base-multilingual-cased',       # Google's multilingual BERT (RECOMMENDED - no login needed)
    'hindibert': 'neuralspace-reverie/indic-transformers-hi-bert'  # Hindi-specific
}

# Selected model for this project
MODEL_NAME = MODEL_OPTIONS['mbert']  # Using mBERT - freely available, excellent for Hindi

# Model parameters
MAX_LENGTH = 128          # Maximum sequence length for tokenization
NUM_LABELS = 3            # Positive, Negative, Neutral
HIDDEN_DROPOUT = 0.1      # Dropout rate
ATTENTION_DROPOUT = 0.1   # Attention dropout rate

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

BATCH_SIZE = 4            # Batch size for training (reduced for 4GB GPU)
EVAL_BATCH_SIZE = 8       # Batch size for evaluation
NUM_EPOCHS = 3            # Number of training epochs
LEARNING_RATE = 2e-5      # Learning rate for optimizer
WARMUP_STEPS = 100        # Warmup steps for learning rate scheduler
WEIGHT_DECAY = 0.01       # Weight decay for AdamW optimizer

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Label mapping
LABEL_MAP = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# Data split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============================================================================
# PATHS
# ============================================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# EXPLAINABILITY CONFIGURATION
# ============================================================================

# Number of samples to explain in detail
NUM_EXPLANATION_SAMPLES = 10

# Attention visualization settings
ATTENTION_LAYERS = [0, 5, 11]  # First, middle, and last layers
ATTENTION_HEADS = 'all'         # or specify list like [0, 1, 2]

# SHAP settings
SHAP_NUM_SAMPLES = 100         # Number of samples for SHAP background

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Color scheme for visualizations
COLORS = {
    'positive': '#4CAF50',      # Green
    'neutral': '#9E9E9E',       # Gray
    'negative': '#F44336',      # Red
    'background': '#FFFFFF',    # White
    'text': '#000000'           # Black
}

# Font for Hindi text
HINDI_FONT = 'Noto Sans Devanagari'

# ============================================================================
# COUNTERFACTUAL CONFIGURATION
# ============================================================================

# Words for counterfactual testing
COUNTERFACTUAL_WORDS = {
    'negation': ['नहीं', 'मत', 'न', 'कभी नहीं'],
    'intensifiers': ['बहुत', 'अत्यधिक', 'काफी', 'बेहद', 'अत्यंत'],
    'diminishers': ['थोड़ा', 'कम', 'लगभग', 'कुछ'],
    'positive_adjectives': ['अच्छा', 'सुंदर', 'खुश', 'प्रसन्न', 'उत्कृष्ट', 'शानदार'],
    'negative_adjectives': ['बुरा', 'भद्दा', 'दुखी', 'निराश', 'खराब', 'घटिया']
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_INTERVAL = 10             # Log every N batches during training
SAVE_BEST_MODEL = True        # Save model with best validation accuracy
EARLY_STOPPING_PATIENCE = 3   # Stop if no improvement for N epochs

print(f"Configuration loaded successfully!")
print(f"Model: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Max Length: {MAX_LENGTH}")
print(f"Batch Size: {BATCH_SIZE}")


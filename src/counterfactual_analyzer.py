"""
Phase 5: Counterfactual Analysis
=================================
Generates counterfactual examples showing what changes would flip predictions.
Techniques:
1. Word Replacement - Replace words with antonyms/alternatives
2. Word Removal - Remove influential words
3. Word Addition - Add sentiment-changing words
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path
import sys
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import DEVICE, MODEL_NAME, MAX_LENGTH


# ============================================================================
# Font Setup for Hindi Text
# ============================================================================

def setup_hindi_font():
    """
    Configure matplotlib to properly render Hindi (Devanagari) text.
    Uses a hybrid approach:
    - DejaVu Sans (default) for English text
    - Noto Sans Devanagari for Hindi text
    """
    hindi_fonts = [
        'Noto Sans Devanagari',
        'Lohit Devanagari',
        'Mangal',
        'Devanagari MT'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in hindi_fonts:
        if font in available_fonts:
            global HINDI_FONT
            HINDI_FONT = font
            print(f"✓ Using '{font}' for Hindi text rendering")
            break
    else:
        HINDI_FONT = 'DejaVu Sans'
        print("⚠ Warning: No Hindi font found.")
    
    # Set DejaVu Sans as default for English
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['font.size'] = 10
    rcParams['axes.unicode_minus'] = False

# Initialize fonts
HINDI_FONT = 'DejaVu Sans'
setup_hindi_font()


# ============================================================================
# Hindi Sentiment Word Lists
# ============================================================================

# Positive to Negative replacements
SENTIMENT_REPLACEMENTS = {
    # Positive -> Negative
    'अच्छी': ['खराब', 'बुरी', 'घटिया'],
    'अच्छा': ['खराब', 'बुरा', 'घटिया'],
    'बेहतरीन': ['खराब', 'भयानक', 'निराशाजनक'],
    'शानदार': ['खराब', 'भयानक', 'बेकार'],
    'बढ़िया': ['खराब', 'निराशाजनक', 'घटिया'],
    'पसंद': ['नापसंद', 'घृणा'],
    'सुंदर': ['बदसूरत', 'भद्दा'],
    'प्यारा': ['घिनौना', 'बेकार'],
    'उत्कृष्ट': ['खराब', 'भयंकर'],
    'बेहद': ['बिल्कुल नहीं', 'कम'],
    'महान': ['घटिया', 'तुच्छ'],
    'उम्दा': ['खराब', 'घटिया'],
    
    # Negative -> Positive
    'खराब': ['अच्छा', 'बेहतरीन', 'शानदार'],
    'बुरा': ['अच्छा', 'बेहतरीन'],
    'बुरी': ['अच्छी', 'बढ़िया'],
    'घटिया': ['बेहतरीन', 'शानदार', 'उत्कृष्ट'],
    'भयानक': ['शानदार', 'बेहतरीन'],
    'नापसंद': ['पसंद'],
    'घृणा': ['प्यार', 'पसंद'],
    'बदसूरत': ['सुंदर'],
    'बेकार': ['बढ़िया', 'शानदार'],
    'निराशाजनक': ['उत्साहजनक', 'शानदार'],
    'तुच्छ': ['महान', 'उत्कृष्ट'],
    'समय की बर्बादी': ['समय का सदुपयोग', 'बेहतरीन'],
    'बर्बादी': ['लाभ', 'फायदा'],
}

# Negation words
NEGATION_WORDS = ['नहीं', 'न', 'बिल्कुल', 'कभी', 'मत']

# Sentiment intensifiers
POSITIVE_INTENSIFIERS = ['बेहद', 'बहुत', 'अत्यधिक', 'बेहतरीन', 'जबरदस्त']
NEGATIVE_INTENSIFIERS = ['बिल्कुल', 'बहुत', 'अत्यधिक', 'पूरी तरह']


# ============================================================================
# Counterfactual Analyzer Class
# ============================================================================

class CounterfactualAnalyzer:
    """
    Generates and analyzes counterfactual examples.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the counterfactual analyzer.
        
        Args:
            model: Trained BERTSentimentClassifier
            tokenizer: BERT tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = DEVICE
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
        # Ensure model is in eval mode
        self.model.eval()
    
    # ========================================================================
    # Prediction Methods
    # ========================================================================
    
    def predict(self, text: str) -> Dict:
        """Get model prediction for text."""
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_class = logits.argmax(dim=1).item()
        
        return {
            'text': text,
            'predicted_class': pred_class,
            'predicted_label': self.sentiment_labels[pred_class],
            'probabilities': probs,
            'confidence': probs[pred_class]
        }
    
    # ========================================================================
    # Method 1: Word Replacement Counterfactuals
    # ========================================================================
    
    def generate_word_replacement_counterfactuals(
        self,
        text: str,
        max_counterfactuals: int = 5
    ) -> List[Dict]:
        """
        Generate counterfactuals by replacing sentiment words with antonyms.
        
        Args:
            text: Original text
            max_counterfactuals: Maximum number of counterfactuals to generate
            
        Returns:
            List of counterfactual results
        """
        original_pred = self.predict(text)
        words = text.split()
        counterfactuals = []
        
        for i, word in enumerate(words):
            # Remove punctuation for matching
            clean_word = re.sub(r'[।,]', '', word)
            
            if clean_word in SENTIMENT_REPLACEMENTS:
                replacements = SENTIMENT_REPLACEMENTS[clean_word]
                
                for replacement in replacements:
                    # Create new text with replacement
                    new_words = words.copy()
                    # Preserve punctuation
                    if word != clean_word:
                        punct = word.replace(clean_word, '')
                        new_words[i] = replacement + punct
                    else:
                        new_words[i] = replacement
                    
                    new_text = ' '.join(new_words)
                    new_pred = self.predict(new_text)
                    
                    # Check if prediction changed
                    if new_pred['predicted_class'] != original_pred['predicted_class']:
                        counterfactuals.append({
                            'original': original_pred,
                            'counterfactual': new_pred,
                            'change': f"Replaced '{clean_word}' → '{replacement}'",
                            'word_index': i,
                            'method': 'Word Replacement'
                        })
                        
                        if len(counterfactuals) >= max_counterfactuals:
                            return counterfactuals
        
        return counterfactuals
    
    # ========================================================================
    # Method 2: Word Removal Counterfactuals
    # ========================================================================
    
    def generate_word_removal_counterfactuals(
        self,
        text: str,
        max_counterfactuals: int = 5
    ) -> List[Dict]:
        """
        Generate counterfactuals by removing individual words.
        
        Args:
            text: Original text
            max_counterfactuals: Maximum number of counterfactuals to generate
            
        Returns:
            List of counterfactual results
        """
        original_pred = self.predict(text)
        words = text.split()
        counterfactuals = []
        
        for i, word in enumerate(words):
            # Skip very short words and common words
            if len(word) <= 2:
                continue
            
            # Create new text without this word
            new_words = words[:i] + words[i+1:]
            new_text = ' '.join(new_words)
            new_pred = self.predict(new_text)
            
            # Check if prediction changed
            if new_pred['predicted_class'] != original_pred['predicted_class']:
                counterfactuals.append({
                    'original': original_pred,
                    'counterfactual': new_pred,
                    'change': f"Removed '{word}'",
                    'word_index': i,
                    'method': 'Word Removal'
                })
                
                if len(counterfactuals) >= max_counterfactuals:
                    return counterfactuals
        
        return counterfactuals
    
    # ========================================================================
    # Method 3: Negation Addition Counterfactuals
    # ========================================================================
    
    def generate_negation_counterfactuals(
        self,
        text: str
    ) -> List[Dict]:
        """
        Generate counterfactuals by adding/removing negation.
        
        Args:
            text: Original text
            
        Returns:
            List of counterfactual results
        """
        original_pred = self.predict(text)
        words = text.split()
        counterfactuals = []
        
        # Try adding negation before sentiment words
        for i, word in enumerate(words):
            clean_word = re.sub(r'[।,]', '', word)
            
            if clean_word in SENTIMENT_REPLACEMENTS:
                # Add "नहीं" before this word
                new_words = words[:i] + ['नहीं'] + words[i:]
                new_text = ' '.join(new_words)
                new_pred = self.predict(new_text)
                
                if new_pred['predicted_class'] != original_pred['predicted_class']:
                    counterfactuals.append({
                        'original': original_pred,
                        'counterfactual': new_pred,
                        'change': f"Added 'नहीं' before '{clean_word}'",
                        'word_index': i,
                        'method': 'Negation Addition'
                    })
        
        # Try removing existing negations
        for i, word in enumerate(words):
            if word in NEGATION_WORDS:
                new_words = words[:i] + words[i+1:]
                new_text = ' '.join(new_words)
                new_pred = self.predict(new_text)
                
                if new_pred['predicted_class'] != original_pred['predicted_class']:
                    counterfactuals.append({
                        'original': original_pred,
                        'counterfactual': new_pred,
                        'change': f"Removed negation '{word}'",
                        'word_index': i,
                        'method': 'Negation Removal'
                    })
        
        return counterfactuals
    
    # ========================================================================
    # Comprehensive Analysis
    # ========================================================================
    
    def analyze_all_counterfactuals(
        self,
        text: str,
        max_per_method: int = 3
    ) -> Dict:
        """
        Generate counterfactuals using all methods.
        
        Args:
            text: Original text
            max_per_method: Maximum counterfactuals per method
            
        Returns:
            Dictionary with all counterfactual results
        """
        results = {
            'original': self.predict(text),
            'word_replacement': self.generate_word_replacement_counterfactuals(
                text, max_per_method
            ),
            'word_removal': self.generate_word_removal_counterfactuals(
                text, max_per_method
            ),
            'negation': self.generate_negation_counterfactuals(text)[:max_per_method]
        }
        
        return results
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def plot_counterfactuals(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Visualize counterfactual examples.
        
        Args:
            results: Results from analyze_all_counterfactuals
            save_path: Path to save figure
        """
        original = results['original']
        all_counterfactuals = (
            results['word_replacement'] + 
            results['word_removal'] + 
            results['negation']
        )
        
        if not all_counterfactuals:
            print("⚠ No counterfactuals found that flip the prediction")
            return
        
        # Create figure
        n_cf = len(all_counterfactuals)
        fig_height = max(8, 2 + n_cf * 1.2)
        fig, ax = plt.subplots(figsize=(14, fig_height))
        ax.axis('off')
        
        # Title
        title = f"Counterfactual Analysis\n" \
                f"Original Prediction: {original['predicted_label']} " \
                f"({original['confidence']:.1%})\n"
        ax.text(0.5, 0.98, title, ha='center', va='top', fontsize=14,
               fontweight='bold', transform=ax.transAxes)
        
        # Original text
        y_pos = 0.92
        ax.text(0.05, y_pos, "Original:", fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.15, y_pos, original['text'], fontsize=10,
               fontproperties=fm.FontProperties(family=HINDI_FONT),
               transform=ax.transAxes, wrap=True)
        
        # Counterfactuals
        y_pos -= 0.06
        ax.text(0.05, y_pos, "\nCounterfactuals:", fontsize=11, fontweight='bold',
               transform=ax.transAxes)
        y_pos -= 0.04
        
        for i, cf in enumerate(all_counterfactuals, 1):
            cf_pred = cf['counterfactual']
            
            # Draw box
            y_pos -= 0.08
            
            # Counterfactual number and change
            ax.text(0.05, y_pos, f"{i}.", fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            
            # Method and change
            ax.text(0.09, y_pos, f"[{cf['method']}]", fontsize=9,
                   style='italic', color='gray', transform=ax.transAxes)
            
            ax.text(0.09, y_pos - 0.02, cf['change'], fontsize=9,
                   color='blue', transform=ax.transAxes)
            
            # Modified text
            ax.text(0.09, y_pos - 0.04, cf_pred['text'], fontsize=10,
                   fontproperties=fm.FontProperties(family=HINDI_FONT),
                   transform=ax.transAxes, wrap=True)
            
            # Prediction change arrow
            arrow_text = (f"{original['predicted_label']} ({original['confidence']:.0%}) "
                         f"→ {cf_pred['predicted_label']} ({cf_pred['confidence']:.0%})")
            
            # Color based on sentiment direction
            if cf_pred['predicted_class'] < original['predicted_class']:
                color = 'red'
            else:
                color = 'green'
            
            ax.text(0.09, y_pos - 0.06, arrow_text, fontsize=9,
                   color=color, fontweight='bold', transform=ax.transAxes)
            
            y_pos -= 0.02
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_counterfactual_comparison(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create a bar chart comparing confidence changes.
        
        Args:
            results: Results from analyze_all_counterfactuals
            save_path: Path to save figure
        """
        original = results['original']
        all_counterfactuals = (
            results['word_replacement'] + 
            results['word_removal'] + 
            results['negation']
        )
        
        if not all_counterfactuals:
            return
        
        # Prepare data
        labels = ['Original'] + [f"CF {i+1}" for i in range(len(all_counterfactuals))]
        neg_probs = [original['probabilities'][0]]
        neu_probs = [original['probabilities'][1]]
        pos_probs = [original['probabilities'][2]]
        
        for cf in all_counterfactuals:
            probs = cf['counterfactual']['probabilities']
            neg_probs.append(probs[0])
            neu_probs.append(probs[1])
            pos_probs.append(probs[2])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, neg_probs, width, label='Negative', color='#ff6b6b')
        ax.bar(x, neu_probs, width, label='Neutral', color='#ffd93d')
        ax.bar(x + width, pos_probs, width, label='Positive', color='#6bcf7f')
        
        ax.set_xlabel('Example', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Sentiment Probability Changes Across Counterfactuals',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()


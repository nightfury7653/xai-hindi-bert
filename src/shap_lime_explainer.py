"""
SHAP and LIME Explainability Module for BERT Sentiment Classifier.

This module provides model-agnostic explanation methods:
- SHAP (SHapley Additive exPlanations): Global and local feature importance
- LIME (Local Interpretable Model-agnostic Explanations): Local explanations
- Comparative analysis between different explanation methods
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import List, Dict, Tuple, Optional, Callable
import pandas as pd
from transformers import AutoTokenizer
import shap
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

# Setup Hindi font for visualizations
def setup_hindi_font():
    """Configure matplotlib to properly display both Hindi and English."""
    hindi_fonts = [f for f in fm.fontManager.ttflist if 'Noto Sans Devanagari' in f.name]
    
    if hindi_fonts:
        font_path = hindi_fonts[0].fname
        hindi_prop = fm.FontProperties(fname=font_path)
    else:
        hindi_prop = None
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    return hindi_prop

HINDI_FONT = setup_hindi_font()


class SHAPLIMEExplainer:
    """
    Explainer class for SHAP and LIME interpretability methods.
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize the explainer.
        
        Args:
            model: Trained BERT sentiment classifier
            tokenizer: BERT tokenizer
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        self.label_names = ['Negative', 'Neutral', 'Positive']
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=self.label_names,
            bow=False  # Use word-based explanations, not bag-of-words
        )
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict probabilities for a list of texts.
        Required for LIME.
        
        Args:
            texts: List of input texts
            
        Returns:
            Array of shape (len(texts), num_classes) with probabilities
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get prediction (model returns dict with 'logits')
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy()[0])
        
        return np.array(all_probs)
    
    def explain_with_lime(
        self,
        text: str,
        num_features: int = 10,
        num_samples: int = 1000
    ) -> Tuple[object, np.ndarray, int]:
        """
        Explain a prediction using LIME.
        
        Args:
            text: Input text to explain
            num_features: Number of features to show
            num_samples: Number of samples for LIME
            
        Returns:
            Tuple of (lime_explanation, probabilities, predicted_class)
        """
        # Get prediction
        probs = self.predict_proba([text])[0]
        pred_class = np.argmax(probs)
        
        # Generate LIME explanation
        explanation = self.lime_explainer.explain_instance(
            text,
            self.predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=3
        )
        
        return explanation, probs, pred_class
    
    def explain_with_shap(
        self,
        text: str,
        background_texts: Optional[List[str]] = None,
        max_evals: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
        """
        Explain a prediction using SHAP.
        
        Args:
            text: Input text to explain
            background_texts: Background dataset for SHAP (if None, uses single text)
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            Tuple of (shap_values, probabilities, predicted_class, words)
        """
        # Simple word-based SHAP using masking
        words = text.split()
        
        # Get base prediction
        base_probs = self.predict_proba([text])[0]
        pred_class = np.argmax(base_probs)
        
        # Calculate SHAP values by masking each word
        shap_values = np.zeros((len(words), len(self.label_names)))
        
        for i, word in enumerate(words):
            # Create masked text
            masked_words = words.copy()
            masked_words[i] = '[MASK]'
            masked_text = ' '.join(masked_words)
            
            # Get prediction for masked text
            masked_probs = self.predict_proba([masked_text])[0]
            
            # SHAP value = original prob - masked prob
            shap_values[i] = base_probs - masked_probs
        
        return shap_values, base_probs, pred_class, words
    
    def plot_lime_explanation(
        self,
        text: str,
        num_features: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot LIME explanation for a text.
        
        Args:
            text: Input text
            num_features: Number of features to show
            save_path: Path to save figure
        """
        # Get LIME explanation
        explanation, probs, pred_class = self.explain_with_lime(text, num_features)
        
        # Get explanation for predicted class
        exp_list = explanation.as_list(label=pred_class)
        
        # Sort by absolute importance
        exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        words = [item[0] for item in exp_list]
        values = [item[1] for item in exp_list]
        
        # Color based on positive/negative contribution
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.barh(range(len(words)), values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        
        # Apply Hindi font to y-tick labels
        if HINDI_FONT:
            for label in ax.get_yticklabels():
                label.set_fontproperties(HINDI_FONT)
        
        ax.set_xlabel('LIME Feature Importance', fontsize=11)
        ax.set_title(
            f'LIME Explanation\n'
            f'Text: "{text}"\n'
            f'Prediction: {self.label_names[pred_class]} ({probs[pred_class]*100:.1f}%)',
            fontsize=12,
            pad=15
        )
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Positive contribution'),
            Patch(facecolor='red', alpha=0.7, label='Negative contribution')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ LIME explanation saved to: {save_path}")
        
        return fig
    
    def plot_shap_explanation(
        self,
        text: str,
        save_path: Optional[str] = None
    ):
        """
        Plot SHAP explanation for a text.
        
        Args:
            text: Input text
            save_path: Path to save figure
        """
        # Get SHAP explanation
        shap_values, probs, pred_class, words = self.explain_with_shap(text)
        
        # Get SHAP values for predicted class
        class_shap = shap_values[:, pred_class]
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(class_shap))[::-1]
        top_k = min(10, len(words))
        top_indices = sorted_indices[:top_k]
        
        top_words = [words[i] for i in top_indices]
        top_values = [class_shap[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color based on positive/negative contribution
        colors = ['green' if v > 0 else 'red' for v in top_values]
        
        ax.barh(range(len(top_words)), top_values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_words)))
        ax.set_yticklabels(top_words, fontsize=11)
        ax.invert_yaxis()
        
        # Apply Hindi font to y-tick labels
        if HINDI_FONT:
            for label in ax.get_yticklabels():
                label.set_fontproperties(HINDI_FONT)
        
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
        ax.set_title(
            f'SHAP Explanation\n'
            f'Text: "{text}"\n'
            f'Prediction: {self.label_names[pred_class]} ({probs[pred_class]*100:.1f}%)',
            fontsize=12,
            pad=15
        )
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Pushes toward prediction'),
            Patch(facecolor='red', alpha=0.7, label='Pushes away from prediction')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP explanation saved to: {save_path}")
        
        return fig
    
    def compare_explanations(
        self,
        text: str,
        save_path: Optional[str] = None
    ):
        """
        Compare SHAP and LIME explanations side by side.
        
        Args:
            text: Input text
            save_path: Path to save figure
        """
        # Get both explanations
        lime_exp, lime_probs, lime_pred = self.explain_with_lime(text, num_features=10)
        shap_values, shap_probs, shap_pred, words = self.explain_with_shap(text)
        
        # Get LIME features
        lime_dict = dict(lime_exp.as_list(label=lime_pred))
        
        # Get SHAP features for predicted class
        shap_dict = {word: shap_values[i, shap_pred] for i, word in enumerate(words)}
        
        # Get common words
        all_words = set(lime_dict.keys()) | set(shap_dict.keys())
        
        # Create comparison dataframe
        comparison = []
        for word in all_words:
            comparison.append({
                'Word': word,
                'LIME': lime_dict.get(word, 0),
                'SHAP': shap_dict.get(word, 0)
            })
        
        df = pd.DataFrame(comparison)
        df['Avg_Abs'] = (abs(df['LIME']) + abs(df['SHAP'])) / 2
        df = df.sort_values('Avg_Abs', ascending=False).head(10)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # LIME plot
        lime_colors = ['green' if v > 0 else 'red' for v in df['LIME']]
        ax1.barh(range(len(df)), df['LIME'], color=lime_colors, alpha=0.7)
        ax1.set_yticks(range(len(df)))
        ax1.set_yticklabels(df['Word'], fontsize=11)
        ax1.invert_yaxis()
        ax1.set_xlabel('LIME Importance', fontsize=11)
        ax1.set_title(f'LIME Explanation\nPrediction: {self.label_names[lime_pred]}', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax1.grid(axis='x', alpha=0.3)
        
        # SHAP plot
        shap_colors = ['green' if v > 0 else 'red' for v in df['SHAP']]
        ax2.barh(range(len(df)), df['SHAP'], color=shap_colors, alpha=0.7)
        ax2.set_yticks(range(len(df)))
        ax2.set_yticklabels(df['Word'], fontsize=11)
        ax2.invert_yaxis()
        ax2.set_xlabel('SHAP Value', fontsize=11)
        ax2.set_title(f'SHAP Explanation\nPrediction: {self.label_names[shap_pred]}', fontsize=12)
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # Apply Hindi font to both plots
        if HINDI_FONT:
            for ax in [ax1, ax2]:
                for label in ax.get_yticklabels():
                    label.set_fontproperties(HINDI_FONT)
        
        fig.suptitle(f'SHAP vs LIME Comparison\nText: "{text}"', fontsize=14, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison saved to: {save_path}")
        
        return fig
    
    def generate_explanation_report(
        self,
        text: str,
        output_dir: str = 'outputs/phase3'
    ):
        """
        Generate a complete explanation report for a text.
        
        Args:
            text: Input text to explain
            output_dir: Directory to save outputs
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all explanations
        print(f"\nGenerating explanations for: \"{text}\"")
        print("-" * 70)
        
        # LIME
        lime_exp, lime_probs, lime_pred = self.explain_with_lime(text)
        print(f"LIME Prediction: {self.label_names[lime_pred]} ({lime_probs[lime_pred]*100:.1f}%)")
        
        # SHAP
        shap_values, shap_probs, shap_pred, words = self.explain_with_shap(text)
        print(f"SHAP Prediction: {self.label_names[shap_pred]} ({shap_probs[shap_pred]*100:.1f}%)")
        
        # Generate visualizations
        base_name = text[:30].replace(' ', '_').replace(',', '').replace('।', '')
        
        self.plot_lime_explanation(
            text,
            save_path=f"{output_dir}/{base_name}_lime.png"
        )
        
        self.plot_shap_explanation(
            text,
            save_path=f"{output_dir}/{base_name}_shap.png"
        )
        
        self.compare_explanations(
            text,
            save_path=f"{output_dir}/{base_name}_comparison.png"
        )
        
        print(f"✓ All explanations saved to: {output_dir}/")
        
        return {
            'lime': (lime_exp, lime_probs, lime_pred),
            'shap': (shap_values, shap_probs, shap_pred, words)
        }


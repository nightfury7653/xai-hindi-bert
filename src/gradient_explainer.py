"""
Phase 4: Gradient-based Interpretability
=========================================
Implements gradient-based explanation methods:
1. Saliency Maps - Shows which tokens have largest gradient magnitude
2. Integrated Gradients - Attributes importance by integrating gradients
3. Gradient × Input - Combines gradient information with input embeddings
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
    # Try to find and use Noto Sans Devanagari for Hindi
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
        print("⚠ Warning: No Hindi font found. Install 'fonts-noto' package.")
        print("  Run: sudo apt-get install fonts-noto fonts-noto-core")
    
    # Set DejaVu Sans as default for English
    rcParams['font.family'] = 'DejaVu Sans'
    rcParams['font.size'] = 10
    rcParams['axes.unicode_minus'] = False

# Initialize fonts
HINDI_FONT = 'DejaVu Sans'
setup_hindi_font()


# ============================================================================
# Helper Functions
# ============================================================================

def merge_subword_tokens(tokens: List[str], scores: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """
    Merge subword tokens (starting with ##) back into complete words.
    Aggregate scores by taking the mean for merged tokens.
    
    Args:
        tokens: List of tokens (may include ##subwords)
        scores: Attribution scores for each token
        
    Returns:
        merged_tokens: List of complete words
        merged_scores: Aggregated scores for merged words
    """
    merged_tokens = []
    merged_scores = []
    
    current_word = ""
    current_scores = []
    
    for token, score in zip(tokens, scores):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            if current_word:
                merged_tokens.append(current_word)
                merged_scores.append(np.mean(current_scores))
                current_word = ""
                current_scores = []
            continue
            
        if token.startswith('##'):
            # Continuation of previous word
            current_word += token[2:]
            current_scores.append(score)
        else:
            # New word
            if current_word:
                merged_tokens.append(current_word)
                merged_scores.append(np.mean(current_scores))
            current_word = token
            current_scores = [score]
    
    # Add last word
    if current_word:
        merged_tokens.append(current_word)
        merged_scores.append(np.mean(current_scores))
    
    return merged_tokens, np.array(merged_scores)


# ============================================================================
# Gradient-based Explainer Class
# ============================================================================

class GradientExplainer:
    """
    Implements gradient-based interpretability methods for BERT models.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the gradient explainer.
        
        Args:
            model: Trained BERTSentimentClassifier
            tokenizer: BERT tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = DEVICE
        
        # Ensure model is in eval mode
        self.model.eval()
        
    # ========================================================================
    # Method 1: Saliency Maps
    # ========================================================================
    
    def compute_saliency(
        self,
        text: str,
        target_class: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Compute saliency map - gradient magnitude of loss w.r.t. input embeddings.
        
        Args:
            text: Input text
            target_class: Target class for gradient computation (None = predicted class)
            
        Returns:
            Dictionary containing tokens and their saliency scores
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get embeddings with gradient tracking
        embeddings = self.model.bert.embeddings(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Retain gradients for non-leaf tensor
        
        # Forward pass through rest of model
        encoder_outputs = self.model.bert.encoder(
            embeddings,
            attention_mask=self._get_extended_attention_mask(attention_mask)
        )
        
        pooled_output = self.model.bert.pooler(encoder_outputs[0])
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Compute gradient
        self.model.zero_grad()
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        # Get saliency scores (L2 norm of gradients)
        saliency = embeddings.grad.abs().sum(dim=-1).squeeze().cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out padding
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        saliency = saliency[:valid_length]
        
        # Normalize scores
        if saliency.max() > 0:
            saliency = saliency / saliency.max()
        
        # Get prediction
        probs = F.softmax(logits, dim=1)[0].cpu().detach().numpy()
        
        return {
            'tokens': tokens,
            'scores': saliency,
            'predicted_class': logits.argmax(dim=1).item(),
            'probabilities': probs,
            'target_class': target_class,
            'method': 'Saliency Map'
        }
    
    # ========================================================================
    # Method 2: Integrated Gradients
    # ========================================================================
    
    def compute_integrated_gradients(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 50,
        baseline_type: str = 'zero'
    ) -> Dict[str, any]:
        """
        Compute Integrated Gradients - integrates gradients along path from baseline.
        
        Args:
            text: Input text
            target_class: Target class (None = predicted class)
            n_steps: Number of integration steps
            baseline_type: 'zero' or 'pad' baseline
            
        Returns:
            Dictionary containing tokens and their IG attributions
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get input embeddings
        input_embeddings = self.model.bert.embeddings(input_ids)
        
        # Create baseline
        if baseline_type == 'zero':
            baseline = torch.zeros_like(input_embeddings)
        else:  # pad baseline
            pad_ids = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            baseline = self.model.bert.embeddings(pad_ids)
        
        # Get target class
        with torch.no_grad():
            encoder_outputs = self.model.bert.encoder(
                input_embeddings,
                attention_mask=self._get_extended_attention_mask(attention_mask)
            )
            pooled_output = self.model.bert.pooler(encoder_outputs[0])
            pooled_output = self.model.dropout(pooled_output)
            logits = self.model.classifier(pooled_output)
            
            if target_class is None:
                target_class = logits.argmax(dim=1).item()
            
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Compute integrated gradients
        integrated_grads = torch.zeros_like(input_embeddings)
        
        for step in range(n_steps):
            # Interpolate between baseline and input
            alpha = (step + 1) / n_steps
            interpolated = baseline + alpha * (input_embeddings - baseline)
            interpolated.requires_grad_(True)
            interpolated.retain_grad()  # Retain gradients for non-leaf tensor
            
            # Forward pass
            encoder_outputs = self.model.bert.encoder(
                interpolated,
                attention_mask=self._get_extended_attention_mask(attention_mask)
            )
            pooled_output = self.model.bert.pooler(encoder_outputs[0])
            pooled_output = self.model.dropout(pooled_output)
            logits = self.model.classifier(pooled_output)
            
            # Compute gradient
            self.model.zero_grad()
            target_logit = logits[0, target_class]
            target_logit.backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_grads += interpolated.grad.clone()
        
        # Average gradients and multiply by input difference
        integrated_grads = integrated_grads / n_steps
        attributions = (input_embeddings - baseline) * integrated_grads
        
        # Sum over embedding dimension
        attributions = attributions.sum(dim=-1).squeeze().abs().cpu().detach().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out padding
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        attributions = attributions[:valid_length]
        
        # Normalize scores
        if attributions.max() > 0:
            attributions = attributions / attributions.max()
        
        return {
            'tokens': tokens,
            'scores': attributions,
            'predicted_class': target_class,
            'probabilities': probs,
            'target_class': target_class,
            'method': f'Integrated Gradients ({n_steps} steps)'
        }
    
    # ========================================================================
    # Method 3: Gradient × Input
    # ========================================================================
    
    def compute_gradient_x_input(
        self,
        text: str,
        target_class: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Compute Gradient × Input - element-wise product of gradients and inputs.
        
        Args:
            text: Input text
            target_class: Target class (None = predicted class)
            
        Returns:
            Dictionary containing tokens and their Grad×Input scores
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get embeddings with gradient tracking
        embeddings = self.model.bert.embeddings(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Retain gradients for non-leaf tensor
        
        # Forward pass
        encoder_outputs = self.model.bert.encoder(
            embeddings,
            attention_mask=self._get_extended_attention_mask(attention_mask)
        )
        pooled_output = self.model.bert.pooler(encoder_outputs[0])
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Compute gradient
        self.model.zero_grad()
        target_logit = logits[0, target_class]
        target_logit.backward()
        
        # Gradient × Input
        grad_x_input = (embeddings.grad * embeddings).sum(dim=-1).squeeze().abs()
        grad_x_input = grad_x_input.cpu().detach().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Filter out padding
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        grad_x_input = grad_x_input[:valid_length]
        
        # Normalize scores
        if grad_x_input.max() > 0:
            grad_x_input = grad_x_input / grad_x_input.max()
        
        # Get prediction
        probs = F.softmax(logits, dim=1)[0].cpu().detach().numpy()
        
        return {
            'tokens': tokens,
            'scores': grad_x_input,
            'predicted_class': logits.argmax(dim=1).item(),
            'probabilities': probs,
            'target_class': target_class,
            'method': 'Gradient × Input'
        }
    
    # ========================================================================
    # Visualization Methods
    # ========================================================================
    
    def plot_gradient_attribution(
        self,
        result: Dict,
        text: str,
        save_path: Optional[str] = None,
        top_k: int = 15,
        merge_subwords: bool = True
    ):
        """
        Plot gradient-based attribution scores.
        
        Args:
            result: Result dictionary from gradient method
            text: Original text
            save_path: Path to save figure
            top_k: Number of top tokens to show
            merge_subwords: Whether to merge subword tokens
        """
        tokens = result['tokens']
        scores = result['scores']
        method = result['method']
        pred_class = result['predicted_class']
        probs = result['probabilities']
        
        # Merge subwords if requested
        if merge_subwords:
            tokens, scores = merge_subword_tokens(tokens, scores)
        
        # Get top-k tokens
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_scores = scores[top_indices]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Sentiment labels
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
        # Plot 1: Bar chart of top attributions
        colors = plt.cm.RdYlGn(top_scores)
        bars = ax1.barh(range(len(top_tokens)), top_scores, color=colors)
        ax1.set_yticks(range(len(top_tokens)))
        ax1.set_yticklabels(top_tokens, fontproperties=fm.FontProperties(family=HINDI_FONT, size=10))
        ax1.set_xlabel('Attribution Score', fontsize=11)
        ax1.set_title(f'{method} - Top {top_k} Important Tokens', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Text with highlighted attributions
        ax2.axis('off')
        
        # Title with prediction
        title = f'Prediction: {sentiment_labels[pred_class]} ({probs[pred_class]:.2%})\n\n'
        ax2.text(0.5, 0.95, title, ha='center', va='top', fontsize=12, 
                fontweight='bold', transform=ax2.transAxes)
        
        # Create color-coded text
        y_pos = 0.85
        x_pos = 0.05
        line_tokens = []
        line_scores = []
        
        for token, score in zip(tokens, scores):
            line_tokens.append(token)
            line_scores.append(score)
            
            # Start new line if too long
            if len(' '.join(line_tokens)) > 50:
                self._render_colored_text(ax2, line_tokens, line_scores, x_pos, y_pos)
                y_pos -= 0.08
                line_tokens = []
                line_scores = []
        
        # Render remaining tokens
        if line_tokens:
            self._render_colored_text(ax2, line_tokens, line_scores, x_pos, y_pos)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, orientation='horizontal', 
                           pad=0.05, aspect=30, shrink=0.8)
        cbar.set_label('Attribution Score', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    def plot_method_comparison(
        self,
        results: Dict[str, Dict],
        text: str,
        save_path: Optional[str] = None,
        top_k: int = 10,
        merge_subwords: bool = True
    ):
        """
        Compare multiple gradient methods side by side.
        
        Args:
            results: Dictionary of method_name -> result
            text: Original text
            save_path: Path to save figure
            top_k: Number of top tokens to show
            merge_subwords: Whether to merge subword tokens
        """
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 8))
        
        if n_methods == 1:
            axes = [axes]
        
        for ax, (method_name, result) in zip(axes, results.items()):
            tokens = result['tokens']
            scores = result['scores']
            
            # Merge subwords if requested
            if merge_subwords:
                tokens, scores = merge_subword_tokens(tokens, scores)
            
            # Get top-k tokens
            top_indices = np.argsort(scores)[-top_k:][::-1]
            top_tokens = [tokens[i] for i in top_indices]
            top_scores = scores[top_indices]
            
            # Plot
            colors = plt.cm.RdYlGn(top_scores)
            ax.barh(range(len(top_tokens)), top_scores, color=colors)
            ax.set_yticks(range(len(top_tokens)))
            ax.set_yticklabels(top_tokens, fontproperties=fm.FontProperties(family=HINDI_FONT, size=10))
            ax.set_xlabel('Attribution Score', fontsize=11)
            ax.set_title(method_name, fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
        
        # Add overall title
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        pred_class = list(results.values())[0]['predicted_class']
        probs = list(results.values())[0]['probabilities']
        
        fig.suptitle(f'Gradient Methods Comparison\nPrediction: {sentiment_labels[pred_class]} ({probs[pred_class]:.2%})',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.close()
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_extended_attention_mask(self, attention_mask):
        """Convert attention mask to format expected by BERT encoder."""
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
    
    def _render_colored_text(self, ax, tokens, scores, x_pos, y_pos):
        """Render tokens with color-coded importance."""
        cmap = plt.cm.RdYlGn
        
        for token, score in zip(tokens, scores):
            color = cmap(score)
            ax.text(x_pos, y_pos, token + ' ', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none'),
                   fontproperties=fm.FontProperties(family=HINDI_FONT),
                   transform=ax.transAxes, va='top')
            # Approximate x position increment
            x_pos += len(token) * 0.015 + 0.02
            
            if x_pos > 0.95:
                break


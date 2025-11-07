"""
Attention Analysis Module for BERT Sentiment Classifier.

This module provides tools to:
- Extract attention weights from BERT layers
- Visualize attention patterns
- Analyze which tokens the model focuses on
- Create interactive attention visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from transformers import AutoTokenizer

# Configure matplotlib to use a font that supports both Hindi and English
def setup_hindi_font():
    """Configure matplotlib to properly display both Hindi Devanagari and English text."""
    import matplotlib.font_manager as fm
    
    # Find the Noto Sans Devanagari font for Hindi text
    hindi_fonts = [f for f in fm.fontManager.ttflist if 'Noto Sans Devanagari' in f.name]
    
    if hindi_fonts:
        font_path = hindi_fonts[0].fname
        hindi_prop = fm.FontProperties(fname=font_path)
    else:
        hindi_prop = None
    
    # Use default font (DejaVu Sans) which supports Latin characters
    # We'll apply Hindi font selectively only to Hindi text elements
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    return hindi_prop

# Apply font configuration and get Hindi font properties
HINDI_FONT = setup_hindi_font()


class AttentionAnalyzer:
    """
    Analyzer for BERT attention patterns in sentiment classification.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize the attention analyzer.
        
        Args:
            model: Trained BERTSentimentClassifier
            tokenizer: Tokenizer used for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Ensure model outputs attention weights
        if not model.output_attentions:
            print("Warning: Model was not configured to output attentions.")
            print("Creating new model instance with attention outputs enabled...")
            from config import MODEL_NAME, NUM_LABELS
            from src.model import BERTSentimentClassifier
            self.model = BERTSentimentClassifier(
                MODEL_NAME, 
                NUM_LABELS, 
                output_attentions=True
            )
            # Load the same weights
            self.model.load_state_dict(model.state_dict())
            self.model.eval()
            self.model.to(self.device)
    
    def get_attention_weights(
        self, 
        text: str, 
        max_length: int = 128
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Extract attention weights for a given text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (attentions, tokens, predicted_probs)
            - attentions: Shape [num_layers, num_heads, seq_length, seq_length]
            - tokens: List of token strings
            - predicted_probs: Probability distribution over classes
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            max_length=max_length,
            truncation=True, 
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Extract attention weights
        attentions = outputs['attentions']  # Tuple of tensors
        attentions = torch.stack(attentions)  # [num_layers, batch, num_heads, seq_len, seq_len]
        attentions = attentions.squeeze(1)   # Remove batch dimension
        
        # Get predictions
        logits = outputs['logits']
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        
        # Find actual sequence length (before padding)
        actual_length = attention_mask.sum().item()
        
        return attentions, tokens, probs, actual_length
    
    def plot_attention_heatmap(
        self,
        text: str,
        layer: int = -1,
        head: int = 0,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot attention heatmap for a specific layer and head.
        
        Args:
            text: Input text
            layer: Layer index (-1 for last layer)
            head: Attention head index
            figsize: Figure size
            save_path: Path to save the figure (optional)
        """
        # Get attention weights
        attentions, tokens, probs, actual_length = self.get_attention_weights(text)
        
        # Select layer and head
        if layer == -1:
            layer = attentions.size(0) - 1
        
        attention = attentions[layer, head, :actual_length, :actual_length].cpu().numpy()
        tokens = tokens[:actual_length]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.5,
            linecolor='gray'
        )
        
        # Get prediction
        label_names = ['Negative', 'Neutral', 'Positive']
        pred_label = label_names[torch.argmax(probs).item()]
        pred_conf = probs.max().item() * 100
        
        # Set title (English labels, Hindi text will render with fallback)
        title_text = (
            f'Attention Pattern - Layer {layer}, Head {head}\n'
            f'Text: "{text}"\n'
            f'Prediction: {pred_label} ({pred_conf:.1f}%)'
        )
        ax.set_title(title_text, fontsize=12, pad=20)
        
        # English axis labels (will use default font)
        ax.set_xlabel('Key Tokens', fontsize=10)
        ax.set_ylabel('Query Tokens', fontsize=10)
        
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        # Apply Hindi font ONLY to tick labels (which are Hindi words/tokens)
        if HINDI_FONT:
            for label in ax.get_xticklabels():
                label.set_fontproperties(HINDI_FONT)
            for label in ax.get_yticklabels():
                label.set_fontproperties(HINDI_FONT)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention heatmap saved to: {save_path}")
        
        return fig
    
    def plot_attention_summary(
        self,
        text: str,
        layers: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ):
        """
        Plot attention summary across multiple layers.
        
        Args:
            text: Input text
            layers: List of layer indices to visualize (None for all)
            figsize: Figure size
            save_path: Path to save the figure
        """
        # Get attention weights
        attentions, tokens, probs, actual_length = self.get_attention_weights(text)
        
        num_layers = attentions.size(0)
        num_heads = attentions.size(1)
        
        if layers is None:
            # Show first, middle, and last layers
            layers = [0, num_layers // 2, num_layers - 1]
        
        tokens = tokens[:actual_length]
        
        # Create subplots
        fig, axes = plt.subplots(1, len(layers), figsize=figsize)
        if len(layers) == 1:
            axes = [axes]
        
        for idx, layer in enumerate(layers):
            # Average attention across all heads
            avg_attention = attentions[layer, :, :actual_length, :actual_length].mean(0).cpu().numpy()
            
            # Plot
            sns.heatmap(
                avg_attention,
                xticklabels=tokens,
                yticklabels=tokens if idx == 0 else [],
                cmap='YlOrRd',
                ax=axes[idx],
                cbar=True,
                square=True,
                linewidths=0.3,
                linecolor='lightgray'
            )
            
            # English labels for axes
            axes[idx].set_title(f'Layer {layer}\n(Avg across {num_heads} heads)', fontsize=10)
            axes[idx].set_xlabel('Key Tokens', fontsize=9)
            if idx == 0:
                axes[idx].set_ylabel('Query Tokens', fontsize=9)
            
            plt.setp(axes[idx].get_xticklabels(), rotation=90, fontsize=7)
            plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=7)
            
            # Apply Hindi font ONLY to tick labels (Hindi tokens)
            if HINDI_FONT:
                for label in axes[idx].get_xticklabels():
                    label.set_fontproperties(HINDI_FONT)
                for label in axes[idx].get_yticklabels():
                    label.set_fontproperties(HINDI_FONT)
        
        # Get prediction
        label_names = ['Negative', 'Neutral', 'Positive']
        pred_label = label_names[torch.argmax(probs).item()]
        pred_conf = probs.max().item() * 100
        
        # Main title (English labels, Hindi text handled by default rendering)
        fig.suptitle(
            f'Attention Across Layers\n'
            f'Text: "{text}"\n'
            f'Prediction: {pred_label} ({pred_conf:.1f}%)',
            fontsize=14,
            y=0.98
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention summary saved to: {save_path}")
        
        return fig
    
    def get_token_importance(
        self,
        text: str,
        layer: int = -1,
        merge_subwords: bool = False
    ) -> pd.DataFrame:
        """
        Calculate token importance based on attention weights.
        
        Args:
            text: Input text
            layer: Layer index (-1 for last layer)
            merge_subwords: Whether to merge subword tokens into complete words
            
        Returns:
            DataFrame with tokens and their importance scores
        """
        # Get attention weights
        attentions, tokens, probs, actual_length = self.get_attention_weights(text)
        
        if layer == -1:
            layer = attentions.size(0) - 1
        
        # Get attention for [CLS] token (first token) as it's used for classification
        cls_attention = attentions[layer, :, 0, :actual_length]  # [num_heads, seq_len]
        
        # Average across heads
        avg_cls_attention = cls_attention.mean(0).cpu().numpy()
        
        tokens = tokens[:actual_length]
        
        # Create dataframe
        df = pd.DataFrame({
            'Token': tokens,
            'Attention_Weight': avg_cls_attention,
            'Normalized_Importance': (avg_cls_attention / avg_cls_attention.sum()) * 100
        })
        
        if merge_subwords:
            # Merge subword tokens into complete words
            merged_tokens = []
            merged_weights = []
            
            current_word = ""
            current_weight = 0.0
            
            for idx, row in df.iterrows():
                token = row['Token']
                weight = row['Attention_Weight']
                
                # Skip special tokens
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    if current_word:
                        merged_tokens.append(current_word)
                        merged_weights.append(current_weight)
                        current_word = ""
                        current_weight = 0.0
                    merged_tokens.append(token)
                    merged_weights.append(weight)
                    continue
                
                # Check if it's a subword token (starts with ##)
                if token.startswith('##'):
                    current_word += token[2:]  # Remove ## prefix
                    current_weight += weight
                else:
                    # New word starts
                    if current_word:
                        merged_tokens.append(current_word)
                        merged_weights.append(current_weight)
                    current_word = token
                    current_weight = weight
            
            # Add last word
            if current_word:
                merged_tokens.append(current_word)
                merged_weights.append(current_weight)
            
            # Create new DataFrame with merged tokens
            merged_weights_array = np.array(merged_weights)
            df = pd.DataFrame({
                'Token': merged_tokens,
                'Attention_Weight': merged_weights_array,
                'Normalized_Importance': (merged_weights_array / merged_weights_array.sum()) * 100
            })
        
        # Sort by importance
        df = df.sort_values('Attention_Weight', ascending=False)
        
        return df
    
    def plot_token_importance(
        self,
        text: str,
        layer: int = -1,
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot token importance scores.
        
        Args:
            text: Input text
            layer: Layer index (-1 for last layer)
            top_k: Number of top tokens to show
            figsize: Figure size
            save_path: Path to save the figure
        """
        # Get token importance with merged subwords for better readability
        df = self.get_token_importance(text, layer, merge_subwords=True)
        
        # Filter special tokens and get top k
        df_filtered = df[~df['Token'].isin(['[CLS]', '[SEP]', '[PAD]'])]
        df_top = df_filtered.head(top_k)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot horizontal bar chart
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(df_top)))
        ax.barh(range(len(df_top)), df_top['Normalized_Importance'], color=colors)
        ax.set_yticks(range(len(df_top)))
        ax.set_yticklabels(df_top['Token'], fontsize=13)
        ax.invert_yaxis()
        
        # English labels for title and axes
        ax.set_xlabel('Relative Importance (%)', fontsize=11)
        ax.set_title(f'Top {top_k} Most Important Words\nText: "{text}"', fontsize=12, pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        # Apply Hindi font ONLY to y-axis tick labels (the Hindi words)
        if HINDI_FONT:
            for label in ax.get_yticklabels():
                label.set_fontproperties(HINDI_FONT)
                label.set_fontsize(13)
        
        # Add value labels
        for i, (idx, row) in enumerate(df_top.iterrows()):
            ax.text(
                row['Normalized_Importance'] + 0.5,
                i,
                f"{row['Normalized_Importance']:.1f}%",
                va='center',
                fontsize=9
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Token importance plot saved to: {save_path}")
        
        return fig
    
    def analyze_attention_flow(
        self,
        text: str,
        source_token_idx: int = 0,
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Analyze how attention flows from a source token across layers.
        
        Args:
            text: Input text
            source_token_idx: Index of source token (0 for [CLS])
            figsize: Figure size
            save_path: Path to save the figure
        """
        # Get attention weights
        attentions, tokens, probs, actual_length = self.get_attention_weights(text)
        
        num_layers = attentions.size(0)
        tokens = tokens[:actual_length]
        
        # Extract attention flow for source token
        attention_flow = []
        for layer in range(num_layers):
            # Average across heads
            layer_attention = attentions[layer, :, source_token_idx, :actual_length].mean(0).cpu().numpy()
            attention_flow.append(layer_attention)
        
        attention_flow = np.array(attention_flow)  # [num_layers, seq_len]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            attention_flow,
            xticklabels=tokens,
            yticklabels=[f'L{i}' for i in range(num_layers)],
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        # English labels for title and axes
        ax.set_title(
            f'Attention Flow from "{tokens[source_token_idx]}" Token\n'
            f'Text: "{text}"',
            fontsize=12,
            pad=15
        )
        ax.set_xlabel('Target Tokens', fontsize=10)
        ax.set_ylabel('Layers', fontsize=10)
        
        plt.xticks(rotation=90, fontsize=8)
        
        # Apply Hindi font ONLY to x-axis tick labels (Hindi tokens)
        if HINDI_FONT:
            for label in ax.get_xticklabels():
                label.set_fontproperties(HINDI_FONT)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention flow plot saved to: {save_path}")
        
        return fig


def analyze_multiple_samples(
    model,
    tokenizer,
    texts: List[str],
    save_dir: str = 'outputs/phase2'
):
    """
    Analyze attention patterns for multiple text samples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of texts to analyze
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    analyzer = AttentionAnalyzer(model, tokenizer)
    
    print(f"\n{'='*80}")
    print("ANALYZING ATTENTION PATTERNS")
    print(f"{'='*80}\n")
    
    for i, text in enumerate(texts, 1):
        print(f"\n{i}. Analyzing: \"{text}\"")
        print("-" * 70)
        
        # Get predictions
        _, _, probs, _ = analyzer.get_attention_weights(text)
        label_names = ['Negative', 'Neutral', 'Positive']
        pred_label = label_names[torch.argmax(probs).item()]
        pred_conf = probs.max().item() * 100
        
        print(f"   Prediction: {pred_label} ({pred_conf:.1f}%)")
        
        # Token importance (with merged subwords for better readability)
        token_df = analyzer.get_token_importance(text, merge_subwords=True)
        print("\n   Top 5 Important Words:")
        for idx, row in token_df.head(5).iterrows():
            if row['Token'] not in ['[CLS]', '[SEP]', '[PAD]']:
                print(f"      • {row['Token']}: {row['Normalized_Importance']:.1f}%")
        
        # Save visualizations
        base_name = f"sample_{i}"
        
        # Attention heatmap
        analyzer.plot_attention_heatmap(
            text,
            save_path=f"{save_dir}/{base_name}_attention_heatmap.png"
        )
        plt.close()
        
        # Token importance
        analyzer.plot_token_importance(
            text,
            save_path=f"{save_dir}/{base_name}_token_importance.png"
        )
        plt.close()
        
        # Attention summary
        analyzer.plot_attention_summary(
            text,
            save_path=f"{save_dir}/{base_name}_attention_summary.png"
        )
        plt.close()
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! Visualizations saved to: {save_dir}/")
    print(f"{'='*80}\n")


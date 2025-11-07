"""
Phase 6: Interactive Visualization Interface
============================================
Gradio-based web interface for interactive sentiment analysis and explainability.
"""

import torch
import gradio as gr
from transformers import AutoTokenizer
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import BERTSentimentClassifier
from src.attention_analysis import AttentionAnalyzer
from src.shap_lime_explainer import SHAP_LIME_Explainer
from src.gradient_explainer import GradientExplainer
from src.counterfactual_analyzer import CounterfactualAnalyzer
from config import MODEL_NAME, NUM_LABELS, DEVICE


# ============================================================================
# Model Loading
# ============================================================================

print("🔄 Loading model and tokenizer...")

model_path = Path('models/model.pt')
if not model_path.exists():
    raise FileNotFoundError("Model not found! Please train the model first (run_phase1.py)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"✓ Model loaded successfully on {DEVICE}")

# Initialize analyzers
attention_analyzer = AttentionAnalyzer(model, tokenizer)
shap_lime_explainer = SHAP_LIME_Explainer(model, tokenizer)
gradient_explainer = GradientExplainer(model, tokenizer)
counterfactual_analyzer = CounterfactualAnalyzer(model, tokenizer)

print("✓ All analyzers initialized")


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_sentiment(text: str):
    """
    Perform complete sentiment analysis with all explainability methods.
    
    Args:
        text: Input Hindi text
        
    Returns:
        Tuple of (prediction_text, attention_plot, shap_plot, lime_plot,
                 gradient_plot, counterfactual_plot)
    """
    if not text.strip():
        return "⚠ Please enter some text", None, None, None, None, None
    
    try:
        # ====================================================================
        # 1. Basic Prediction
        # ====================================================================
        
        pred = counterfactual_analyzer.predict(text)
        sentiment_labels = ['😞 Negative', '😐 Neutral', '😊 Positive']
        
        prediction_html = f"""
        <div style='padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
            <h2 style='color: white; text-align: center;'>Sentiment Analysis Result</h2>
            <div style='background: white; padding: 20px; border-radius: 8px; margin-top: 15px;'>
                <h3 style='text-align: center; color: #333;'>
                    Prediction: <span style='color: #667eea;'>{sentiment_labels[pred['predicted_class']]}</span>
                </h3>
                <p style='text-align: center; font-size: 18px;'>
                    Confidence: <strong>{pred['confidence']:.2%}</strong>
                </p>
                <div style='margin-top: 15px;'>
                    <p><strong>Probabilities:</strong></p>
                    <div style='background: #ffe0e0; padding: 8px; border-radius: 5px; margin: 5px 0;'>
                        😞 Negative: {pred['probabilities'][0]:.2%}
                    </div>
                    <div style='background: #fff4e0; padding: 8px; border-radius: 5px; margin: 5px 0;'>
                        😐 Neutral: {pred['probabilities'][1]:.2%}
                    </div>
                    <div style='background: #e0ffe0; padding: 8px; border-radius: 5px; margin: 5px 0;'>
                        😊 Positive: {pred['probabilities'][2]:.2%}
                    </div>
                </div>
            </div>
        </div>
        """
        
        # ====================================================================
        # 2. Attention Analysis
        # ====================================================================
        
        print("🔍 Computing attention analysis...")
        attention_result = attention_analyzer.get_token_importance(text, merge_subwords=True)
        
        # Create attention plot
        fig, ax = plt.subplots(figsize=(10, 6))
        tokens = attention_result['merged_words'][:15]
        scores = attention_result['merged_scores'][:15]
        
        colors = plt.cm.RdYlGn(scores / scores.max())
        ax.barh(range(len(tokens)), scores, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Attention Score')
        ax.set_title('Top Token Importance (Attention)', fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        attention_img = Image.open(buf)
        plt.close()
        
        # ====================================================================
        # 3. SHAP Explanation
        # ====================================================================
        
        print("🔍 Computing SHAP explanation...")
        shap_result = shap_lime_explainer.explain_with_shap(text, merge_subwords=True)
        
        # Create SHAP plot
        fig, ax = plt.subplots(figsize=(10, 6))
        tokens = shap_result['merged_words'][:15]
        scores = shap_result['merged_scores'][:15]
        
        colors = ['red' if s < 0 else 'green' for s in scores]
        ax.barh(range(len(tokens)), np.abs(scores), color=colors, alpha=0.7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('|SHAP Value|')
        ax.set_title('Token Importance (SHAP)', fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        shap_img = Image.open(buf)
        plt.close()
        
        # ====================================================================
        # 4. LIME Explanation
        # ====================================================================
        
        print("🔍 Computing LIME explanation...")
        lime_result = shap_lime_explainer.explain_with_lime(text, merge_subwords=True)
        
        # Create LIME plot
        fig, ax = plt.subplots(figsize=(10, 6))
        tokens = lime_result['merged_words'][:15]
        scores = lime_result['merged_scores'][:15]
        
        colors = ['red' if s < 0 else 'green' for s in scores]
        ax.barh(range(len(tokens)), np.abs(scores), color=colors, alpha=0.7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('|LIME Weight|')
        ax.set_title('Token Importance (LIME)', fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        lime_img = Image.open(buf)
        plt.close()
        
        # ====================================================================
        # 5. Gradient Analysis
        # ====================================================================
        
        print("🔍 Computing gradient analysis...")
        saliency_result = gradient_explainer.compute_saliency(text)
        
        # Create gradient plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        from src.gradient_explainer import merge_subword_tokens
        tokens, scores = merge_subword_tokens(
            saliency_result['tokens'],
            saliency_result['scores']
        )
        
        tokens = tokens[:15]
        scores = scores[:15]
        
        colors = plt.cm.RdYlGn(scores)
        ax.barh(range(len(tokens)), scores, color=colors)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Saliency Score')
        ax.set_title('Token Importance (Gradient Saliency)', fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        gradient_img = Image.open(buf)
        plt.close()
        
        # ====================================================================
        # 6. Counterfactual Analysis
        # ====================================================================
        
        print("🔍 Computing counterfactual analysis...")
        cf_results = counterfactual_analyzer.analyze_all_counterfactuals(text, max_per_method=2)
        
        all_cf = (cf_results['word_replacement'] + 
                  cf_results['word_removal'] + 
                  cf_results['negation'])
        
        if all_cf:
            # Create counterfactual plot
            fig, ax = plt.subplots(figsize=(12, max(4, len(all_cf) * 1.5)))
            ax.axis('off')
            
            y_pos = 0.95
            ax.text(0.5, y_pos, 'Counterfactual Examples', ha='center',
                   fontsize=14, fontweight='bold', transform=ax.transAxes)
            
            y_pos -= 0.1
            for i, cf in enumerate(all_cf[:5], 1):
                cf_pred = cf['counterfactual']
                
                # Box background
                rect = plt.Rectangle((0.05, y_pos - 0.15), 0.9, 0.13,
                                    facecolor='lightgray', alpha=0.3,
                                    transform=ax.transAxes)
                ax.add_patch(rect)
                
                # Counterfactual info
                ax.text(0.06, y_pos, f"{i}. {cf['method']}: {cf['change']}",
                       fontsize=10, fontweight='bold', transform=ax.transAxes)
                
                arrow_text = (f"{pred['predicted_label']} → "
                             f"{cf_pred['predicted_label']} "
                             f"({cf_pred['confidence']:.0%})")
                
                ax.text(0.06, y_pos - 0.05, arrow_text,
                       fontsize=9, color='blue', transform=ax.transAxes)
                
                ax.text(0.06, y_pos - 0.10, f"Text: {cf_pred['text'][:80]}...",
                       fontsize=8, style='italic', transform=ax.transAxes)
                
                y_pos -= 0.18
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            cf_img = Image.open(buf)
            plt.close()
        else:
            # No counterfactuals found
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.axis('off')
            ax.text(0.5, 0.5, '⚠ No counterfactuals found\n(Model prediction is very confident)',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   transform=ax.transAxes)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            cf_img = Image.open(buf)
            plt.close()
        
        print("✅ Analysis complete!")
        
        return (prediction_html, attention_img, shap_img, lime_img,
                gradient_img, cf_img)
    
    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        print(error_msg)
        return error_msg, None, None, None, None, None


# ============================================================================
# Gradio Interface
# ============================================================================

def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Hindi Sentiment Analysis with XAI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 Hindi Sentiment Analysis with Explainable AI
        
        Enter Hindi text to analyze sentiment and view **6 different explainability methods**:
        1. **Attention Analysis** - Shows which tokens the model focuses on
        2. **SHAP** - Shows token contributions (positive/negative)
        3. **LIME** - Local interpretable model-agnostic explanations
        4. **Gradient Saliency** - Gradient-based importance
        5. **Counterfactuals** - What changes would flip the prediction?
        
        ### Examples:
        - **Positive**: यह फिल्म बहुत अच्छी थी, मुझे बेहद पसंद आई।
        - **Neutral**: खाना ठीक-ठाक था, कुछ खास नहीं।
        - **Negative**: सेवा बहुत खराब थी, बिल्कुल समय की बर्बादी।
        """)
        
        with gr.Row():
            input_text = gr.Textbox(
                label="📝 Enter Hindi Text",
                placeholder="यहाँ अपना हिंदी टेक्स्ट दर्ज करें...",
                lines=3
            )
        
        analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", size="lg")
        
        with gr.Row():
            prediction_output = gr.HTML(label="Prediction")
        
        gr.Markdown("## 📊 Explainability Visualizations")
        
        with gr.Row():
            with gr.Column():
                attention_output = gr.Image(label="🎯 Attention Analysis")
            with gr.Column():
                shap_output = gr.Image(label="📈 SHAP Explanation")
        
        with gr.Row():
            with gr.Column():
                lime_output = gr.Image(label="🔬 LIME Explanation")
            with gr.Column():
                gradient_output = gr.Image(label="🌊 Gradient Saliency")
        
        with gr.Row():
            cf_output = gr.Image(label="🔄 Counterfactual Examples")
        
        # Connect button to analysis function
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=[input_text],
            outputs=[prediction_output, attention_output, shap_output,
                    lime_output, gradient_output, cf_output]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["यह फिल्म बहुत अच्छी थी, मुझे बेहद पसंद आई।"],
                ["खाना ठीक-ठाक था, कुछ खास नहीं।"],
                ["सेवा बहुत खराब थी, बिल्कुल समय की बर्बादी।"],
                ["होटल का कमरा साफ और आरामदायक था।"],
                ["मुझे इस रेस्टोरेंट का अनुभव निराशाजनक लगा।"]
            ],
            inputs=[input_text]
        )
        
        gr.Markdown("""
        ---
        ### 📚 Method Descriptions:
        
        - **Attention**: Shows which words the BERT model pays attention to
        - **SHAP**: Game-theory based feature importance (positive values support prediction)
        - **LIME**: Approximates model locally with interpretable model
        - **Gradient Saliency**: Uses gradients to find influential tokens
        - **Counterfactuals**: Shows minimal changes that would flip the prediction
        
        ### 🎯 Model Details:
        - **Architecture**: BERT (bert-base-multilingual-cased)
        - **Task**: 3-class sentiment classification (Negative, Neutral, Positive)
        - **Language**: Hindi (Devanagari script)
        - **Device**: {device}
        """.format(device=DEVICE))
    
    return demo


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    print("\n" + "="*70)
    print("🚀 Launching Interactive Interface...")
    print("="*70)
    print(f"   Model: {MODEL_NAME}")
    print(f"   Device: {DEVICE}")
    print(f"   Access the interface in your web browser")
    print("="*70 + "\n")
    
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860
    )


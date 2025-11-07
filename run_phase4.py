"""
Run Phase 4: Gradient-based Interpretability
============================================
Generates gradient-based explanations for sentiment predictions.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model import BERTSentimentClassifier
from src.gradient_explainer import GradientExplainer
from config import MODEL_NAME, NUM_LABELS, DEVICE


def main():
    """Run Phase 4 - Gradient-based Interpretability."""
    
    print("=" * 70)
    print("Phase 4: Gradient-based Interpretability")
    print("=" * 70)
    print()
    
    # ========================================================================
    # 1. Load Model and Tokenizer
    # ========================================================================
    
    print("📦 Loading model and tokenizer...")
    
    model_path = Path('models/model.pt')
    if not model_path.exists():
        print("❌ Error: Model not found!")
        print("   Please run Phase 1 first to train the model.")
        return
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model
    model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Device: {DEVICE}")
    print()
    
    # ========================================================================
    # 2. Initialize Gradient Explainer
    # ========================================================================
    
    print("🔍 Initializing Gradient Explainer...")
    explainer = GradientExplainer(model, tokenizer)
    print("✓ Gradient explainer initialized")
    print()
    
    # ========================================================================
    # 3. Test Samples
    # ========================================================================
    
    test_samples = [
        "यह फिल्म बहुत अच्छी थी, मुझे बेहद पसंद आई।",  # Positive
        "खाना ठीक-ठाक था, कुछ खास नहीं।",  # Neutral
        "सेवा बहुत खराब थी, बिल्कुल समय की बर्बादी।"  # Negative
    ]
    
    # Create output directory
    output_dir = Path('outputs/phase4')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Analyzing {len(test_samples)} test samples...")
    print()
    
    # ========================================================================
    # 4. Generate Explanations for Each Sample
    # ========================================================================
    
    for idx, text in enumerate(test_samples, 1):
        print(f"\n{'='*70}")
        print(f"Sample {idx}: {text}")
        print(f"{'='*70}\n")
        
        # ====================================================================
        # Method 1: Saliency Maps
        # ====================================================================
        
        print("🔹 Computing Saliency Map...")
        saliency_result = explainer.compute_saliency(text)
        
        print(f"   Prediction: {['Negative', 'Neutral', 'Positive'][saliency_result['predicted_class']]}")
        print(f"   Confidence: {saliency_result['probabilities'][saliency_result['predicted_class']]:.2%}")
        
        # Visualize
        explainer.plot_gradient_attribution(
            saliency_result,
            text,
            save_path=output_dir / f'sample_{idx}_saliency.png',
            merge_subwords=True
        )
        
        # ====================================================================
        # Method 2: Integrated Gradients
        # ====================================================================
        
        print("🔹 Computing Integrated Gradients...")
        ig_result = explainer.compute_integrated_gradients(text, n_steps=50)
        
        # Visualize
        explainer.plot_gradient_attribution(
            ig_result,
            text,
            save_path=output_dir / f'sample_{idx}_integrated_gradients.png',
            merge_subwords=True
        )
        
        # ====================================================================
        # Method 3: Gradient × Input
        # ====================================================================
        
        print("🔹 Computing Gradient × Input...")
        grad_input_result = explainer.compute_gradient_x_input(text)
        
        # Visualize
        explainer.plot_gradient_attribution(
            grad_input_result,
            text,
            save_path=output_dir / f'sample_{idx}_grad_x_input.png',
            merge_subwords=True
        )
        
        # ====================================================================
        # Comparison Plot
        # ====================================================================
        
        print("🔹 Creating comparison visualization...")
        results_dict = {
            'Saliency Map': saliency_result,
            'Integrated Gradients': ig_result,
            'Gradient × Input': grad_input_result
        }
        
        explainer.plot_method_comparison(
            results_dict,
            text,
            save_path=output_dir / f'sample_{idx}_comparison.png',
            merge_subwords=True
        )
        
        print(f"✓ All visualizations saved for sample {idx}")
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("✅ Phase 4 Complete!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\n📊 Generated {len(test_samples) * 4} visualizations:")
    print("   • Saliency Maps - Shows gradient magnitude")
    print("   • Integrated Gradients - Integrates gradients from baseline")
    print("   • Gradient × Input - Combines gradients with embeddings")
    print("   • Comparison plots for each sample")
    print()
    print("🔬 Key Insights:")
    print("   • Saliency: Simple and fast, shows instantaneous importance")
    print("   • Integrated Gradients: More stable, accounts for full path")
    print("   • Grad×Input: Balances gradient and input magnitude")
    print()
    print("Next: Phase 5 - Counterfactual Analysis")
    print()


if __name__ == "__main__":
    main()


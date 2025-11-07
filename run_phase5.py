"""
Run Phase 5: Counterfactual Analysis
=====================================
Generates counterfactual examples showing minimal changes that flip predictions.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.model import BERTSentimentClassifier
from src.counterfactual_analyzer import CounterfactualAnalyzer
from config import MODEL_NAME, NUM_LABELS, DEVICE


def main():
    """Run Phase 5 - Counterfactual Analysis."""
    
    print("=" * 70)
    print("Phase 5: Counterfactual Analysis")
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
    # 2. Initialize Counterfactual Analyzer
    # ========================================================================
    
    print("🔄 Initializing Counterfactual Analyzer...")
    analyzer = CounterfactualAnalyzer(model, tokenizer)
    print("✓ Counterfactual analyzer initialized")
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
    output_dir = Path('outputs/phase5')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 Analyzing {len(test_samples)} test samples...")
    print()
    
    # ========================================================================
    # 4. Generate Counterfactuals for Each Sample
    # ========================================================================
    
    for idx, text in enumerate(test_samples, 1):
        print(f"\n{'='*70}")
        print(f"Sample {idx}: {text}")
        print(f"{'='*70}\n")
        
        # Get original prediction
        original = analyzer.predict(text)
        print(f"📊 Original Prediction: {original['predicted_label']}")
        print(f"   Confidence: {original['confidence']:.2%}")
        print(f"   Probabilities: Neg={original['probabilities'][0]:.2%}, "
              f"Neu={original['probabilities'][1]:.2%}, "
              f"Pos={original['probabilities'][2]:.2%}")
        print()
        
        # ====================================================================
        # Generate All Counterfactuals
        # ====================================================================
        
        print("🔄 Generating counterfactuals...")
        results = analyzer.analyze_all_counterfactuals(text, max_per_method=3)
        
        # Display results
        n_word_replacement = len(results['word_replacement'])
        n_word_removal = len(results['word_removal'])
        n_negation = len(results['negation'])
        total = n_word_replacement + n_word_removal + n_negation
        
        print(f"\n✓ Found {total} counterfactuals:")
        print(f"   • Word Replacement: {n_word_replacement}")
        print(f"   • Word Removal: {n_word_removal}")
        print(f"   • Negation Changes: {n_negation}")
        
        # Show examples
        if results['word_replacement']:
            print("\n   Example (Word Replacement):")
            cf = results['word_replacement'][0]
            print(f"   → {cf['change']}")
            print(f"   → New prediction: {cf['counterfactual']['predicted_label']} "
                  f"({cf['counterfactual']['confidence']:.2%})")
        
        if results['word_removal']:
            print("\n   Example (Word Removal):")
            cf = results['word_removal'][0]
            print(f"   → {cf['change']}")
            print(f"   → New prediction: {cf['counterfactual']['predicted_label']} "
                  f"({cf['counterfactual']['confidence']:.2%})")
        
        if results['negation']:
            print("\n   Example (Negation):")
            cf = results['negation'][0]
            print(f"   → {cf['change']}")
            print(f"   → New prediction: {cf['counterfactual']['predicted_label']} "
                  f"({cf['counterfactual']['confidence']:.2%})")
        
        # ====================================================================
        # Visualizations
        # ====================================================================
        
        print(f"\n📊 Creating visualizations...")
        
        # Counterfactual details plot
        analyzer.plot_counterfactuals(
            results,
            save_path=output_dir / f'sample_{idx}_counterfactuals.png'
        )
        
        # Probability comparison plot
        analyzer.plot_counterfactual_comparison(
            results,
            save_path=output_dir / f'sample_{idx}_probability_comparison.png'
        )
        
        print(f"✓ Visualizations saved for sample {idx}")
    
    # ========================================================================
    # 5. Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("✅ Phase 5 Complete!")
    print("="*70)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    print(f"\n🔄 Counterfactual Methods Used:")
    print("   1. Word Replacement - Replaces sentiment words with antonyms")
    print("   2. Word Removal - Removes influential words")
    print("   3. Negation Changes - Adds/removes negation words")
    print()
    print("💡 Key Insights:")
    print("   • Counterfactuals reveal model's decision boundaries")
    print("   • Shows which words are most influential for predictions")
    print("   • Helps identify model biases and weaknesses")
    print()
    print("Next: Phase 6 - Interactive Visualization Interface")
    print()


if __name__ == "__main__":
    main()


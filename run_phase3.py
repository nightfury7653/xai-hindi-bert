"""
Phase 3: SHAP and LIME Explainability

This script demonstrates model-agnostic explanation methods for the Hindi sentiment classifier:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations)
3. Comparative analysis between explanation methods
"""

import torch
import numpy as np
import os
import sys
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import BERTSentimentClassifier
from src.shap_lime_explainer import SHAPLIMEExplainer
from config import MODEL_NAME, NUM_LABELS

print(f"{'='*80}")
print("                 PHASE 3: SHAP & LIME EXPLAINABILITY")
print(f"{'='*80}\n")

# 1. Load model and tokenizer
print("🤖 STEP 1: Loading Model and Tokenizer")
print("-" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model
model = BERTSentimentClassifier(MODEL_NAME, NUM_LABELS)
model.load_state_dict(torch.load('models/model.pt', map_location=device))
model = model.to(device)
model.eval()

print(f"✓ Model loaded from: models/model.pt")
print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens\n")

# 2. Initialize explainer
print("🔍 STEP 2: Initializing SHAP/LIME Explainer")
print("-" * 80)

explainer = SHAPLIMEExplainer(model, tokenizer, device=device)
print("✓ SHAP/LIME explainer initialized\n")

# 3. Prepare test samples
print("📝 STEP 3: Preparing Test Samples")
print("-" * 80)

test_samples = [
    # Positive sentiment
    "यह फिल्म बहुत शानदार और मनोरंजक है",
    "खाना बहुत स्वादिष्ट था, मुझे बहुत पसंद आया",
    
    # Negative sentiment
    "बहुत खराब सेवा, बिल्कुल निराश हूं",
    "यह उत्पाद घटिया है, पैसे बर्बाद हुए",
    
    # Neutral sentiment
    "मैं कल दिल्ली जाऊंगा",
]

print(f"✓ Prepared {len(test_samples)} test samples\n")

# 4. Generate predictions
print("🎯 STEP 4: Model Predictions")
print("-" * 80)

label_names = ['Negative', 'Neutral', 'Positive']

for i, text in enumerate(test_samples, 1):
    probs = explainer.predict_proba([text])[0]
    pred_idx = probs.argmax()
    print(f'{i}. "{text}"')
    print(f'   → {label_names[pred_idx]} ({probs[pred_idx]*100:.1f}%)')
    print(f'   Distribution: Neg={probs[0]*100:.1f}%, Neu={probs[1]*100:.1f}%, Pos={probs[2]*100:.1f}%')
    print()

# 5. LIME Explanations
print(f"\n{'='*80}")
print("🔬 STEP 5: LIME (Local Interpretable Model-agnostic Explanations)")
print(f"{'='*80}\n")

print("LIME creates local linear approximations to explain individual predictions.")
print("It perturbs the input and observes changes in predictions.\n")

os.makedirs('outputs/phase3', exist_ok=True)

for i, text in enumerate(test_samples[:3], 1):  # First 3 samples
    print(f"\nSample {i}: \"{text}\"")
    print("-" * 70)
    
    lime_exp, probs, pred_class = explainer.explain_with_lime(text, num_features=10)
    
    print(f"Prediction: {label_names[pred_class]} ({probs[pred_class]*100:.1f}%)\n")
    
    print("Top contributing features (LIME):")
    for word, importance in lime_exp.as_list(label=pred_class)[:5]:
        direction = "→" if importance > 0 else "←"
        print(f"   {direction} {word}: {importance:+.4f}")
    
    # Generate visualization
    explainer.plot_lime_explanation(
        text,
        num_features=10,
        save_path=f'outputs/phase3/sample_{i}_lime.png'
    )

# 6. SHAP Explanations
print(f"\n\n{'='*80}")
print("🔬 STEP 6: SHAP (SHapley Additive exPlanations)")
print(f"{'='*80}\n")

print("SHAP uses game theory to assign each feature an importance value.")
print("It measures how much each word contributes to the prediction.\n")

for i, text in enumerate(test_samples[:3], 1):  # First 3 samples
    print(f"\nSample {i}: \"{text}\"")
    print("-" * 70)
    
    shap_values, probs, pred_class, words = explainer.explain_with_shap(text)
    
    print(f"Prediction: {label_names[pred_class]} ({probs[pred_class]*100:.1f}%)\n")
    
    # Get top features for predicted class
    class_shap = shap_values[:, pred_class]
    top_indices = np.argsort(np.abs(class_shap))[::-1][:5]
    
    print("Top contributing words (SHAP):")
    for idx in top_indices:
        value = class_shap[idx]
        direction = "→" if value > 0 else "←"
        print(f"   {direction} {words[idx]}: {value:+.4f}")
    
    # Generate visualization
    explainer.plot_shap_explanation(
        text,
        save_path=f'outputs/phase3/sample_{i}_shap.png'
    )

# 7. Comparative Analysis
print(f"\n\n{'='*80}")
print("📊 STEP 7: SHAP vs LIME Comparison")
print(f"{'='*80}\n")

print("Comparing SHAP and LIME explanations side-by-side...")
print("Both methods should identify similar important features.\n")

for i, text in enumerate(test_samples[:3], 1):
    print(f"\nGenerating comparison for sample {i}...")
    explainer.compare_explanations(
        text,
        save_path=f'outputs/phase3/sample_{i}_comparison.png'
    )

print(f"\n✓ All comparisons generated!")

# 8. Summary
print(f"\n\n{'='*80}")
print("✅ PHASE 3 COMPLETE!")
print(f"{'='*80}\n")

print("📁 Generated Outputs:")
print("   • LIME explanations (local linear approximations)")
print("   • SHAP explanations (game-theoretic attributions)")
print("   • Comparative visualizations (SHAP vs LIME)")
print()
print("📂 All visualizations saved to: outputs/phase3/")
print()

print("🔑 Key Insights:")
print("   1. LIME explains predictions through local approximations")
print("   2. SHAP provides theoretically grounded feature attributions")
print("   3. Both methods reveal which Hindi words drive sentiment")
print("   4. Consistent explanations across methods build trust")
print()

print(f"{'='*80}")
print("➡️  Next: Phase 4 - Gradient-based Interpretability")
print(f"{'='*80}\n")


"""
Phase 2: Attention Visualization

This script analyzes and visualizes BERT attention patterns for Hindi sentiment analysis.
It demonstrates:
1. How BERT attends to different words in Hindi sentences
2. Which tokens are most important for sentiment classification
3. How attention flows through different layers
4. Layer-wise attention patterns
"""

import torch
import os
import sys
from transformers import AutoTokenizer

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import BERTSentimentClassifier
from src.attention_analysis import AttentionAnalyzer, analyze_multiple_samples
from config import MODEL_NAME, NUM_LABELS

print(f"{'='*80}")
print("                    PHASE 2: ATTENTION VISUALIZATION")
print(f"{'='*80}\n")

# 1. Load model and tokenizer
print("🤖 STEP 1: Loading Model and Tokenizer")
print("-" * 80)

# Create model with attention outputs enabled
model = BERTSentimentClassifier(
    MODEL_NAME,
    NUM_LABELS,
    output_attentions=True  # Enable attention outputs
)

# Load trained weights
print(f"Loading trained model from: models/model.pt")
state_dict = torch.load('models/model.pt', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✓ Model loaded on device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('models/')
print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens\n")

# 2. Initialize analyzer
print("🔍 STEP 2: Initializing Attention Analyzer")
print("-" * 80)
analyzer = AttentionAnalyzer(model, tokenizer)
print("✓ Attention analyzer ready\n")

# 3. Prepare test samples
print("📝 STEP 3: Preparing Test Samples")
print("-" * 80)

test_samples = [
    # Positive samples
    "यह फिल्म बहुत शानदार और मनोरंजक है",
    "खाना बहुत स्वादिष्ट था, मुझे बहुत पसंद आया",
    
    # Negative samples
    "बहुत खराब सेवा, बिल्कुल निराश हूं",
    "यह उत्पाद घटिया है, पैसे बर्बाद हुए",
    
    # Neutral samples
    "मैं कल दिल्ली जाऊंगा",
    "फिल्म तीन घंटे लंबी थी",
]

print(f"✓ Prepared {len(test_samples)} test samples\n")

# 4. Quick predictions
print("🎯 STEP 4: Model Predictions")
print("-" * 80)

label_names = ['Negative', 'Neutral', 'Positive']

for i, text in enumerate(test_samples, 1):
    _, _, probs, _ = analyzer.get_attention_weights(text)
    pred_idx = torch.argmax(probs).item()
    pred_label = label_names[pred_idx]
    confidence = probs[pred_idx].item() * 100
    
    print(f"{i}. \"{text}\"")
    print(f"   → {pred_label} ({confidence:.1f}%)")
    print(f"   Distribution: Neg={probs[0]*100:.1f}%, Neu={probs[1]*100:.1f}%, Pos={probs[2]*100:.1f}%\n")

# 5. Detailed attention analysis for selected samples
print(f"\n{'='*80}")
print("🔬 STEP 5: Detailed Attention Analysis")
print(f"{'='*80}\n")

# Create output directory
os.makedirs('outputs/phase2', exist_ok=True)

# Analyze selected samples in detail
selected_samples = [
    test_samples[0],  # Positive
    test_samples[2],  # Negative
    test_samples[4],  # Neutral
]

for i, text in enumerate(selected_samples, 1):
    print(f"\n{'-'*80}")
    print(f"Sample {i}: \"{text}\"")
    print(f"{'-'*80}\n")
    
    # Get prediction
    _, tokens, probs, actual_len = analyzer.get_attention_weights(text)
    pred_idx = torch.argmax(probs).item()
    pred_label = label_names[pred_idx]
    confidence = probs[pred_idx].item() * 100
    
    print(f"Prediction: {pred_label} ({confidence:.1f}%)\n")
    
    # Token importance (with merged subwords for readability)
    print("📊 Word Importance Analysis:")
    token_df = analyzer.get_token_importance(text, merge_subwords=True)
    
    print("\nTop 10 Most Important Words:")
    count = 0
    for idx, row in token_df.iterrows():
        if row['Token'] not in ['[CLS]', '[SEP]', '[PAD]'] and count < 10:
            print(f"   {count+1}. {row['Token']:<20} {row['Normalized_Importance']:>6.2f}%")
            count += 1
    
    # Create visualizations
    print("\n📈 Generating Visualizations...")
    
    # Attention heatmap for last layer
    analyzer.plot_attention_heatmap(
        text,
        layer=-1,
        head=0,
        save_path=f'outputs/phase2/sample_{i}_attention_heatmap.png'
    )
    
    # Token importance bar chart
    analyzer.plot_token_importance(
        text,
        save_path=f'outputs/phase2/sample_{i}_token_importance.png'
    )
    
    # Attention summary across layers
    analyzer.plot_attention_summary(
        text,
        save_path=f'outputs/phase2/sample_{i}_attention_summary.png'
    )
    
    # Attention flow from [CLS] token
    analyzer.analyze_attention_flow(
        text,
        source_token_idx=0,
        save_path=f'outputs/phase2/sample_{i}_attention_flow.png'
    )
    
    print(f"✓ Visualizations saved for sample {i}")

# 6. Comparative analysis
print(f"\n{'='*80}")
print("📊 STEP 6: Comparative Analysis")
print(f"{'='*80}\n")

print("Analyzing attention patterns across sentiment classes...\n")

# Analyze all samples
analyze_multiple_samples(
    model,
    tokenizer,
    test_samples[:3],  # Analyze first 3 samples
    save_dir='outputs/phase2/detailed'
)

# 7. Summary
print(f"\n{'='*80}")
print("✅ PHASE 2 COMPLETE!")
print(f"{'='*80}\n")

print("📁 Generated Outputs:")
print("   • Attention heatmaps (layer-wise attention patterns)")
print("   • Token importance charts (which words matter most)")
print("   • Attention summaries (attention across all layers)")
print("   • Attention flow diagrams (how attention propagates)")
print()
print("📂 All visualizations saved to: outputs/phase2/")
print()

print("🔑 Key Insights:")
print("   1. BERT attention reveals which Hindi words influence sentiment")
print("   2. Different layers focus on different linguistic features")
print("   3. [CLS] token aggregates information from all tokens")
print("   4. Sentiment-bearing words receive higher attention weights")
print()

print(f"{'='*80}")
print("➡️  Next: Phase 3 - SHAP/LIME Explainability")
print(f"{'='*80}\n")
"""
Phase 1: BERT-based Hindi Sentiment Classifier - Complete Workflow

This script demonstrates the complete Phase 1 implementation.
Run this to train your sentiment model!

Usage:
    python run_phase1.py
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

# Import our modules
from src.data_preprocessing import (
    HindiTextCleaner,
    SentimentDataset,
    create_sample_dataset,
    prepare_data_splits,
    print_dataset_stats
)
from src.model import (
    BERTSentimentClassifier,
    train_epoch,
    evaluate,
    predict_text,
    save_model
)
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    """Main execution function for Phase 1."""
    
    print("="*80)
    print(" " * 20 + "PHASE 1: MODEL TRAINING")
    print("="*80)
    print("\n🚀 Starting Hindi Sentiment Analysis Pipeline\n")
    
    # ========================================================================
    # STEP 1: Create/Load Dataset
    # ========================================================================
    print("📊 STEP 1: Loading Dataset")
    print("-" * 80)
    
    # Option 1: Create sample dataset (for demo)
    print("Creating sample dataset with 900 samples...")
    df = create_sample_dataset(num_samples=900)
    
    # Option 2: Load your real dataset (uncomment when ready)
    # df = pd.read_csv('data/raw/hindi_sentiment.csv')
    
    print(f"✓ Dataset loaded: {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\n  Sample data:")
    print(df.head(3).to_string())
    
    # ========================================================================
    # STEP 2: Data Preprocessing
    # ========================================================================
    print("\n\n🧹 STEP 2: Data Preprocessing")
    print("-" * 80)
    
    cleaner = HindiTextCleaner(
        remove_emojis=False,
        remove_special_chars=False
    )
    
    df_clean = cleaner.clean_dataframe(df, text_column='text')
    print(f"✓ Text cleaned: {len(df_clean)} samples")
    print(f"  Removed {len(df) - len(df_clean)} invalid samples")
    
    # Check distribution
    print(f"\n  Class distribution:")
    print(df_clean['label'].value_counts().to_string())
    
    # ========================================================================
    # STEP 3: Train/Val/Test Split
    # ========================================================================
    print("\n\n✂️  STEP 3: Creating Data Splits")
    print("-" * 80)
    
    train_df, val_df, test_df = prepare_data_splits(
        df_clean,
        text_column='text',
        label_column='label',
        label_map=config.LABEL_MAP,
        train_size=config.TRAIN_SPLIT,
        val_size=config.VAL_SPLIT,
        random_state=config.RANDOM_SEED
    )
    
    print_dataset_stats(train_df, val_df, test_df, label_column='label')
    
    # ========================================================================
    # STEP 4: Initialize Tokenizer
    # ========================================================================
    print("\n\n🔤 STEP 4: Loading Tokenizer")
    print("-" * 80)
    print(f"Loading tokenizer for: {config.MODEL_NAME}")
    print("(This may take a minute on first run...)\n")
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print(f"✓ Tokenizer loaded")
    print(f"  Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Test tokenization
    sample_text = "यह फिल्म बहुत अच्छी है"
    tokens = tokenizer.tokenize(sample_text)
    print(f"\n  Example tokenization:")
    print(f"  Text: {sample_text}")
    print(f"  Tokens: {tokens}")
    
    # ========================================================================
    # STEP 5: Create Datasets and Loaders
    # ========================================================================
    print("\n\n📦 STEP 5: Creating PyTorch Datasets")
    print("-" * 80)
    
    train_dataset = SentimentDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    val_dataset = SentimentDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    test_dataset = SentimentDataset(
        texts=test_df['text'].tolist(),
        labels=test_df['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )
    
    print(f"✓ Datasets created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    # ========================================================================
    # STEP 6: Initialize Model
    # ========================================================================
    print("\n\n🤖 STEP 6: Initializing BERT Model")
    print("-" * 80)
    print(f"Loading model: {config.MODEL_NAME}")
    print("(Downloading pretrained weights - may take 2-5 minutes...)\n")
    
    model = BERTSentimentClassifier(
        model_name=config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        hidden_dropout=config.HIDDEN_DROPOUT
    )
    
    model = model.to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model initialized successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB")
    print(f"  Device: {config.DEVICE}")
    
    # ========================================================================
    # STEP 7: Setup Optimizer and Scheduler
    # ========================================================================
    print("\n\n⚙️  STEP 7: Configuring Optimizer")
    print("-" * 80)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer configured:")
    print(f"  Type: AdamW")
    print(f"  Learning rate: {config.LEARNING_RATE}")
    print(f"  Weight decay: {config.WEIGHT_DECAY}")
    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {config.WARMUP_STEPS}")
    
    # ========================================================================
    # STEP 8: Training
    # ========================================================================
    print("\n\n🎓 STEP 8: Training Model")
    print("=" * 80)
    print(f"Training for {config.NUM_EPOCHS} epochs...")
    print("=" * 80)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_accuracy = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            log_interval=config.LOG_INTERVAL
        )
        
        # Validate
        val_loss, val_accuracy, _, _ = evaluate(
            model=model,
            dataloader=val_loader,
            device=config.DEVICE
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print results
        print(f"\n📊 Results:")
        print(f"   Train Loss:    {train_loss:.4f}")
        print(f"   Val Loss:      {val_loss:.4f}")
        print(f"   Val Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_model(model, tokenizer, config.MODEL_DIR)
            print(f"   ✅ New best model saved!")
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETED!")
    print(f"   Best validation accuracy: {best_accuracy*100:.2f}%")
    print(f"{'='*80}")
    
    # ========================================================================
    # STEP 9: Test Set Evaluation
    # ========================================================================
    print("\n\n📈 STEP 9: Final Evaluation on Test Set")
    print("=" * 80)
    
    test_loss, test_accuracy, test_preds, test_labels = evaluate(
        model=model,
        dataloader=test_loader,
        device=config.DEVICE
    )
    
    print(f"\n✅ TEST RESULTS:")
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 80)
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    print("-" * 80)
    report = classification_report(
        test_labels,
        test_preds,
        target_names=['Negative', 'Neutral', 'Positive'],
        digits=4
    )
    print(report)
    
    # Confusion matrix
    print("\n🔢 Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    
    # ========================================================================
    # STEP 10: Test Predictions
    # ========================================================================
    print("\n\n🧪 STEP 10: Testing with Sample Sentences")
    print("=" * 80)
    
    test_sentences = [
        "यह फिल्म बहुत शानदार है, मुझे बहुत पसंद आई",
        "खाना बिल्कुल बेस्वाद था, बहुत निराश हुआ",
        "मैंने किताब पढ़ ली",
        "सेवा अच्छी नहीं थी",
        "मौसम आज ठीक है",
    ]
    
    for i, text in enumerate(test_sentences, 1):
        pred_label, probs, pred_class = predict_text(
            model=model,
            text=text,
            tokenizer=tokenizer,
            device=config.DEVICE,
            max_length=config.MAX_LENGTH
        )
        
        print(f"\n{i}. Text: {text}")
        print(f"   Prediction: {pred_class.upper()}")
        print(f"   Confidence: Neg={probs[0]:.1%}, Neu={probs[1]:.1%}, Pos={probs[2]:.1%}")
        print("   " + "-"*70)
    
    # ========================================================================
    # STEP 11: Save Results
    # ========================================================================
    print("\n\n💾 STEP 11: Saving Results")
    print("=" * 80)
    
    # Save plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/phase1_training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Training curves saved to: outputs/phase1_training_curves.png")
    
    # Confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=['Negative', 'Neutral', 'Positive'],
        yticklabels=['Negative', 'Neutral', 'Positive']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('outputs/phase1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved to: outputs/phase1_confusion_matrix.png")
    
    print("\n" + "=" * 80)
    print("🎉 PHASE 1 COMPLETE!")
    print("=" * 80)
    print("\n✅ Deliverables:")
    print("   • Trained model: models/model.pt")
    print("   • Tokenizer: models/")
    print("   • Training curves: outputs/phase1_training_curves.png")
    print("   • Confusion matrix: outputs/phase1_confusion_matrix.png")
    print("\n➡️  Next: Open phase2_attention_analysis.ipynb for explainability!")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    # Run main pipeline
    main()


"""
Data preprocessing utilities for Hindi sentiment analysis.

This module handles:
- Text cleaning and normalization
- Dataset loading and preparation
- Train/validation/test splitting
- Tokenization for BERT models
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class HindiTextCleaner:
    """
    Cleans and preprocesses Hindi text data.
    """
    
    def __init__(self, remove_emojis: bool = False, remove_special_chars: bool = False):
        """
        Initialize the text cleaner.
        
        Args:
            remove_emojis: Whether to remove emoji characters
            remove_special_chars: Whether to remove special characters (use carefully!)
        """
        self.remove_emojis = remove_emojis
        self.remove_special_chars = remove_special_chars
        
        # Emoji pattern
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove emojis if specified
        if self.remove_emojis:
            text = self.emoji_pattern.sub(r'', text)
        
        # Remove special characters (be careful - might affect Hindi punctuation)
        if self.remove_special_chars:
            text = re.sub(r'[^\w\s।,?!]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Clean text column in a dataframe.
        
        Args:
            df: Input dataframe
            text_column: Name of the column containing text
            
        Returns:
            Dataframe with cleaned text
        """
        df = df.copy()
        df[text_column] = df[text_column].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df[text_column].str.len() > 0]
        
        return df


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis.
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of label integers (0, 1, 2 for negative, neutral, positive)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Sentiment label
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_dataset(file_path: str, text_column: str, label_column: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Loaded dataframe
    """
    df = pd.read_csv(file_path)
    
    # Verify columns exist
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in dataset")
    
    return df


def create_sample_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Create a sample Hindi sentiment dataset for testing.
    
    This is a placeholder - in real project, you'll load actual data.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Sample dataframe with text and labels
    """
    
    # Sample Hindi sentences (positive, neutral, negative)
    positive_samples = [
        "यह फिल्म बहुत अच्छी है",
        "मुझे यह किताब बेहद पसंद आई",
        "आज का दिन बहुत सुंदर है",
        "यह खाना स्वादिष्ट है",
        "मैं बहुत खुश हूं",
        "यह प्रदर्शन शानदार था",
        "सेवा उत्कृष्ट थी",
        "यह एक अद्भुत अनुभव था"
    ]
    
    neutral_samples = [
        "मैं घर जा रहा हूं",
        "आज बुधवार है",
        "यह एक साधारण दिन है",
        "मैंने खाना खा लिया",
        "फिल्म तीन घंटे की है",
        "मौसम ठीक है",
        "यह एक सामान्य किताब है"
    ]
    
    negative_samples = [
        "यह फिल्म बहुत बुरी है",
        "मुझे यह पसंद नहीं आया",
        "सेवा बहुत खराब थी",
        "यह एक निराशाजनक अनुभव था",
        "मैं बहुत दुखी हूं",
        "खाना बेस्वाद था",
        "यह बिल्कुल घटिया है",
        "मैं इससे नाखुश हूं"
    ]
    
    # Generate balanced dataset
    data = []
    for i in range(num_samples // 3):
        data.append({
            'text': np.random.choice(positive_samples),
            'label': 'positive'
        })
        data.append({
            'text': np.random.choice(neutral_samples),
            'label': 'neutral'
        })
        data.append({
            'text': np.random.choice(negative_samples),
            'label': 'negative'
        })
    
    df = pd.DataFrame(data)
    return df


def prepare_data_splits(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    label_map: Dict[str, int],
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        text_column: Name of text column
        label_column: Name of label column
        label_map: Dictionary mapping label names to integers
        train_size: Proportion for training
        val_size: Proportion for validation
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Map labels to integers
    df = df.copy()
    df['label_id'] = df[label_column].map(label_map)
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df['label_id']
    )
    
    # Second split: val vs test
    val_ratio = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        stratify=temp_df['label_id']
    )
    
    return train_df, val_df, test_df


def get_class_distribution(df: pd.DataFrame, label_column: str) -> pd.Series:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        df: Input dataframe
        label_column: Name of label column
        
    Returns:
        Series with class counts
    """
    return df[label_column].value_counts()


def print_dataset_stats(train_df, val_df, test_df, label_column='label'):
    """
    Print statistics about the dataset splits.
    """
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal samples: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Training:   {len(train_df):5d} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  Validation: {len(val_df):5d} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"  Test:       {len(test_df):5d} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    
    print("\nClass distribution:")
    print("\nTraining set:")
    print(get_class_distribution(train_df, label_column))
    print("\nValidation set:")
    print(get_class_distribution(val_df, label_column))
    print("\nTest set:")
    print(get_class_distribution(test_df, label_column))
    print("=" * 60)


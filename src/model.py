"""
BERT-based sentiment classification model.

This module contains:
- Model architecture
- Training and evaluation functions
- Model saving/loading utilities
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm


class BERTSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier with a classification head.
    """
    
    def __init__(
        self,
        model_name: str,
        num_labels: int = 3,
        hidden_dropout: float = 0.1,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_name: Name of the pretrained BERT model
            num_labels: Number of sentiment classes (3: positive, neutral, negative)
            hidden_dropout: Dropout rate for the classification head
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
        """
        super(BERTSentimentClassifier, self).__init__()
        
        # Load pretrained BERT model
        config = AutoConfig.from_pretrained(
            model_name,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(hidden_dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize the weights of the classification head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Ground truth labels [batch_size] (optional)
            
        Returns:
            Dictionary containing:
            - logits: Class logits [batch_size, num_labels]
            - loss: Cross-entropy loss (if labels provided)
            - attentions: Attention weights (if output_attentions=True)
            - hidden_states: Hidden states (if output_hidden_states=True)
        """
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Prepare output dictionary
        output_dict = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            output_dict['loss'] = loss
        
        # Add attention weights if requested
        if self.output_attentions and hasattr(outputs, 'attentions'):
            output_dict['attentions'] = outputs.attentions
        
        # Add hidden states if requested
        if self.output_hidden_states and hasattr(outputs, 'hidden_states'):
            output_dict['hidden_states'] = outputs.hidden_states
        
        return output_dict


def train_epoch(
    model: BERTSentimentClassifier,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    log_interval: int = 10
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: The sentiment classifier
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        log_interval: How often to log progress
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(
    model: BERTSentimentClassifier,
    dataloader,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The sentiment classifier
        dataloader: Validation/test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (loss, accuracy, predictions, true_labels)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def predict_text(
    model: BERTSentimentClassifier,
    text: str,
    tokenizer,
    device: torch.device,
    max_length: int = 128
) -> Tuple[int, torch.Tensor, str]:
    """
    Predict sentiment for a single text.
    
    Args:
        model: The sentiment classifier
        text: Input text string
        tokenizer: Tokenizer
        device: Device to run prediction on
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (predicted_label, probabilities, predicted_class_name)
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs['logits']
    
    # Get prediction
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    # Map to class name
    id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_class = id_to_label[predicted_label]
    
    return predicted_label, probabilities[0], predicted_class


def save_model(
    model: BERTSentimentClassifier,
    tokenizer,
    path: str
):
    """
    Save model and tokenizer.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        path: Directory path to save to
    """
    import os
    os.makedirs(path, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    
    # Save tokenizer
    tokenizer.save_pretrained(path)
    
    print(f"Model and tokenizer saved to {path}")


def load_model(
    model_name: str,
    path: str,
    num_labels: int = 3,
    device: torch.device = None
) -> Tuple[BERTSentimentClassifier, any]:
    """
    Load model and tokenizer.
    
    Args:
        model_name: Name of the pretrained BERT model
        path: Directory path to load from
        num_labels: Number of sentiment classes
        device: Device to load model to
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer
    import os
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BERTSentimentClassifier(model_name, num_labels=num_labels)
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    print(f"Model and tokenizer loaded from {path}")
    
    return model, tokenizer


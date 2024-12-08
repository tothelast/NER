import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence  
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import classification_report, precision_recall_curve

def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, num_epochs: int = 10):
    """Training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            words = batch['words'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            
            optimizer.zero_grad()
            logits = model(words, lengths)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            
            loss = criterion(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                words = batch['words'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths']
                
                logits = model(words, lengths)
                logits_flat = logits.view(-1, logits.shape[-1])
                labels_flat = labels.view(-1)
                
                loss = criterion(logits_flat, labels_flat)
                total_val_loss += loss.item()
                
                # Store predictions and labels for metrics
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_discontinuous_model.pt')
        
        # Print classification report 
        if (epoch + 1) % 5 == 0:
            print('\nClassification Report:')
            print(classification_report(all_labels, all_preds))
    
    return train_losses, val_losses

def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable length sequences.
    Pads sequences in the batch to the length of the longest sequence.
    """
    # Sort batch by sequence length (descending) to optimize LSTM processing
    batch = sorted(batch, key=lambda x: x['lengths'], reverse=True)
    
    # Get the individual components
    words = [x['words'] for x in batch]
    labels = [x['labels'] for x in batch]
    lengths = torch.tensor([x['lengths'] for x in batch])
    
    # Pad sequences
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {
        'words': words_padded,
        'labels': labels_padded,
        'lengths': lengths
    }

class DiscontinuousNERDataset(Dataset):
    """
    Dataset class for handling discontinuous NER data with proper padding.
    All sequences in a batch will be padded to the length of the longest sequence
    in that specific batch.
    """
    def __init__(self, data_path: Path, word2idx: Dict[str, int], label2idx: Dict[str, int]):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.word2idx = word2idx
        self.label2idx = label2idx
        
        # Print statistics
        lengths = [len(item[0]) for item in self.data]
        print(f"\nDataset statistics for {data_path}:")
        print(f"Number of sequences: {len(self.data)}")
        print(f"Max sequence length: {max(lengths)}")
        print(f"Min sequence length: {min(lengths)}")
        print(f"Average sequence length: {sum(lengths)/len(lengths):.2f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        words_pos, labels = self.data[idx]
        
        # Convert words and labels to indices
        word_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                       for word, _ in words_pos]
        label_indices = [self.label2idx[label] for label in labels]
        
        # Return tensors without padding - padding will be done at batch level
        return {
            'words': torch.tensor(word_indices, dtype=torch.long),
            'labels': torch.tensor(label_indices, dtype=torch.long),
            'lengths': len(word_indices)
        }


class DiscontinuousLSTM(nn.Module):
    """LSTM model adapted for discontinuous NER with proper sequence handling"""
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int, num_labels: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                           bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)
    
    def forward(self, words, lengths):
        # Embed the input
        embedded = self.embedding(words)
        
        # Pack the padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Process with LSTM
        lstm_out, _ = self.lstm(packed)
        
        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        
        # Apply dropout and project to label space
        lstm_out = self.dropout(lstm_out)
        logits = self.hidden2label(lstm_out)
        
        return logits



__all__ = ['DiscontinuousNERDataset', 'DiscontinuousLSTM', 'train_model', 'collate_batch']
globals()['collate_batch'] = collate_batch
"""
BiLSTM-CRF Model for Discontinuous Named Entity Recognition
--------------------------------------------------------
This module implements a Bidirectional LSTM with Conditional Random Fields (CRF) layer
specifically designed for handling discontinuous named entity recognition tasks.

Architecture Overview:
    [Input] -> [Embeddings] -> [BiLSTM] -> [CRF] -> [Output]
        ↓          ↓            ↓           ↓          ↓
      Words     Word+POS     Context    Sequence    Entity
      +POS     Vectors      Encoding    Scoring     Labels
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

class NERDataset(Dataset):
    """
    Custom Dataset Class for NER Processing
    =====================================
    
    Input Format:                Output Format:
    [                           [
      (word1, POS1)    →         tensor([1,2,3...])  # Word indices
      (word2, POS2)    →         tensor([4,5,6...])  # POS indices
      ...                        tensor([7,8,9...])  # Label indices
    ]                           ]
    """
    def __init__(self, sentences: List[List[Tuple[str, str]]], labels: List[List[str]], 
                 word2idx: Dict[str, int], label2idx: Dict[str, int], pos2idx: Dict[str, int]):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.pos2idx = pos2idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label_seq = self.labels[idx]
        
        # Convert words and POS tags to indices
        word_indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word, _ in sentence]
        pos_indices = [self.pos2idx.get(pos, self.pos2idx['<UNK>']) for _, pos in sentence]
        label_indices = [self.label2idx[label] for label in label_seq]
        
        return (torch.tensor(word_indices), 
                torch.tensor(pos_indices), 
                torch.tensor(label_indices))

class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF Neural Network for Named Entity Recognition
    """
    def __init__(self, vocab_size: int, pos_size: int, embedding_dim: int, 
                 hidden_dim: int, num_layers: int, num_labels: int, dropout: float = 0.5):
        super(BiLSTMCRF, self).__init__()
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_size, embedding_dim // 4)
        
        # Combined embedding dimension
        combined_dim = embedding_dim + (embedding_dim // 4)
        
        # LSTM layer
        self.lstm = nn.LSTM(combined_dim, hidden_dim, num_layers, 
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        
        # Output layers
        self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        # CRF transitions
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        
    def _get_lstm_features(self, word_ids, pos_ids, lengths):
        """Extract features using BiLSTM"""
        # Embeddings
        word_embeds = self.word_embeddings(word_ids)
        pos_embeds = self.pos_embeddings(pos_ids)
        combined_embeds = torch.cat((word_embeds, pos_embeds), dim=-1)
        
        # Pack padded sequence
        packed_embeds = pack_padded_sequence(combined_embeds, lengths, 
                                           batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(packed_embeds)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Project to label space
        lstm_feats = self.hidden2label(self.dropout(lstm_out))
        return lstm_feats

    def _score_sentence(self, feats, tags):
        """Score a batch of sentences"""
        # Handle both batched and unbatched inputs
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        if tags.dim() == 1:
            tags = tags.unsqueeze(0)
        
        batch_size = feats.size(0)
        seq_length = feats.size(1)
        
        # Initialize scores
        scores = torch.zeros(batch_size).to(feats.device)
        
        # Add start transition scores
        start_tag = torch.full((batch_size, 1), 0, dtype=torch.long).to(feats.device)
        tags = torch.cat([start_tag, tags], dim=1)
        
        # Add transition and emission scores
        for i in range(seq_length):
            trans_score = self.transitions[tags[:, i+1], tags[:, i]]
            emit_score = torch.gather(feats[:, i], 1, tags[:, i+1].unsqueeze(1)).squeeze(1)
            scores += trans_score + emit_score
        
        return scores

    def _forward_alg(self, feats):
        """Forward algorithm for CRF"""
        # Handle both batched and unbatched inputs
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        
        batch_size = feats.size(0)
        seq_length = feats.size(1)
        num_tags = feats.size(2)
        
        # Initialize forward variables
        init_alphas = torch.full((batch_size, num_tags), -10000.).to(feats.device)
        init_alphas[:, 0] = 0.
        
        forward_var = init_alphas
        
        # Iterate through the sentence
        for feat in feats.transpose(0, 1):  # feat: (batch_size, num_tags)
            alphas_t = []
            for next_tag in range(num_tags):
                emit_score = feat[:, next_tag].view(batch_size, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(torch.logsumexp(next_tag_var, dim=1))
            forward_var = torch.stack(alphas_t, dim=1)
        
        terminal_var = forward_var + self.transitions[-1].view(1, -1)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def neg_log_likelihood(self, word_ids, pos_ids, tags, lengths):
        """Calculate negative log likelihood loss"""
        feats = self._get_lstm_features(word_ids, pos_ids, lengths)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return (forward_score - gold_score).mean()

    def _viterbi_decode(self, feats):
        """Viterbi decoding"""
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        
        batch_size = feats.size(0)
        seq_length = feats.size(1)
        num_tags = feats.size(2)
        
        # Initialize variables
        viterbi = torch.zeros((batch_size, seq_length, num_tags)).to(feats.device)
        backpointers = torch.zeros((batch_size, seq_length, num_tags), dtype=torch.long).to(feats.device)
        
        # Initialize with starting probabilities
        viterbi[:, 0] = feats[:, 0]
        
        # Iterate through the sequence
        for t in range(1, seq_length):
            next_tag_var = viterbi[:, t-1].unsqueeze(2) + self.transitions.unsqueeze(0)
            best_tag_id = torch.argmax(next_tag_var, dim=1)
            bptrs_t = best_tag_id
            viterbivars_t = torch.gather(next_tag_var, 1, best_tag_id.unsqueeze(1))
            viterbi[:, t] = viterbivars_t.squeeze(1) + feats[:, t]
            backpointers[:, t] = bptrs_t
        
        # Find best path
        best_path_scores, best_path = torch.max(viterbi[:, -1], dim=1)
        
        # Follow the backpointers to decode the best path
        best_paths = []
        for i in range(batch_size):
            best_path_i = [best_path[i].item()]
            for bptrs_t in reversed(backpointers[i]):
                best_path_i.append(bptrs_t[best_path_i[-1]].item())
            best_path_i.reverse()
            best_paths.append(best_path_i)
        
        return best_path_scores, best_paths[0] if batch_size == 1 else best_paths

    def forward(self, word_ids, pos_ids, lengths):
        """Forward pass"""
        lstm_feats = self._get_lstm_features(word_ids, pos_ids, lengths)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def train_lstm_model(train_data: List[Tuple[List[Tuple[str, str]], List[str]]], 
                    dev_data: List[Tuple[List[Tuple[str, str]], List[str]]], 
                    num_epochs: int = 10,
                    batch_size: int = 16):
    """Train the BiLSTM-CRF model"""
    # Create vocabularies
    word2idx = defaultdict(lambda: len(word2idx))
    pos2idx = defaultdict(lambda: len(pos2idx))
    label2idx = defaultdict(lambda: len(label2idx))
    
    # Add special tokens
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    pos2idx['<PAD>'] = 0
    pos2idx['<UNK>'] = 1
    
    # Build vocabularies from training data
    for sentence, labels in train_data:
        for (word, pos), label in zip(sentence, labels):
            word2idx[word]
            pos2idx[pos]
            label2idx[label]
    
    # Convert defaultdict to regular dict
    word2idx = dict(word2idx)
    pos2idx = dict(pos2idx)
    label2idx = dict(label2idx)
    
    # Create datasets
    train_dataset = NERDataset([s[0] for s in train_data], 
                              [s[1] for s in train_data],
                              word2idx, label2idx, pos2idx)
    dev_dataset = NERDataset([s[0] for s in dev_data],
                            [s[1] for s in dev_data],
                            word2idx, label2idx, pos2idx)
    
    def collate_fn(batch):
        """Collate function for DataLoader"""
        words, pos, labels = zip(*batch)
        lengths = torch.tensor([len(x) for x in words])
        
        words_padded = pad_sequence(words, batch_first=True)
        pos_padded = pad_sequence(pos, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True)
        
        return words_padded, pos_padded, labels_padded, lengths
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = BiLSTMCRF(
        vocab_size=len(word2idx),
        pos_size=len(pos2idx),
        embedding_dim=100,
        hidden_dim=256,
        num_layers=2,
        num_labels=len(label2idx),
        dropout=0.5
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_dev_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for words, pos, labels, lengths in train_loader:
            # Move to device
            words = words.to(device)
            pos = pos.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            model.zero_grad()
            loss = model.neg_log_likelihood(words, pos, labels, lengths)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluation on dev set
        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for words, pos, labels, lengths in dev_loader:
                words = words.to(device)
                pos = pos.to(device)
                labels = labels.to(device)
                lengths = lengths.to(device)
                
                loss = model.neg_log_likelihood(words, pos, labels, lengths)
                dev_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Dev Loss: {dev_loss/len(dev_loader):.4f}')
        
        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), 'best_lstm_model.pt')
    
    return model, word2idx, pos2idx, label2idx

def predict_with_lstm(model, sentence: List[Tuple[str, str]], 
                     word2idx: Dict[str, int], pos2idx: Dict[str, int], 
                     idx2label: Dict[int, str], device: torch.device):
    """Make predictions using trained model"""
    model.eval()
    
    # Convert input to tensors
    word_indices = [word2idx.get(word, word2idx['<UNK>']) for word, _ in sentence]
    pos_indices = [pos2idx.get(pos, pos2idx['<UNK>']) for _, pos in sentence]
    
    word_tensor = torch.tensor(word_indices).unsqueeze(0).to(device)
    pos_tensor = torch.tensor(pos_indices).unsqueeze(0).to(device)
    lengths = torch.tensor([len(sentence)])
    
    with torch.no_grad():
        _, tag_seq = model(word_tensor, pos_tensor, lengths)
    
    return [idx2label[t] for t in tag_seq]

if __name__ == "__main__":
    # Example usage
    train_data = [
        (
            [("Anthony", "NNP"), ("lives", "VBZ"), ("in", "IN"), ("Tucson", "NNP"), ("York", "NNP")],
            ["B-PER", "O", "O", "B-LOC", "I-LOC"]
        ),
        (
            [("Mary", "NNP"), ("works", "VBZ"), ("at", "IN"), ("Google", "NNP")],
            ["B-PER", "O", "O", "B-ORG"]
        )
    ]
    
    dev_data = [
        (
            [("Peter", "NNP"), ("visited", "VBD"), ("Paris", "NNP")],
            ["B-PER", "O", "B-LOC"]
        )
    ]
    
    # Train model
    print("Starting training...")
    model, word2idx, pos2idx, label2idx = train_lstm_model(
        train_data=train_data,
        dev_data=dev_data,
        num_epochs=5,
        batch_size=2
    )
    
    # Test prediction
    test_sentence = [("Sarah", "NNP"), ("visited", "VBD"), ("London", "NNP")]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inverse mapping for labels
    idx2label = {v: k for k, v in label2idx.items()}
    
    # Make prediction
    predictions = predict_with_lstm(
        model=model,
        sentence=test_sentence,
        word2idx=word2idx,
        pos2idx=pos2idx,
        idx2label=idx2label,
        device=device
    )
    
    print("\nTest Prediction:")
    for (word, pos), label in zip(test_sentence, predictions):
        print(f"{word} ({pos}) -> {label}")
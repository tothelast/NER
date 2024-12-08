import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from typing import List, Tuple, Dict
import torch.optim as optim

class NERDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[List[str]], 
                 word2idx: Dict[str, int], label2idx: Dict[str, int], max_len: int = 100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert words to indices and pad
        words = self.texts[idx].split()
        word_ids = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        word_ids = word_ids[:self.max_len] + [0] * (self.max_len - len(word_ids))
        
        # Convert labels to indices and pad
        label_ids = [self.label2idx[l] for l in self.labels[idx]]
        label_ids = label_ids[:self.max_len] + [0] * (self.max_len - len(label_ids))
        
        return {
            'input_ids': torch.tensor(word_ids),
            'labels': torch.tensor(label_ids),
            'attention_mask': torch.tensor([1] * min(len(words), self.max_len) + 
                                        [0] * (self.max_len - min(len(words), self.max_len)))
        }

class LSTMforNER(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_labels: int, num_layers: int = 2, dropout: float = 0.1):
        super(LSTMforNER, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)  # *2 for bidirectional
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        
        if attention_mask is not None:
            embedded = embedded * attention_mask.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        
        return logits

def train_model(model, train_dataloader, val_dataloader, num_epochs=5, learning_rate=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        if val_dataloader is not None and len(val_dataloader) > 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    outputs = outputs.view(-1, outputs.shape[-1])
                    labels = labels.view(-1)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            print(f'Validation Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pt')
        else:
            print("No validation data available")
            # Save model at each epoch when no validation data is available
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')

def evaluate_model(model, test_dataloader, idx2label):
    if test_dataloader is None or len(test_dataloader) == 0:
        print("No test data available for evaluation")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=-1)
            
            # Only consider predictions where attention_mask is 1
            for i in range(len(preds)):
                mask = attention_mask[i]
                pred = preds[i][mask == 1]
                label = labels[i][mask == 1]
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
    
    if len(all_preds) > 0:
        # Convert indices back to labels
        pred_labels = [idx2label[idx] for idx in all_preds]
        true_labels = [idx2label[idx] for idx in all_labels]
        
        # Print classification report
        print(classification_report(true_labels, pred_labels))
    else:
        print("No predictions were made")

# Example usage:
def prepare_sample_data():
    """Prepare larger sample data with diverse NER instances"""
    texts = [
        # Tech Companies and People
        "Apple CEO Tim Cook announced new products at their headquarters in Cupertino",
        "Microsoft's Satya Nadella spoke about AI at the conference in Seattle",
        "Jeff Bezos founded Amazon in Seattle back in 1994",
        "Mark Zuckerberg created Facebook while studying at Harvard University",
        "Sundar Pichai leads Google and its parent company Alphabet",
        
        # Universities and Locations
        "Students from Stanford University often intern at Silicon Valley companies",
        "MIT and Harvard are located in Cambridge, Massachusetts",
        "The University of Oxford is situated in England, United Kingdom",
        "Northwestern University's main campus is in Evanston, Illinois",
        
        # Mixed Entities
        "Tesla opened a new factory in Austin, Texas last year",
        "Intel announced their partnership with TSMC in Taiwan",
        "The New York Times reported on events in Washington, DC",
        "Researchers at IBM's facility in Tokyo, Japan made a breakthrough",
        
        # Sports Teams and Venues
        "The Los Angeles Lakers played against the Boston Celtics at Madison Square Garden",
        "Manchester United faced Real Madrid at Old Trafford stadium",
        
        # Entertainment
        "Warner Bros Studios in Hollywood produced several blockbusters",
        "Netflix CEO Reed Hastings spoke at the conference in Las Vegas",
        
        # Negative Examples
        "The weather was particularly nice today",
        "No specific entities mentioned in this sentence",
        "She went to the store yesterday afternoon",
        
        # Multiple Entities
        "Both Apple and Google compete while Microsoft focuses on cloud services",
        "When Steve Jobs and Bill Gates met at the conference in San Francisco",
        "Amazon and Facebook opened offices in London and Paris last year",
        
        # Academic and Research
        "Professor Sarah Johnson from Johns Hopkins University published a paper",
        "The research team at Berkeley Lab made significant discoveries",
        
        # Financial
        "Morgan Stanley and Goldman Sachs reported their quarterly earnings",
        "JPMorgan Chase opened a new branch in Chicago, Illinois",
        
        # International
        "Toyota's headquarters in Toyota City, Japan announced new plans",
        "Samsung's facility in Seoul, South Korea expanded operations"
    ]
    
    labels = [
        ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "B-LOC"],
        ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "B-LOC"],
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O", "O", "O", "O"],
        ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O", "B-ORG", "I-ORG"],
        ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "O", "B-ORG"],
        
        ["O", "O", "B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-ORG", "O", "B-ORG", "O", "O", "O", "B-LOC", "B-LOC"],
        ["O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "B-LOC", "B-LOC"],
        ["B-ORG", "I-ORG", "O", "O", "O", "O", "B-LOC", "B-LOC"],
        
        ["B-ORG", "O", "O", "O", "O", "B-LOC", "B-LOC", "O", "O"],
        ["B-ORG", "O", "O", "O", "O", "B-ORG", "O", "B-LOC"],
        ["O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O", "B-LOC", "I-LOC"],
        ["O", "O", "B-ORG", "O", "O", "O", "B-LOC", "B-LOC", "O", "O", "O"],
        
        ["O", "B-ORG", "I-ORG", "O", "O", "O", "B-ORG", "I-ORG", "O", "B-LOC", "I-LOC", "I-LOC"],
        ["B-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "O", "B-LOC", "I-LOC", "O"],
        
        ["B-ORG", "I-ORG", "I-ORG", "O", "B-LOC", "O", "O", "O"],
        ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "B-LOC", "I-LOC"],
        
        ["O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O"],
        ["O", "O", "O", "O", "O", "O", "O"],
        
        ["O", "B-ORG", "O", "B-ORG", "O", "O", "B-ORG", "O", "O", "O", "O"],
        ["O", "B-PER", "I-PER", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "B-LOC", "I-LOC"],
        ["B-ORG", "O", "B-ORG", "O", "O", "O", "B-LOC", "O", "B-LOC", "O", "O"],
        
        ["O", "B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O", "O"],
        ["O", "O", "O", "O", "B-ORG", "I-ORG", "O", "O", "O"],
        
        ["B-ORG", "I-ORG", "O", "B-ORG", "I-ORG", "O", "O", "O", "O"],
        ["B-ORG", "I-ORG", "O", "O", "O", "O", "O", "B-LOC", "B-LOC"],
        
        ["B-ORG", "O", "O", "O", "B-LOC", "I-LOC", "B-LOC", "O", "O", "O"],
        ["B-ORG", "O", "O", "B-LOC", "B-LOC", "O", "O", "O"]
    ]
    
    # Create vocabulary and label mappings
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    label2idx = {'O': 0}
    
    # Build vocabularies
    for text in texts:
        for word in text.split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)
    
    for label_seq in labels:
        for label in label_seq:
            if label not in label2idx:
                label2idx[label] = len(label2idx)
    
    return texts, labels, word2idx, label2idx

def print_dataset_split(dataset, indices, split_name=""):
    """Helper function to print the contents of a dataset split"""
    print(f"\n{split_name} Dataset Contents:")
    print("-" * 50)
    for idx in indices:
        original_text = dataset.texts[idx]
        original_labels = dataset.labels[idx]
        print(f"Text: {original_text}")
        print(f"Labels: {original_labels}")
        print("-" * 30)

def main():
    # Prepare data
    texts, labels, word2idx, label2idx = prepare_sample_data()
    
    # Create dataset
    dataset = NERDataset(texts, labels, word2idx, label2idx)
    
    # Calculate minimum sizes for splits
    total_size = len(dataset)
    min_size = 2  # Minimum size for each split
    
    print(f"\nTotal dataset size: {total_size} examples")
    
    if total_size < 3 * min_size:
        print(f"Dataset too small for splitting. Using all data for training.")
        train_size = total_size
        val_size = 0
        test_size = 0
    else:
        # Calculate sizes ensuring minimum size for each split
        train_size = max(min_size, int(0.7 * total_size))
        remaining = total_size - train_size
        val_size = max(min_size, int(0.5 * remaining))
        test_size = total_size - train_size - val_size
    
    print(f"\nSplit sizes:")
    print(f"Training: {train_size}")
    print(f"Validation: {val_size}")
    print(f"Test: {test_size}")
    
    # Create splits
    if val_size > 0 and test_size > 0:
        # Generate splits using random_split
        generator = torch.Generator().manual_seed(42)  # For reproducibility
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Print contents of each split
        print_dataset_split(dataset, train_dataset.indices, "Training")
        print_dataset_split(dataset, val_dataset.indices, "Validation")
        print_dataset_split(dataset, test_dataset.indices, "Test")
        
        val_dataloader = DataLoader(val_dataset, batch_size=2)
        test_dataloader = DataLoader(test_dataset, batch_size=2)
    else:
        train_dataset = dataset
        val_dataloader = None
        test_dataloader = None
        
        # Print all data as training data
        print_dataset_split(dataset, list(range(len(dataset))), "Training (All Data)")
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Initialize model
    model = LSTMforNER(
        vocab_size=len(word2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_labels=len(label2idx)
    )
    
    # Print label mapping
    print("\nLabel Mapping:")
    print(label2idx)
    
    # Train model
    train_model(model, train_dataloader, val_dataloader)
    
    # Create inverse label mapping for evaluation
    idx2label = {v: k for k, v in label2idx.items()}
    
    # Evaluate model
    if test_dataloader is not None:
        print("\nEvaluating on test set:")
        evaluate_model(model, test_dataloader, idx2label)
    else:
        print("\nNo test data available for evaluation")

if __name__ == "__main__":
    main()
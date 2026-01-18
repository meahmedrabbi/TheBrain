"""
Training script for TheBrain PyTorch models
Supports multiple open datasets and training configurations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import json
from tqdm import tqdm
import argparse

from pytorch_model import TransformerModel, LSTMModel


class TextDataset(Dataset):
    """Generic text dataset for training"""
    
    def __init__(self, data_pairs, vocab, max_len=128):
        """
        Initialize dataset
        
        Args:
            data_pairs: List of (input_text, output_text) tuples
            vocab: Vocabulary object
            max_len: Maximum sequence length
        """
        self.data_pairs = data_pairs
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        input_text, output_text = self.data_pairs[idx]
        
        # Tokenize and convert to indices
        src_tokens = self.vocab.encode(input_text, add_special_tokens=False)
        tgt_tokens = self.vocab.encode(output_text, add_special_tokens=True)
        
        # Pad sequences
        src_tokens = self._pad_sequence(src_tokens, self.max_len)
        tgt_tokens = self._pad_sequence(tgt_tokens, self.max_len)
        
        # Create padding masks (True for padding positions)
        src_mask = [1 if tok == 0 else 0 for tok in src_tokens]
        tgt_mask = [1 if tok == 0 else 0 for tok in tgt_tokens]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long),
            'src_mask': torch.tensor(src_mask, dtype=torch.bool),
            'tgt_mask': torch.tensor(tgt_mask, dtype=torch.bool)
        }
    
    def _pad_sequence(self, tokens, max_len):
        """Pad or truncate sequence to max_len"""
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens


class Vocabulary:
    """Vocabulary for text tokenization"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4
    
    def add_word(self, word):
        """Add word to vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
    
    def encode(self, text, add_special_tokens=False):
        """Encode text to indices"""
        tokens = text.lower().split()
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx['<START>'])
        
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx['<UNK>']))
        
        if add_special_tokens:
            indices.append(self.word2idx['<END>'])
        
        return indices
    
    def decode(self, indices):
        """Decode indices to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<END>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<START>']]:
                word = self.idx2word.get(idx, '<UNK>')
                words.append(word)
        return ' '.join(words)
    
    def save(self, filepath):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f)
    
    def load(self, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        # Convert string keys back to integers for idx2word
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']


def load_conversational_data(filepath):
    """Load conversational data from file"""
    pairs = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pairs
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line and not line.startswith('#'):
                parts = line.split('|', 1)
                if len(parts) == 2:
                    input_text, output_text = parts
                    pairs.append((input_text.strip(), output_text.strip()))
    
    print(f"Loaded {len(pairs)} pairs from {filepath}")
    return pairs


def load_dailydialog_data(filepath):
    """Load DailyDialog dataset format"""
    pairs = []
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pairs
    
    with open(filepath, 'r', encoding='utf-8') as f:
        dialogues = f.readlines()
    
    for dialogue in dialogues:
        turns = dialogue.strip().split('__eou__')
        turns = [t.strip() for t in turns if t.strip()]
        
        # Create pairs from consecutive turns
        for i in range(len(turns) - 1):
            pairs.append((turns[i], turns[i + 1]))
    
    print(f"Loaded {len(pairs)} pairs from DailyDialog format")
    return pairs


def load_cornell_movie_data(lines_file, conversations_file):
    """Load Cornell Movie Dialogs dataset"""
    pairs = []
    
    if not os.path.exists(lines_file) or not os.path.exists(conversations_file):
        print(f"Warning: Cornell Movie dataset files not found")
        return pairs
    
    # Load lines
    lines = {}
    with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 5:
                lines[parts[0]] = parts[4].strip()
    
    # Load conversations
    with open(conversations_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split(' +++$+++ ')
            if len(parts) >= 4:
                conv = eval(parts[3])
                for i in range(len(conv) - 1):
                    if conv[i] in lines and conv[i + 1] in lines:
                        pairs.append((lines[conv[i]], lines[conv[i + 1]]))
    
    print(f"Loaded {len(pairs)} pairs from Cornell Movie dataset")
    return pairs


def build_vocabulary(data_pairs, min_freq=1):
    """Build vocabulary from data pairs"""
    vocab = Vocabulary()
    word_freq = {}
    
    # Count word frequencies
    for input_text, output_text in data_pairs:
        for word in (input_text + ' ' + output_text).lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Add words that meet minimum frequency
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab.add_word(word)
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    return vocab


def train_epoch(model, dataloader, criterion, optimizer, device, model_type='transformer'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_mask = batch['src_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'transformer':
            # Transformer: use input as tgt_input (excluding last token)
            tgt_input = tgt[:, :-1]
            tgt_mask_input = tgt_mask[:, :-1]
            output = model(src, tgt_input, 
                          src_padding_mask=src_mask,
                          tgt_padding_mask=tgt_mask_input)
            
            # Target is tgt shifted by 1 (excluding first token)
            tgt_output = tgt[:, 1:]
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
        else:
            # LSTM
            output = model(src, tgt)
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, model_type='transformer'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_mask = batch['src_mask'].to(device)
            tgt_mask = batch['tgt_mask'].to(device)
            
            if model_type == 'transformer':
                tgt_input = tgt[:, :-1]
                tgt_mask_input = tgt_mask[:, :-1]
                output = model(src, tgt_input,
                             src_padding_mask=src_mask,
                             tgt_padding_mask=tgt_mask_input)
                tgt_output = tgt[:, 1:]
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
            else:
                output = model(src, tgt, teacher_forcing_ratio=0.0)
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt.reshape(-1)
            
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train TheBrain PyTorch Model')
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['transformer', 'lstm'], help='Model type')
    parser.add_argument('--data', type=str, nargs='+', 
                       default=['conversational_data.txt'],
                       help='Data files to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data from multiple sources
    print("Loading data...")
    all_pairs = []
    
    for data_file in args.data:
        if data_file.endswith('.txt') and '|' in open(data_file, 'r').read():
            pairs = load_conversational_data(data_file)
        elif 'dailydialog' in data_file.lower():
            pairs = load_dailydialog_data(data_file)
        else:
            pairs = load_conversational_data(data_file)
        all_pairs.extend(pairs)
    
    if len(all_pairs) == 0:
        print("No data loaded! Please check your data files.")
        return
    
    print(f"Total pairs loaded: {len(all_pairs)}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocabulary(all_pairs, min_freq=1)
    vocab.save(os.path.join(args.save_dir, 'vocabulary.json'))
    
    # Create datasets
    train_size = int(0.9 * len(all_pairs))
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:]
    
    train_dataset = TextDataset(train_pairs, vocab, args.max_len)
    val_dataset = TextDataset(val_pairs, vocab, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=0)
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'transformer':
        model = TransformerModel(
            vocab_size=vocab.vocab_size,
            d_model=args.d_model,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=args.d_model * 2,
            dropout=0.1,
            max_len=args.max_len
        )
    else:
        model = LSTMModel(
            vocab_size=vocab.vocab_size,
            embedding_dim=args.d_model,
            hidden_dim=args.d_model * 2,
            num_layers=2,
            dropout=0.3
        )
    
    model = model.to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=2)
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                device, args.model)
        val_loss = evaluate(model, val_loader, criterion, device, args.model)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'model_type': args.model,
                'vocab_size': vocab.vocab_size,
                'd_model': args.d_model,
                'max_len': args.max_len
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(args.save_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()

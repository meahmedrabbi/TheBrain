"""
TheBrain PyTorch Implementation
A modern transformer-based language model using PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input embeddings"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer-based language model for text generation
    Uses encoder-decoder architecture
    """
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=512):
        """
        Initialize the transformer model
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Dimension of embeddings (default: 256)
            nhead: Number of attention heads (default: 8)
            num_encoder_layers: Number of encoder layers (default: 3)
            num_decoder_layers: Number of decoder layers (default: 3)
            dim_feedforward: Dimension of feedforward network (default: 512)
            dropout: Dropout rate (default: 0.1)
            max_len: Maximum sequence length (default: 512)
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate mask for decoder to prevent attending to future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            src_padding_mask: Source padding mask (batch_size, src_len)
            tgt_padding_mask: Target padding mask (batch_size, tgt_len)
            
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Embed and add positional encoding
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_decoder(tgt)
        
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def generate(self, src, src_padding_mask, max_len=50, temperature=1.0, 
                 start_token=2, end_token=3, device='cpu'):
        """
        Generate text autoregressively
        
        Args:
            src: Source sequence (batch_size, src_len)
            src_padding_mask: Source padding mask
            max_len: Maximum length to generate
            temperature: Sampling temperature
            start_token: Start token ID
            end_token: End token ID
            device: Device to use
            
        Returns:
            Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encode source
        src_embed = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.pos_encoder(src_embed)
        
        memory = self.transformer.encoder(src_embed, src_key_padding_mask=src_padding_mask)
        
        # Initialize with start token
        tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_len):
                # Create target mask
                tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                # Embed target
                tgt_embed = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
                tgt_embed = self.pos_decoder(tgt_embed)
                
                # Decode
                output = self.transformer.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                output = self.fc_out(output)
                
                # Get next token
                next_token_logits = output[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to sequence
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Stop if end token is generated
                if (next_token == end_token).all():
                    break
        
        return tgt


class LSTMModel(nn.Module):
    """
    LSTM-based sequence-to-sequence model
    Alternative to transformer for smaller datasets
    """
    
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, 
                 num_layers=2, dropout=0.3):
        """
        Initialize LSTM model
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 256)
            hidden_dim: Hidden dimension (default: 512)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.3)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Forward pass with teacher forcing
        
        Args:
            src: Source sequence (batch_size, src_len)
            tgt: Target sequence (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        
        # Encode source
        src_embed = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.encoder_lstm(src_embed)
        
        # Decode
        outputs = torch.zeros(batch_size, tgt_len, self.vocab_size).to(src.device)
        
        # First input to decoder is start token
        input_token = tgt[:, 0].unsqueeze(1)
        
        for t in range(tgt_len):
            # Embed input
            input_embed = self.dropout(self.embedding(input_token))
            
            # Pass through decoder LSTM
            output, (hidden, cell) = self.decoder_lstm(input_embed, (hidden, cell))
            
            # Generate prediction
            prediction = self.fc_out(output.squeeze(1))
            outputs[:, t, :] = prediction
            
            # Teacher forcing: use actual next token or predicted token
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing and t < tgt_len - 1:
                input_token = tgt[:, t + 1].unsqueeze(1)
            else:
                input_token = prediction.argmax(dim=1, keepdim=True)
        
        return outputs
    
    def generate(self, src, max_len=50, start_token=2, end_token=3, device='cpu'):
        """
        Generate text autoregressively
        
        Args:
            src: Source sequence (batch_size, src_len)
            max_len: Maximum length to generate
            start_token: Start token ID
            end_token: End token ID
            device: Device to use
            
        Returns:
            Generated sequence
        """
        self.eval()
        batch_size = src.size(0)
        
        # Encode source
        src_embed = self.embedding(src)
        _, (hidden, cell) = self.encoder_lstm(src_embed)
        
        # Initialize with start token
        input_token = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        generated = [input_token]
        
        with torch.no_grad():
            for _ in range(max_len):
                # Embed input
                input_embed = self.embedding(input_token)
                
                # Pass through decoder
                output, (hidden, cell) = self.decoder_lstm(input_embed, (hidden, cell))
                
                # Get next token
                prediction = self.fc_out(output.squeeze(1))
                next_token = prediction.argmax(dim=1, keepdim=True)
                
                generated.append(next_token)
                
                # Stop if end token
                if (next_token == end_token).all():
                    break
                
                input_token = next_token
        
        return torch.cat(generated, dim=1)

"""
Inference script for TheBrain PyTorch models
Chat with the trained model
"""

import torch
import json
import argparse
from pytorch_model import TransformerModel, LSTMModel


class ChatBot:
    """Chat bot interface for trained model"""
    
    def __init__(self, model_path, vocab_path, device='auto'):
        """
        Initialize chatbot
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary file
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load vocabulary
        print("Loading vocabulary...")
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model_type = checkpoint.get('model_type', 'transformer')
        d_model = checkpoint.get('d_model', 256)
        max_len = checkpoint.get('max_len', 128)
        
        if model_type == 'transformer':
            self.model = TransformerModel(
                vocab_size=self.vocab_size,
                d_model=d_model,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                max_len=max_len
            )
        else:
            self.model = LSTMModel(
                vocab_size=self.vocab_size,
                embedding_dim=d_model,
                hidden_dim=d_model * 2,
                num_layers=2,
                dropout=0.3
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.model_type = model_type
        self.max_len = max_len
        
        print(f"✓ Model loaded ({model_type}, {d_model}D)")
    
    def encode_text(self, text):
        """Encode text to indices"""
        tokens = text.lower().split()
        indices = []
        for token in tokens:
            indices.append(self.word2idx.get(token, self.word2idx['<UNK>']))
        
        # Pad to max length
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor([indices], dtype=torch.long)
    
    def decode_indices(self, indices):
        """Decode indices to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<END>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<START>'], self.word2idx['<UNK>']]:
                word = self.idx2word.get(idx, '<UNK>')
                if word != '<UNK>':
                    words.append(word)
        return ' '.join(words) if words else "I'm not sure how to respond to that."
    
    def generate_response(self, text, max_len=50, temperature=1.0):
        """
        Generate response to input text
        
        Args:
            text: Input text
            max_len: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated response text
        """
        # Encode input
        src = self.encode_text(text).to(self.device)
        src_mask = (src == 0)
        
        # Generate response
        with torch.no_grad():
            if self.model_type == 'transformer':
                output = self.model.generate(
                    src, src_mask, max_len=max_len,
                    temperature=temperature,
                    start_token=self.word2idx['<START>'],
                    end_token=self.word2idx['<END>'],
                    device=self.device
                )
            else:
                output = self.model.generate(
                    src, max_len=max_len,
                    start_token=self.word2idx['<START>'],
                    end_token=self.word2idx['<END>'],
                    device=self.device
                )
        
        # Decode output
        response = self.decode_indices(output[0].cpu().numpy())
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "=" * 60)
        print("TheBrain PyTorch Chatbot")
        print("=" * 60)
        print("\nType 'quit', 'exit', or 'bye' to end the conversation")
        print("-" * 60)
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\nTheBrain: Goodbye! Have a great day!")
                break
            
            # Generate response
            try:
                response = self.generate_response(user_input, max_len=50, temperature=0.8)
                print(f"TheBrain: {response}")
            except Exception as e:
                print(f"TheBrain: Sorry, I encountered an error: {e}")
        
        print("\n" + "=" * 60)
        print("Chat session ended")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Chat with TheBrain PyTorch Model')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, default='models/vocabulary.json',
                       help='Path to vocabulary file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use')
    parser.add_argument('--text', type=str, help='Single text to generate response for')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = ChatBot(args.model, args.vocab, args.device)
    
    # Single text mode or interactive chat
    if args.text:
        response = chatbot.generate_response(args.text, temperature=args.temperature)
        print(f"\nInput: {args.text}")
        print(f"Response: {response}")
    else:
        chatbot.chat()


if __name__ == '__main__':
    main()

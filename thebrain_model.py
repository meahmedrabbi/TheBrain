"""
TheBrain - Super Tiny AI Model (No Heavy Libraries!)
Uses only numpy - ultra lightweight!
"""

import numpy as np
import pickle

class TheBrain:
    """
    Super simple neural network using ONLY numpy
    Pattern matching + simple learning
    """

    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32):
        """
        Initialize TheBrain - SUPER LIGHTWEIGHT

        Args:
            vocab_size: Number of unique words
            embedding_dim: Word vector size (default: 16)
            hidden_dim: Hidden layer size (default: 32)
        """
        # Word embeddings (converts words to numbers)
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01

        # Encoder weights
        self.W_encoder = np.random.randn(embedding_dim, hidden_dim) * 0.01
        self.b_encoder = np.zeros(hidden_dim)

        # Decoder weights
        self.W_decoder = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b_decoder = np.zeros(vocab_size)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def sigmoid(self, x):
        """Activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x):
        """Softmax for probabilities"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def encode(self, input_indices):
        """Encode input to hidden state"""
        # Average word embeddings
        word_vecs = self.embeddings[input_indices]
        avg_vec = np.mean(word_vecs, axis=0)

        # Pass through encoder
        hidden = self.sigmoid(np.dot(avg_vec, self.W_encoder) + self.b_encoder)
        return hidden

    def decode(self, hidden):
        """Decode hidden state to output"""
        logits = np.dot(hidden, self.W_decoder) + self.b_decoder
        probs = self.softmax(logits)
        return probs

    def forward(self, input_indices):
        """
        Forward pass

        Args:
            input_indices: Input word indices

        Returns:
            Output probabilities
        """
        hidden = self.encode(input_indices)
        output_probs = self.decode(hidden)
        return output_probs

    def train_step(self, input_indices, target_indices, learning_rate=0.01):
        """
        Single training step with backpropagation

        Args:
            input_indices: Input word indices
            target_indices: Target word indices (list)
            learning_rate: Learning rate

        Returns:
            Loss value
        """
        # Forward pass
        hidden = self.encode(input_indices)
        output_probs = self.decode(hidden)

        # Calculate loss
        loss = 0
        for target_idx in target_indices:
            loss -= np.log(output_probs[target_idx] + 1e-10)
        loss /= len(target_indices)

        # Backpropagation (simplified)
        # Output layer gradients
        d_output = output_probs.copy()
        for target_idx in target_indices:
            d_output[target_idx] -= 1.0 / len(target_indices)

        # Decoder gradients
        d_W_decoder = np.outer(hidden, d_output)
        d_b_decoder = d_output
        d_hidden = np.dot(d_output, self.W_decoder.T)

        # Encoder gradients
        d_encoder = d_hidden * hidden * (1 - hidden)
        word_vecs = self.embeddings[input_indices]
        avg_vec = np.mean(word_vecs, axis=0)
        d_W_encoder = np.outer(avg_vec, d_encoder)
        d_b_encoder = d_encoder

        # Update weights
        self.W_decoder -= learning_rate * d_W_decoder
        self.b_decoder -= learning_rate * d_b_decoder
        self.W_encoder -= learning_rate * d_W_encoder
        self.b_encoder -= learning_rate * d_b_encoder

        return loss

    def predict(self, input_indices):
        """
        Predict output for input

        Args:
            input_indices: Input word indices

        Returns:
            Predicted word indices
        """
        output_probs = self.forward(input_indices)
        predicted = np.argmax(output_probs, axis=-1)
        return predicted

    def save_model(self, path):
        """Save model to file"""
        model_data = {
            'embeddings': self.embeddings,
            'W_encoder': self.W_encoder,
            'b_encoder': self.b_encoder,
            'W_decoder': self.W_decoder,
            'b_decoder': self.b_decoder,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {path}")

    def load_model(self, path):
        """Load model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.embeddings = model_data['embeddings']
        self.W_encoder = model_data['W_encoder']
        self.b_encoder = model_data['b_encoder']
        self.W_decoder = model_data['W_decoder']
        self.b_decoder = model_data['b_decoder']
        self.vocab_size = model_data['vocab_size']
        self.embedding_dim = model_data['embedding_dim']
        self.hidden_dim = model_data['hidden_dim']
        print(f"✓ Model loaded from {path}")

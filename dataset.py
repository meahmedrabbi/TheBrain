"""
Dataset handler for TheBrain
Manages vocabulary and data processing
"""

import numpy as np
import re

class ConversationDataset:
    """
    Custom dataset for conversation pairs
    """

    def __init__(self, data_file, max_length=50):
        """
        Initialize dataset

        Args:
            data_file: Path to conversation data file
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.pairs = []

        # Special tokens
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4

        # Load and process data
        self._load_data(data_file)

    def _clean_text(self, text):
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\?\!\.\']+', '', text)
        return text

    def _load_data(self, data_file):
        """Load conversation pairs from file(s)"""
        import os

        # Support multiple data files
        data_files = []
        if isinstance(data_file, list):
            data_files = data_file
        else:
            data_files = [data_file]

        for file_path in data_files:
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found, skipping...")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line and not line.startswith('#'):
                        input_text, output_text = line.split('|', 1)
                        input_text = self._clean_text(input_text)
                        output_text = self._clean_text(output_text)

                        # Skip empty pairs
                        if not input_text or not output_text:
                            continue

                        # Add words to vocabulary
                        for word in input_text.split() + output_text.split():
                            if word not in self.word2idx:
                                self.word2idx[word] = self.vocab_size
                                self.idx2word[self.vocab_size] = word
                                self.vocab_size += 1

                        self.pairs.append((input_text, output_text))

        print(f"Loaded {len(self.pairs)} conversation pairs")
        print(f"Vocabulary size: {self.vocab_size}")

    def _text_to_indices(self, text, add_special_tokens=False):
        """Convert text to indices"""
        indices = []

        if add_special_tokens:
            indices.append(self.word2idx['<START>'])

        for word in text.split():
            indices.append(self.word2idx.get(word, self.word2idx['<UNK>']))

        if add_special_tokens:
            indices.append(self.word2idx['<END>'])

        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            indices += [self.word2idx['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return indices

    def _indices_to_text(self, indices):
        """Convert indices back to text"""
        words = []
        for idx in indices:
            if idx == self.word2idx['<END>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<START>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)

    def __len__(self):
        """Return dataset size"""
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a single data item"""
        input_text, output_text = self.pairs[idx]

        input_indices = self._text_to_indices(input_text, add_special_tokens=False)
        output_indices = self._text_to_indices(output_text, add_special_tokens=True)

        return (
            np.array(input_indices, dtype=np.int32),
            np.array(output_indices, dtype=np.int32)
        )

    def encode_text(self, text):
        """Encode input text for inference"""
        text = self._clean_text(text)
        indices = self._text_to_indices(text, add_special_tokens=False)
        # Remove padding for encoding
        indices = [i for i in indices if i != self.word2idx['<PAD>']]
        return np.array(indices, dtype=np.int32)

    def decode_text(self, indices):
        """Decode model output to text"""
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if not isinstance(indices, list):
            indices = [indices]
        return self._indices_to_text(indices)

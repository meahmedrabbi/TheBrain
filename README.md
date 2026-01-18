# TheBrain: PyTorch AI Model 🧠

A modern conversational AI model built with PyTorch, featuring Transformer and LSTM architectures, designed for easy training and deployment with support for multiple open datasets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meahmedrabbi/TheBrain/blob/main/TheBrain_PyTorch_Colab.ipynb)

## 🌟 Features

- **Transformer Architecture**: State-of-the-art attention-based model for better context understanding
- **LSTM Architecture**: Efficient recurrent model for faster training on smaller datasets
- **Multiple Dataset Support**: Train on Cornell Movie Dialogs, DailyDialog, and custom datasets
- **Google Colab Integration**: Complete notebook for cloud-based training with GPU support
- **Easy Inference**: Simple API for generating responses
- **Flexible Training**: Customizable hyperparameters and training configurations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Google Colab Guide](#google-colab-guide)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Tips & Best Practices](#tips--best-practices)

## 🚀 Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/meahmedrabbi/TheBrain.git
cd TheBrain

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm

## ⚡ Quick Start

### 1. Train a Model

```bash
# Train with transformer architecture on included data
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir models
```

### 2. Chat with Your Model

```bash
# Interactive chat
python inference_pytorch.py \
    --model models/best_model.pt \
    --vocab models/vocabulary.json

# Single text inference
python inference_pytorch.py \
    --model models/best_model.pt \
    --vocab models/vocabulary.json \
    --text "hello" \
    --temperature 0.8
```

## 🎓 Training

### Training Options

#### Transformer Model (Recommended)

```bash
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/additional_data.txt \
    --epochs 20 \
    --batch_size 32 \
    --lr 0.001 \
    --d_model 256 \
    --max_len 128 \
    --save_dir models
```

#### LSTM Model (Faster, Lighter)

```bash
python train_pytorch.py \
    --model lstm \
    --data conversational_data.txt \
    --epochs 15 \
    --batch_size 64 \
    --lr 0.001 \
    --d_model 256 \
    --save_dir models
```

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `transformer` | Model type: `transformer` or `lstm` |
| `--data` | list | `conversational_data.txt` | List of data files |
| `--epochs` | int | `10` | Number of training epochs |
| `--batch_size` | int | `32` | Batch size for training |
| `--lr` | float | `0.001` | Learning rate |
| `--d_model` | int | `256` | Model dimension |
| `--max_len` | int | `128` | Maximum sequence length |
| `--save_dir` | str | `models` | Directory to save models |
| `--device` | str | `auto` | Device: `auto`, `cpu`, `cuda`, `mps` |

## 💬 Inference

### Interactive Chat

```python
from inference_pytorch import ChatBot

# Initialize chatbot
chatbot = ChatBot(
    model_path='models/best_model.pt',
    vocab_path='models/vocabulary.json',
    device='auto'
)

# Generate response
response = chatbot.generate_response("Hello!", temperature=0.8)
print(response)

# Start interactive chat
chatbot.chat()
```

### Inference Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `models/best_model.pt` | Path to model checkpoint |
| `--vocab` | str | `models/vocabulary.json` | Path to vocabulary file |
| `--device` | str | `auto` | Device to use |
| `--text` | str | None | Single text for inference |
| `--temperature` | float | `0.8` | Sampling temperature (0.5-1.5) |

## 📊 Google Colab Guide

We provide a comprehensive Jupyter notebook for training in Google Colab with free GPU support!

### Steps:

1. **Open the notebook**: Click the "Open in Colab" badge above
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Run all cells**: The notebook will guide you through:
   - Installation
   - Dataset preparation
   - Model training
   - Testing and inference
   - Downloading your trained model

### What's Included in the Notebook:

- ✅ Automatic environment setup
- ✅ Multiple dataset downloads (Cornell Movie, DailyDialog)
- ✅ Step-by-step training guide
- ✅ Interactive testing
- ✅ Model visualization
- ✅ Download trained models
- ✅ Tips and troubleshooting

## 📁 Datasets

### Included Datasets

1. **conversational_data.txt**: Basic conversational pairs included in the repo

### Supported External Datasets

1. **Cornell Movie Dialogs Corpus**
   - 220,000+ conversational exchanges
   - Movie character conversations
   - Download: http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html

2. **DailyDialog**
   - 13,000+ multi-turn dialogues
   - Daily communication topics
   - Download: http://yanran.li/dailydialog

3. **Custom Format**: Create your own dataset with format:
   ```
   input text|output text
   how are you|I'm doing great!
   what is your name|I'm TheBrain
   ```

### Adding Custom Datasets

```python
# Create custom dataset file
custom_data = [
    "question 1|answer 1",
    "question 2|answer 2",
    "question 3|answer 3"
]

with open('my_custom_data.txt', 'w') as f:
    for pair in custom_data:
        f.write(pair + '\n')

# Train with your custom data
python train_pytorch.py --data conversational_data.txt my_custom_data.txt
```

## 🏗️ Model Architecture

### Transformer Model

```
- Encoder-Decoder Architecture
- Multi-head Attention (8 heads)
- 3 Encoder Layers
- 3 Decoder Layers
- Positional Encoding
- Feed-forward Networks (512D)
- Dropout (0.1)
```

**Advantages**:
- Better long-range dependencies
- Parallel processing
- State-of-the-art performance

### LSTM Model

```
- Sequence-to-Sequence Architecture
- 2 LSTM Layers
- Encoder-Decoder with attention
- Teacher Forcing
- Dropout (0.3)
```

**Advantages**:
- Faster training
- Lower memory usage
- Good for smaller datasets

## 📖 API Reference

### TransformerModel

```python
from pytorch_model import TransformerModel

model = TransformerModel(
    vocab_size=10000,      # Vocabulary size
    d_model=256,           # Model dimension
    nhead=8,               # Number of attention heads
    num_encoder_layers=3,  # Encoder layers
    num_decoder_layers=3,  # Decoder layers
    dim_feedforward=512,   # FFN dimension
    dropout=0.1,           # Dropout rate
    max_len=512            # Max sequence length
)

# Generate text
output = model.generate(
    src,                   # Input tensor
    src_padding_mask,      # Padding mask
    max_len=50,           # Max generation length
    temperature=1.0,      # Sampling temperature
    start_token=2,        # Start token ID
    end_token=3,          # End token ID
    device='cuda'         # Device
)
```

### LSTMModel

```python
from pytorch_model import LSTMModel

model = LSTMModel(
    vocab_size=10000,     # Vocabulary size
    embedding_dim=256,    # Embedding dimension
    hidden_dim=512,       # Hidden dimension
    num_layers=2,         # Number of LSTM layers
    dropout=0.3           # Dropout rate
)

# Generate text
output = model.generate(
    src,                  # Input tensor
    max_len=50,          # Max generation length
    start_token=2,       # Start token ID
    end_token=3,         # End token ID
    device='cuda'        # Device
)
```

### ChatBot

```python
from inference_pytorch import ChatBot

chatbot = ChatBot(
    model_path='models/best_model.pt',
    vocab_path='models/vocabulary.json',
    device='auto'
)

# Generate response
response = chatbot.generate_response(
    text="Hello!",
    max_len=50,
    temperature=0.8
)

# Interactive chat
chatbot.chat()
```

## 💡 Tips & Best Practices

### Training Tips

1. **Start Small**: Begin with a small model and less data to verify everything works
2. **Use GPU**: Training is 10-50x faster on GPU (use Google Colab if you don't have one)
3. **Monitor Loss**: Watch training and validation loss - they should both decrease
4. **More Data**: More diverse training data generally leads to better results
5. **Hyperparameter Tuning**:
   - Learning rate: Try 0.0001 to 0.001
   - Batch size: 16-64 depending on GPU memory
   - Model dimension: 256-512 for good balance

### Inference Tips

1. **Temperature Control**:
   - `0.5-0.7`: More focused and deterministic responses
   - `0.8-0.9`: Balanced creativity (recommended)
   - `1.0-1.5`: More random and creative

2. **Context Length**: Longer inputs provide more context but slower inference

3. **Batch Inference**: Process multiple inputs together for efficiency

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch_size` or `--d_model` |
| Poor responses | Add more training data or train longer |
| Repetitive outputs | Adjust temperature or implement beam search |
| Slow training | Use GPU or reduce model size |
| Model not learning | Check learning rate, may need adjustment |

## 📈 Performance Benchmarks

Approximate training times on different hardware:

| Hardware | Model | Dataset Size | Time/Epoch |
|----------|-------|--------------|------------|
| CPU (8 cores) | Transformer | 5K pairs | ~15 min |
| CPU (8 cores) | LSTM | 5K pairs | ~8 min |
| GPU (T4) | Transformer | 5K pairs | ~2 min |
| GPU (T4) | LSTM | 5K pairs | ~1 min |
| GPU (V100) | Transformer | 50K pairs | ~8 min |

## 🔬 Advanced Usage

### Custom Training Loop

```python
from pytorch_model import TransformerModel
from train_pytorch import Vocabulary, TextDataset
import torch
from torch.utils.data import DataLoader

# Load data
vocab = Vocabulary()
# ... build vocabulary ...

# Create dataset
dataset = TextDataset(pairs, vocab, max_len=128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = TransformerModel(vocab_size=vocab.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        # Your custom training logic here
        pass
```

### Model Export

```python
# Export to TorchScript for deployment
model.eval()
example_input = torch.randint(0, vocab_size, (1, 128))
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Cornell Movie Dialogs Corpus
- DailyDialog Dataset
- PyTorch Team
- Hugging Face Transformers (inspiration)

## 📞 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🚀**

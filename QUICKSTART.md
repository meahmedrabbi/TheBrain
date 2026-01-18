# Quick Start Guide for TheBrain PyTorch AI Model

This guide will help you get started with training and using TheBrain in just a few minutes!

## Option 1: Use Google Colab (Recommended - Free GPU!)

The easiest way to get started is using Google Colab:

1. **Open the notebook**: Click the badge below
   
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meahmedrabbi/TheBrain/blob/main/TheBrain_PyTorch_Colab.ipynb)

2. **Enable GPU**: In Colab, go to `Runtime` → `Change runtime type` → Select `GPU`

3. **Run all cells**: Click `Runtime` → `Run all` or run cells one by one

4. **Download your trained model**: The notebook will guide you through downloading your model

That's it! The notebook includes everything you need.

## Option 2: Local Installation (5 Minutes)

### Step 1: Install Dependencies

```bash
git clone https://github.com/meahmedrabbi/TheBrain.git
cd TheBrain
pip install -r requirements.txt
```

### Step 2: Prepare Data

```bash
# Create sample datasets for testing
python download_datasets.py --samples

# Or download real movie dialog data (takes longer)
python download_datasets.py --cornell
```

### Step 3: Train Your Model

**Quick test (2 minutes on CPU)**:
```bash
python train_pytorch.py \
    --model lstm \
    --data conversational_data.txt data/combined_conversations.txt \
    --epochs 5 \
    --batch_size 32
```

**Better quality (10-20 minutes on CPU, 2-3 minutes on GPU)**:
```bash
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/cornell_conversations.txt \
    --epochs 20 \
    --batch_size 32 \
    --d_model 256
```

### Step 4: Chat with Your Model

```bash
# Interactive chat
python inference_pytorch.py

# Single query
python inference_pytorch.py --text "hello, how are you?"
```

## Common Commands Cheat Sheet

```bash
# Download sample datasets
python download_datasets.py --samples

# Download Cornell Movie dataset
python download_datasets.py --cornell

# Train LSTM model (faster)
python train_pytorch.py --model lstm --epochs 10

# Train Transformer model (better quality)
python train_pytorch.py --model transformer --epochs 20

# Train with multiple datasets
python train_pytorch.py --data file1.txt file2.txt file3.txt

# Use GPU (if available)
python train_pytorch.py --device cuda

# Chat with trained model
python inference_pytorch.py

# Get single response
python inference_pytorch.py --text "your question here"
```

## Tips for Best Results

1. **More Data = Better Results**: Combine multiple datasets
   ```bash
   python train_pytorch.py --data conversational_data.txt data/*.txt
   ```

2. **Use GPU**: 10-50x faster than CPU
   - Local: Add `--device cuda`
   - Google Colab: Enable GPU runtime

3. **Train Longer**: More epochs = better quality
   ```bash
   python train_pytorch.py --epochs 50
   ```

4. **Larger Model**: Better understanding but slower
   ```bash
   python train_pytorch.py --d_model 512
   ```

## Troubleshooting

**Problem**: Out of memory
```bash
# Solution: Reduce batch size or model size
python train_pytorch.py --batch_size 16 --d_model 128
```

**Problem**: Poor responses
```bash
# Solution: Train longer with more data
python train_pytorch.py --epochs 30 --data file1.txt file2.txt
```

**Problem**: Training too slow
```bash
# Solution: Use LSTM or smaller model
python train_pytorch.py --model lstm --d_model 128
```

## What's Next?

1. **Read the full README**: More detailed documentation
2. **Try the Colab notebook**: Best for beginners
3. **Experiment**: Try different hyperparameters
4. **Add your own data**: Create custom dataset files
5. **Deploy**: Integrate into your application

## Need Help?

- Check the [full README](README.md) for detailed documentation
- Open an issue on GitHub
- See the [Google Colab notebook](TheBrain_PyTorch_Colab.ipynb) for examples

---

**Happy Training! 🚀**

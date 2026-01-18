# Example training configurations for different use cases
# Copy and modify these commands for your needs

# =============================================================================
# QUICK TESTING (Fast, for verifying everything works)
# =============================================================================

# Minimal training - completes in 1-2 minutes on CPU
python train_pytorch.py \
    --model lstm \
    --data conversational_data.txt \
    --epochs 3 \
    --batch_size 16 \
    --d_model 64 \
    --max_len 64 \
    --save_dir models/quick_test

# =============================================================================
# SMALL MODEL (Good for limited hardware, still decent quality)
# =============================================================================

# LSTM - Fast training, low memory
python train_pytorch.py \
    --model lstm \
    --data conversational_data.txt data/combined_conversations.txt \
    --epochs 15 \
    --batch_size 32 \
    --d_model 128 \
    --max_len 100 \
    --lr 0.001 \
    --save_dir models/small_lstm

# Transformer - Better quality, slightly slower
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/combined_conversations.txt \
    --epochs 10 \
    --batch_size 16 \
    --d_model 128 \
    --max_len 100 \
    --lr 0.0005 \
    --save_dir models/small_transformer

# =============================================================================
# MEDIUM MODEL (Balanced - Recommended for most users)
# =============================================================================

# LSTM - Good balance of speed and quality
python train_pytorch.py \
    --model lstm \
    --data conversational_data.txt data/*.txt \
    --epochs 25 \
    --batch_size 32 \
    --d_model 256 \
    --max_len 128 \
    --lr 0.001 \
    --save_dir models/medium_lstm

# Transformer - Recommended for best results
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/*.txt \
    --epochs 20 \
    --batch_size 32 \
    --d_model 256 \
    --max_len 128 \
    --lr 0.0005 \
    --save_dir models/medium_transformer

# =============================================================================
# LARGE MODEL (Best quality, requires GPU and lots of data)
# =============================================================================

# Transformer - High quality, needs GPU
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/*.txt \
    --epochs 50 \
    --batch_size 64 \
    --d_model 512 \
    --max_len 256 \
    --lr 0.0003 \
    --device cuda \
    --save_dir models/large_transformer

# =============================================================================
# DOMAIN-SPECIFIC (Train on specific topics)
# =============================================================================

# Technical/AI Q&A
python train_pytorch.py \
    --model transformer \
    --data data/technical_qa.txt \
    --epochs 30 \
    --batch_size 16 \
    --d_model 256 \
    --save_dir models/technical_bot

# Customer Service
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/casual_chats.txt \
    --epochs 25 \
    --batch_size 32 \
    --d_model 256 \
    --save_dir models/customer_service

# =============================================================================
# GOOGLE COLAB (Optimized for free GPU)
# =============================================================================

# Maximize free Colab GPU usage
python train_pytorch.py \
    --model transformer \
    --data conversational_data.txt data/*.txt \
    --epochs 30 \
    --batch_size 64 \
    --d_model 384 \
    --max_len 192 \
    --lr 0.0005 \
    --device cuda \
    --save_dir models/colab_model

# =============================================================================
# INFERENCE EXAMPLES
# =============================================================================

# Interactive chat
python inference_pytorch.py \
    --model models/medium_transformer/best_model.pt \
    --vocab models/medium_transformer/vocabulary.json

# Single query with higher creativity
python inference_pytorch.py \
    --model models/medium_transformer/best_model.pt \
    --vocab models/medium_transformer/vocabulary.json \
    --text "tell me about artificial intelligence" \
    --temperature 1.0

# Single query with more focused response
python inference_pytorch.py \
    --model models/medium_transformer/best_model.pt \
    --vocab models/medium_transformer/vocabulary.json \
    --text "what is machine learning" \
    --temperature 0.6

# =============================================================================
# NOTES
# =============================================================================

# Hyperparameter Guidelines:
# - epochs: 10-50 (more = better, but diminishing returns)
# - batch_size: 16-64 (larger = faster but needs more memory)
# - d_model: 128-512 (larger = better but slower)
# - max_len: 64-256 (match your typical conversation length)
# - lr: 0.0001-0.001 (transformer usually needs lower)
# - temperature: 0.5-1.5 (lower = focused, higher = creative)

# Memory Usage Estimates:
# - d_model=128, batch=32: ~2GB RAM/VRAM
# - d_model=256, batch=32: ~4GB RAM/VRAM
# - d_model=512, batch=64: ~8GB RAM/VRAM

# Training Time Estimates (per epoch):
# CPU (8 cores), 5K pairs:
#   - LSTM (d_model=256): ~8 min
#   - Transformer (d_model=256): ~15 min
# GPU (T4), 5K pairs:
#   - LSTM (d_model=256): ~1 min
#   - Transformer (d_model=256): ~2 min

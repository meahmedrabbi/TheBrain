"""
Dataset downloader and preparation script
Downloads and prepares multiple open datasets for training
"""

import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_cornell_movie_dialogs(data_dir='data'):
    """Download and prepare Cornell Movie Dialogs dataset"""
    print("\n" + "="*60)
    print("Downloading Cornell Movie Dialogs Corpus...")
    print("="*60)
    
    os.makedirs(data_dir, exist_ok=True)
    
    url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zip_path = os.path.join(data_dir, "cornell_movie_dialogs.zip")
    
    # Download
    try:
        download_url(url, zip_path)
    except Exception as e:
        print(f"Error downloading: {e}")
        return False
    
    # Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Extracted successfully")
    except Exception as e:
        print(f"Error extracting: {e}")
        return False
    
    # Prepare data
    print("Preparing conversational pairs...")
    try:
        cornell_dir = os.path.join(data_dir, "cornell movie-dialogs corpus")
        lines_file = os.path.join(cornell_dir, "movie_lines.txt")
        conversations_file = os.path.join(cornell_dir, "movie_conversations.txt")
        output_file = os.path.join(data_dir, "cornell_conversations.txt")
        
        # Load lines
        lines = {}
        with open(lines_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) >= 5:
                    lines[parts[0]] = parts[4].strip()
        
        # Create conversation pairs
        pairs = []
        with open(conversations_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(' +++$+++ ')
                if len(parts) >= 4:
                    conv = eval(parts[3])
                    for i in range(len(conv) - 1):
                        if conv[i] in lines and conv[i + 1] in lines:
                            input_text = lines[conv[i]].replace('|', ' ')
                            output_text = lines[conv[i + 1]].replace('|', ' ')
                            # Clean text
                            input_text = ' '.join(input_text.split())
                            output_text = ' '.join(output_text.split())
                            if input_text and output_text:
                                pairs.append(f"{input_text}|{output_text}")
        
        # Save (limit to 10000 for reasonable training time)
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in pairs[:10000]:
                f.write(pair + '\n')
        
        print(f"✓ Prepared {min(len(pairs), 10000)} conversation pairs")
        print(f"✓ Saved to: {output_file}")
        
        # Cleanup
        os.remove(zip_path)
        
        return True
    except Exception as e:
        print(f"Error preparing data: {e}")
        return False


def create_sample_datasets(data_dir='data'):
    """Create sample datasets for quick testing"""
    print("\n" + "="*60)
    print("Creating sample datasets...")
    print("="*60)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample daily conversations
    daily_dialogs = [
        "Good morning!|Good morning! How are you today?",
        "I'm doing great, thanks!|That's wonderful to hear!",
        "What's the weather like?|It's sunny and warm today.",
        "Do you like pizza?|Yes, I love pizza! It's my favorite food.",
        "What are your hobbies?|I enjoy reading, hiking, and playing music.",
        "Have you seen the new movie?|Yes, I watched it last weekend. It was amazing!",
        "What's your favorite color?|I like blue. How about you?",
        "I prefer green.|Green is a beautiful color!",
        "What do you do for work?|I'm an AI assistant, helping people with various tasks.",
        "That sounds interesting!|Thank you! I enjoy helping people.",
        "How was your day?|It was quite productive, thank you for asking!",
        "I'm glad to hear that.|How can I help you today?",
        "Can you recommend a book?|I'd recommend '1984' by George Orwell. It's a classic.",
        "Thanks for the suggestion!|You're welcome! Enjoy the read!",
        "What's for dinner?|How about pasta with marinara sauce?",
        "That sounds delicious!|Great! It's easy to make too.",
        "Tell me a joke.|Why don't scientists trust atoms? Because they make up everything!",
        "That's funny!|I'm glad you enjoyed it!",
        "What time is it?|I don't have access to the current time, but you can check your device.",
        "Fair enough.|Is there anything else I can help you with?",
    ]
    
    # Technical Q&A
    technical_qa = [
        "What is artificial intelligence?|Artificial intelligence is the simulation of human intelligence by machines and computer systems.",
        "What is machine learning?|Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed.",
        "What is deep learning?|Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns.",
        "What is PyTorch?|PyTorch is an open-source machine learning framework developed by Facebook's AI Research lab.",
        "What is a neural network?|A neural network is a computing system inspired by biological neural networks that process information using interconnected nodes.",
        "What is a transformer?|A transformer is a neural network architecture that uses self-attention mechanisms to process sequential data.",
        "What is NLP?|NLP stands for Natural Language Processing, which focuses on the interaction between computers and human language.",
        "What is computer vision?|Computer vision is a field of AI that trains computers to interpret and understand visual information from the world.",
        "What is reinforcement learning?|Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment.",
        "What is supervised learning?|Supervised learning is a machine learning approach where models are trained on labeled data.",
    ]
    
    # Casual conversations
    casual_chats = [
        "Hey there!|Hello! How can I help you today?",
        "How's it going?|Everything's going well! How about you?",
        "What's up?|Not much! Just here to help. What's on your mind?",
        "Nice to meet you.|Nice to meet you too!",
        "Thanks for your help.|You're very welcome! Happy to assist!",
        "Goodbye!|Goodbye! Have a great day!",
        "See you later.|See you! Take care!",
        "Have a nice day.|Thank you! You too!",
        "You're awesome!|Thank you so much! That's very kind of you.",
        "I appreciate your help.|I'm glad I could help you!",
    ]
    
    # Save datasets
    datasets = {
        'daily_conversations.txt': daily_dialogs,
        'technical_qa.txt': technical_qa,
        'casual_chats.txt': casual_chats
    }
    
    for filename, data in datasets.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in data:
                f.write(pair + '\n')
        print(f"✓ Created {filename} with {len(data)} pairs")
    
    # Create combined dataset
    combined_file = os.path.join(data_dir, 'combined_conversations.txt')
    with open(combined_file, 'w', encoding='utf-8') as f:
        for data in datasets.values():
            for pair in data:
                f.write(pair + '\n')
    
    print(f"✓ Created combined dataset with {sum(len(d) for d in datasets.values())} pairs")
    print(f"✓ All datasets saved to: {data_dir}/")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets for TheBrain')
    parser.add_argument('--cornell', action='store_true', 
                       help='Download Cornell Movie Dialogs dataset')
    parser.add_argument('--samples', action='store_true',
                       help='Create sample datasets')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to save datasets')
    
    args = parser.parse_args()
    
    if not any([args.cornell, args.samples, args.all]):
        print("Please specify at least one dataset option:")
        print("  --cornell: Download Cornell Movie Dialogs")
        print("  --samples: Create sample datasets")
        print("  --all: Do everything")
        print("\nExample: python download_datasets.py --samples")
        return
    
    print("\nTheBrain Dataset Downloader")
    print("="*60)
    
    success = True
    
    if args.samples or args.all:
        success = create_sample_datasets(args.data_dir) and success
    
    if args.cornell or args.all:
        success = download_cornell_movie_dialogs(args.data_dir) and success
    
    print("\n" + "="*60)
    if success:
        print("✓ All datasets prepared successfully!")
        print(f"\nDatasets saved to: {args.data_dir}/")
        print("\nYou can now train with:")
        print(f"  python train_pytorch.py --data {args.data_dir}/*.txt")
    else:
        print("⚠ Some datasets failed to download/prepare")
        print("Please check the errors above")
    print("="*60)


if __name__ == '__main__':
    main()

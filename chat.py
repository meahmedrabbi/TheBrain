import numpy as np
import pickle
from thebrain_model import TheBrain
from dataset import ConversationDataset
import os

def find_best_response(user_input, dataset):
    """
    Simple pattern matching fallback
    Finds best matching response from training data
    """
    user_words = set(user_input.lower().split())
    best_match = None
    best_score = 0

    for input_text, output_text in dataset.pairs:
        input_words = set(input_text.lower().split())
        # Calculate word overlap
        overlap = len(user_words & input_words)
        if overlap > best_score:
            best_score = overlap
            best_match = output_text

    return best_match if best_match else "I'm still learning! Can you ask something else?"

def chat_with_brain():
    """
    Interactive chat session with TheBrain
    """
    print("=" * 50)
    print("Chat with TheBrain AI (Numpy Only!)")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists('thebrain_model.pkl'):
        print("\nError: Model not found!")
        print("Please train the model first by running: python train.py")
        return

    print(f"\nLoading model...")

    # Load vocabulary
    with open('vocabulary.pkl', 'rb') as f:
        vocab_data = pickle.load(f)

    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    vocab_size = vocab_data['vocab_size']

    # Load model
    model = TheBrain(vocab_size=vocab_size, embedding_dim=16, hidden_dim=32)
    model.load_model('thebrain_model.pkl')

    # Create dataset helper for text processing
    dataset = ConversationDataset(['conversational_data.txt', 'company_data_CUSTOMIZE_THIS.txt'])
    dataset.word2idx = word2idx
    dataset.idx2word = idx2word
    dataset.vocab_size = vocab_size

    print("\n✓ TheBrain is ready to chat!")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation")
    print("-" * 50)

    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\nTheBrain: Goodbye! Have a great day!")
            break

        # Process input and generate response
        try:
            # Encode input
            input_indices = dataset.encode_text(user_input)

            if len(input_indices) == 0:
                print("TheBrain: I didn't quite catch that. Can you rephrase?")
                continue

            # Generate response using neural network
            predicted_indices = model.predict(input_indices)

            # Decode response
            response_words = []
            for idx in predicted_indices[:10]:  # Max 10 words
                if idx in [0, 2, 3]:  # Skip special tokens
                    continue
                if idx in idx2word:
                    response_words.append(idx2word[idx])

            if response_words:
                response = ' '.join(response_words)
            else:
                # Fallback to pattern matching
                response = find_best_response(user_input, dataset)

            print(f"TheBrain: {response}")

        except Exception as e:
            # Fallback to pattern matching if neural net fails
            response = find_best_response(user_input, dataset)
            print(f"TheBrain: {response}")

    print("\n" + "=" * 50)
    print("Chat session ended")
    print("=" * 50)

if __name__ == "__main__":
    chat_with_brain()

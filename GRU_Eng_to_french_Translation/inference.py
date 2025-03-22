import torch
import numpy as np
import re
import docx2txt
from model import create_translation_model  # Import the model creation function
from dataloader import TranslationDataLoader, EnglishFrenchDataset

def inference_indices_to_sentence(indices, idx2word):
    """
    Convert a sequence of indices back to a sentence - specialized for inference
    
    Args:
        indices: List of indices (could be tensors)
        idx2word: Index to word mapping
        
    Returns:
        Reconstructed sentence
    """
    # Filter out padding, SOS, and EOS tokens
    words = []
    for idx in indices:
        # Convert tensor to integer if needed
        if torch.is_tensor(idx):
            idx = idx.item()
            
        if idx == 0:  # <PAD>
            continue
        if idx == 1:  # < SOS >
            continue
        if idx == 2:  # <EOS>
            break
            
        # Skip if idx is not in the vocabulary
        if idx not in idx2word:
            continue
            
        words.append(idx2word[idx])
    
    return ' '.join(words)

def load_model_for_inference(model_path, data, device):
    """
    Load a trained model for inference
    
    Args:
        model_path: Path to the saved model checkpoint
        data: Dictionary containing model configuration data
        device: Device to run the model on
        
    Returns:
        model: Loaded model ready for inference
    """
    # Create a new model with the same configuration
    model = create_translation_model(
        english_vocab_size=data['english_vocab_size'],
        french_vocab_size=data['french_vocab_size'],
        embedding_size=256,  # Use the same values from training
        hidden_size=512,
        num_layers=2,
        dropout=0.0,  # Set to 0 for inference
        device=device
    )
    
    # Load the saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    
    # Print model information
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Model accuracy: {checkpoint.get('accuracy', 'N/A'):.4f}")
    
    return model

def preprocess_input_sentence(sentence, english_word2idx, max_length):
    """
    Preprocess an input English sentence for translation
    
    Args:
        sentence: Input English sentence
        english_word2idx: Word to index mapping for English
        max_length: Maximum sequence length
        
    Returns:
        tensor: Tensor of indices ready for the model
    """
    # Split the sentence into words
    words = sentence.split()
    
    # Convert words to indices
    indices = [english_word2idx.get(word, english_word2idx.get('< SOS >', 1)) for word in words]
    
    # Add EOS token
    indices = indices + [english_word2idx.get('<EOS>', 2)]
    
    # Pad the sequence
    padded_indices = indices + [english_word2idx.get('<PAD>', 0)] * (max_length - len(indices))
    
    # Convert to tensor
    tensor = torch.tensor(padded_indices, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    return tensor

def translate_sentence(model, sentence, data, device):
    """
    Translate an English sentence to French
    
    Args:
        model: Trained translation model
        sentence: Input English sentence
        data: Dictionary containing vocabulary and configuration data
        device: Device to run the model on
        
    Returns:
        translation: Translated French sentence
    """
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess the input sentence
    src_tensor = preprocess_input_sentence(
        sentence, 
        data['english_word2idx'], 
        data['max_english_length']
    ).to(device)
    
    # Get SOS and EOS token indices
    sos_idx = data['french_word2idx']['< SOS >']
    eos_idx = data['french_word2idx']['<EOS>']
    
    # Translate the sentence
    with torch.no_grad():
        translation_tensor = model.translate(
            src_tensor, 
            max_len=data['max_french_length'], 
            sos_idx=sos_idx, 
            eos_idx=eos_idx
        )
    
    # Convert indices to words
    # First move to CPU and convert to regular Python list/integers
    translation_indices = translation_tensor[0].cpu().tolist()
    translation = inference_indices_to_sentence(translation_indices, data['french_idx2word'])
    
    return translation

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # File paths
    DATA_PATH = "/data1/Sadia/Hamed_GRU/Dataset _English_to _French.docx"  # Update with your actual file path
    MODEL_PATH = "/data1/Sadia/Hamed_GRU/models/best_model.pt"  # Update with your model path
    
    # Load data
    translation_loader = TranslationDataLoader(batch_size=1)  # Batch size 1 for inference
    data = translation_loader.prepare_data(file_path=DATA_PATH)
    
    # Load the model
    model = load_model_for_inference(MODEL_PATH, data, device)
    
    # Test sentences
    test_sentences = [
        "I love learning languages",
        "How are you today",
        "The weather is beautiful",
        "Thank you for your help",
        "I want to go to Paris"
    ]
    
    # Test with sentences from the dataset
    print("\nTesting with sentences from the dataset:")
    for i in range(min(5, len(data['all_english_sentences']))):
        english = data['all_english_sentences'][i]
        actual_french = data['all_french_sentences'][i]
        
        translated_french = translate_sentence(model, english, data, device)
        
        print(f"\nEnglish: {english}")
        print(f"Generated French: {translated_french}")
        print(f"Actual French: {actual_french}")
    
    # Test with new sentences
    print("\nTesting with new sentences:")
    for sentence in test_sentences:
        translated = translate_sentence(model, sentence, data, device)
        print(f"\nEnglish: {sentence}")
        print(f"French: {translated}")
    
    # Interactive mode
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        sentence = input("\nEnter an English sentence: ")
        if sentence.lower() == 'exit':
            break
        
        translated = translate_sentence(model, sentence, data, device)
        print(f"French: {translated}")

if __name__ == "__main__":
    main()
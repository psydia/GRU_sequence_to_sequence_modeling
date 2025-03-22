import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your model and data loader classes
from dataloader import TranslationDataLoader, FrenchEnglishDataset
from model_attention import create_translation_model, indices_to_sentence

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True


def train_translation_model(data_path, output_dir, batch_size=32, embedding_size=256, 
                           hidden_size=512, num_layers=2, dropout=0.3, 
                           learning_rate=0.001, num_epochs=200, save_interval=10,
                           device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the translation model and track metrics
    
    Args:
        data_path: Path to the dataset file
        output_dir: Directory to save model checkpoints and metrics
        batch_size: Batch size for training
        embedding_size: Size of word embeddings
        hidden_size: Hidden size for GRU units
        num_layers: Number of GRU layers
        dropout: Dropout probability
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        save_interval: Interval to save metrics to CSV
        device: Device to train on ('cuda' or 'cpu')
    """
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data loader and prepare data
    print("Loading and preparing data...")
    translation_loader = TranslationDataLoader(batch_size=batch_size)
    data = translation_loader.prepare_data(file_path=data_path)
    
    # Get vocabulary sizes and other info from prepared data
    french_vocab_size = data['french_vocab_size']
    english_vocab_size = data['english_vocab_size']
    french_word2idx = data['french_word2idx']
    french_idx2word = data['french_idx2word']
    english_word2idx = data['english_word2idx']
    english_idx2word = data['english_idx2word']
    max_english_length = data['max_english_length']
    
    # Split dataset into training and validation sets (90/10 split)
    full_dataset = translation_loader.full_dataset
    dataset_size = len(full_dataset)
    val_size = int(0.1 * dataset_size)  # 10% for validation
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Training dataset size: {train_size}")
    print(f"Validation dataset size: {val_size}")
    
    # Create the translation model
    model = create_translation_model(
        french_vocab_size=french_vocab_size,
        english_vocab_size=english_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
        pad_idx=french_word2idx['<PAD>']
    )
    
    # Initialize optimizer and criterion (loss function)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Ignore padding tokens in loss calculation
    PAD_IDX = english_word2idx['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Dictionaries and lists to store metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Variable for tracking best model
    best_val_accuracy = 0.0
    
    print("Beginning training...")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # Training
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for batch in progress_bar:
            # Get inputs and outputs
            src = batch['encoder_input'].to(device)
            trg = batch['decoder_input'].to(device)
            tgt_output = batch['decoder_output'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(src, trg)
            
            # Calculate loss (ignore first token, which is SOS)
            # Reshape outputs for CrossEntropyLoss
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt_output = tgt_output[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt_output)
            
            # Backward pass and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating (Epoch {epoch})"):
                src = batch['encoder_input'].to(device)
                trg = batch['decoder_input'].to(device)
                tgt_output = batch['decoder_output'].to(device)
                
                # Forward pass
                output, _ = model(src, trg, teacher_forcing_ratio=0.0)  # No teacher forcing during validation
                
                # Calculate loss
                output_dim = output.shape[-1]
                output_reshaped = output[:, 1:].contiguous().view(-1, output_dim)
                tgt_output_reshaped = tgt_output[:, 1:].contiguous().view(-1)
                
                loss = criterion(output_reshaped, tgt_output_reshaped)
                val_loss += loss.item()
                
                # Calculate accuracy (ignoring padding tokens)
                predictions = output[:, 1:].argmax(2)
                targets = tgt_output[:, 1:]
                mask = (targets != PAD_IDX)
                correct = ((predictions == targets) * mask).sum().item()
                non_pad_tokens = mask.sum().item()
                
                total_correct += correct
                total_tokens += non_pad_tokens
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"Epoch: {epoch}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
        
        # Store metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        # Save metrics to separate files at specified intervals
        if epoch % save_interval == 0 or epoch == num_epochs:
            # Save metrics to a CSV file
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(f"{output_dir}/training_metrics_french_to_english.csv", index=False)
            
            # Save the current epoch's metrics to separate files
            with open(f"{output_dir}/train_loss_epoch_{epoch}.txt", 'w') as f:
                f.write(f"{train_loss}")
                
            with open(f"{output_dir}/val_loss_epoch_{epoch}.txt", 'w') as f:
                f.write(f"{val_loss}")
                
            with open(f"{output_dir}/val_accuracy_epoch_{epoch}.txt", 'w') as f:
                f.write(f"{val_accuracy}")
                
            print(f"Metrics saved for epoch {epoch}")
        
        # Check if this is the best model so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Save the best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'french_word2idx': french_word2idx,
                'french_idx2word': french_idx2word,
                'english_word2idx': english_word2idx,
                'english_idx2word': english_idx2word
            }
            
            torch.save(checkpoint, f"{output_dir}/best_model.pt")
            print(f"Best model saved (val_accuracy: {best_val_accuracy:.2f}%)")
    
    # Save the final model
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'french_word2idx': french_word2idx,
        'french_idx2word': french_idx2word,
        'english_word2idx': english_word2idx,
        'english_idx2word': english_idx2word
    }
    
    torch.save(final_checkpoint, f"{output_dir}/final_model.pt")
    print(f"Final model saved at epoch {epoch}")
    
    # Plot and save metrics
    plot_metrics(metrics, output_dir)
    
    return model, metrics


def plot_metrics(metrics, output_dir):
    """
    Plot and save training metrics
    
    Args:
        metrics: Dictionary containing training metrics
        output_dir: Directory to save the plots
    """
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation accuracy
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epoch'], metrics['val_accuracy'], label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_metrics_french_to_english.png")
    plt.close()
    print(f"Metrics plot saved to {output_dir}/training_metrics_french_to_english.png")


def translate_examples(model, data_loader, french_sentences, french_word2idx, english_idx2word, 
                     device, num_examples=5, max_len=50):
    """
    Translate a few example sentences to demonstrate model performance
    
    Args:
        model: Trained translation model
        data_loader: TranslationDataLoader instance
        french_sentences: List of French sentences
        french_word2idx: French word to index mapping
        english_idx2word: English index to word mapping
        device: Device to run inference on
        num_examples: Number of examples to translate
        max_len: Maximum length of generated translation
    """
    print("\nTranslating example sentences:")
    model.eval()
    
    # Choose random examples
    indices = np.random.choice(len(french_sentences), min(num_examples, len(french_sentences)), replace=False)
    
    for idx in indices:
        french_sentence = french_sentences[idx]
        print(f"\nFrench: {french_sentence}")
        
        # Convert sentence to indices
        tokens = french_sentence.split()
        input_indices = [french_word2idx.get(word, french_word2idx['< SOS >']) for word in tokens]
        
        # Add EOS token
        input_indices.append(french_word2idx['<EOS>'])
        
        # Pad sequence
        src_len = len(input_indices)
        padding = data_loader.max_french_length - src_len
        input_indices.extend([french_word2idx['<PAD>']] * padding)
        
        # Convert to tensor and add batch dimension
        src_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        
        # Generate translation
        with torch.no_grad():
            translations, attention_weights = model.translate(
                src=src_tensor,
                max_len=max_len,
                sos_idx=data_loader.english_word2idx['< SOS >'],
                eos_idx=data_loader.english_word2idx['<EOS>']
            )
        
        # Convert back to words
        for translation in translations:
            english_translation = indices_to_sentence(translation.cpu().numpy(), english_idx2word)
            print(f"English (translated): {english_translation}")


# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/data1/Sadia/Hamed_GRU/Dataset _English_to _French.docx"  # Path to your dataset
    OUTPUT_DIR = "french_to_english_model"  # Directory to save outputs
    
    # Training parameters
    BATCH_SIZE = 64
    EMBEDDING_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    SAVE_INTERVAL = 10  # Save metrics every 10 epochs
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    print("Starting training process...")
    model, metrics = train_translation_model(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        save_interval=SAVE_INTERVAL,
        device=DEVICE
    )
    
    # Load the data again to get example sentences
    translation_loader = TranslationDataLoader(batch_size=BATCH_SIZE)
    data = translation_loader.prepare_data(file_path=DATA_PATH)
    
    # Translate some examples
    translate_examples(
        model=model,
        data_loader=translation_loader,
        french_sentences=data['all_french_sentences'],
        french_word2idx=data['french_word2idx'],
        english_idx2word=data['english_idx2word'],
        device=DEVICE,
        num_examples=5
    )
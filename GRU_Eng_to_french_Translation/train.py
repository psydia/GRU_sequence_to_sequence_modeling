import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from dataloader import TranslationDataLoader
from model import create_translation_model
# Import your modules (assuming they're in the same directory)
# from data_loader import TranslationDataLoader
# from model import create_translation_model

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 500
SAVE_INTERVAL = 10
TEACHER_FORCING_RATIO = 0.5

# File paths
DATA_PATH = "/data1/Sadia/Hamed_GRU/Dataset _English_to _French.docx"  # Update with your actual file path
MODEL_SAVE_DIR = "/data1/Sadia/Hamed_GRU/models"
LOG_DIR = "/data1/Sadia/Hamed_GRU/logs"

# Ensure directories exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def calculate_word_accuracy(predictions, targets, french_idx2word, french_word2idx):
    """
    Calculate word-level accuracy (not strict sequence matching)
    
    Args:
        predictions: Predicted token indices [batch_size, seq_len]
        targets: Target token indices [batch_size, seq_len]
        french_idx2word: Index to word mapping
        french_word2idx: Word to index mapping
        
    Returns:
        accuracy: Word-level accuracy
    """
    pad_idx = french_word2idx['<PAD>']
    eos_idx = french_word2idx['<EOS>']
    
    # Convert predictions and targets to list of words
    correct_words = 0
    total_words = 0
    
    # Iterate through each sequence in the batch
    for i in range(predictions.size(0)):
        pred_seq = predictions[i]
        target_seq = targets[i]
        
        # Convert to list of words, ignoring padding and special tokens
        pred_words = []
        for idx in pred_seq:
            idx_item = idx.item()
            if idx_item != pad_idx and idx_item != eos_idx:
                if idx_item in french_idx2word:
                    pred_words.append(french_idx2word[idx_item])
        
        target_words = []
        for idx in target_seq:
            idx_item = idx.item()
            if idx_item != pad_idx and idx_item != eos_idx:
                if idx_item in french_idx2word:
                    target_words.append(french_idx2word[idx_item])
        
        # Count matches (allow for partial credit)
        for word in pred_words:
            if word in target_words:
                correct_words += 1
                # Remove the matched word to prevent double counting
                target_words.remove(word)
        
        total_words += len(pred_words)
    
    # Calculate accuracy
    accuracy = correct_words / total_words if total_words > 0 else 0
    return accuracy


def train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio):
    """
    Train the model for one epoch
    
    Args:
        model: Seq2Seq model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run the model on
        teacher_forcing_ratio: Probability of using teacher forcing
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Get data
        src = batch['encoder_input'].to(device)
        trg = batch['decoder_input'].to(device)
        trg_y = batch['decoder_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        
        # Calculate loss (ignore padding index)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Skip first token (SOS)
        trg_y = trg_y[:, 1:].reshape(-1)  # Skip first token
        
        # Handle float type error - ensure output is float
        output = output.to(torch.float32)
        
        loss = criterion(output, trg_y)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, french_idx2word, french_word2idx):
    """
    Evaluate the model
    
    Args:
        model: Seq2Seq model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run the model on
        french_idx2word: Index to word mapping
        french_word2idx: Word to index mapping
        
    Returns:
        epoch_loss: Average loss for the epoch
        accuracy: Word accuracy
    """
    model.eval()
    epoch_loss = 0
    total_accuracy = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Get data
            src = batch['encoder_input'].to(device)
            trg = batch['decoder_input'].to(device)
            trg_y = batch['decoder_output'].to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            output = model(src, trg, teacher_forcing_ratio=0)
            
            # Calculate loss (ignore padding index)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # Skip first token (SOS)
            trg_y = trg_y[:, 1:].reshape(-1)  # Skip first token
            
            # Handle float type error - ensure output is float
            output = output.to(torch.float32)
            
            loss = criterion(output, trg_y)
            epoch_loss += loss.item()
            
            # Calculate word-level accuracy
            # Get predicted tokens (greedy decoding)
            pred_tokens = torch.argmax(output.reshape(src.shape[0], -1, output_dim), dim=2)
            trg_tokens = trg_y.reshape(src.shape[0], -1)
            
            # Calculate accuracy
            batch_accuracy = calculate_word_accuracy(pred_tokens, trg_tokens, french_idx2word, french_word2idx)
            total_accuracy += batch_accuracy
    
    return epoch_loss / len(val_loader), total_accuracy / len(val_loader)


def main():
    print("Starting English-to-French translation training...")
    start_time = time.time()
    
    # Initialize data loader
    print("Loading and preparing data...")
    translation_loader = TranslationDataLoader(batch_size=BATCH_SIZE)
    data = translation_loader.prepare_data(file_path=DATA_PATH)
    
    # Create model
    print("Creating model...")
    model = create_translation_model(
        english_vocab_size=data['english_vocab_size'],
        french_vocab_size=data['french_vocab_size'],
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=device
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=translation_loader.french_word2idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create CSV file for logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_log_{timestamp}.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,accuracy,time_elapsed\n")
    
    # Training loop
    print("Starting training...")
    best_accuracy = 0
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=data['data_loader'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO
        )
        
        # Evaluate on the same data
        eval_loss, accuracy = evaluate(
            model=model,
            val_loader=data['data_loader'],
            criterion=criterion,
            device=device,
            french_idx2word=data['french_idx2word'],
            french_word2idx=data['french_word2idx']
        )
        
        # Calculate time elapsed
        time_elapsed = time.time() - start_time
        
        # Print progress
        print(f"Epoch: {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save model if it's the best so far (based on accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss,
                'accuracy': accuracy,
            }, checkpoint_path)
            print(f"New best model saved with accuracy: {accuracy:.4f}")
        
        # Log results
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{accuracy:.6f},{time_elapsed:.2f}\n")
    
    print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")
    print(f"Log file saved to: {log_file}")
    print(f"Best model saved to: {os.path.join(MODEL_SAVE_DIR, 'best_model.pt')}")

if __name__ == "__main__":
    main()
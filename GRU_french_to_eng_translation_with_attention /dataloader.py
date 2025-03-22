import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.model_selection import train_test_split
import os
import docx2txt  # For .docx files

class FrenchEnglishDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_outputs):
        """
        Dataset for French to English translation
        
        Args:
            encoder_inputs: Padded French sequences
            decoder_inputs: Padded English sequences for input to decoder
            decoder_outputs: Padded English sequences for output from decoder
        """
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs
        
    def __len__(self):
        return len(self.encoder_inputs)
    
    def __getitem__(self, idx):
        return {
            'encoder_input': torch.tensor(self.encoder_inputs[idx], dtype=torch.long),
            'decoder_input': torch.tensor(self.decoder_inputs[idx], dtype=torch.long),
            'decoder_output': torch.tensor(self.decoder_outputs[idx], dtype=torch.long)
        }


class TranslationDataLoader:
    def __init__(self, max_french_length=None, max_english_length=None, batch_size=32):
        """
        Initialize the DataLoader for translation task
        
        Args:
            max_french_length: Maximum length for French sequences (determined from data if None)
            max_english_length: Maximum length for English sequences (determined from data if None)
            batch_size: Batch size for DataLoader
        """
        self.max_french_length = max_french_length
        self.max_english_length = max_english_length
        self.batch_size = batch_size
        
        self.french_word2idx = {'<PAD>': 0, '< SOS >': 1, '<EOS>': 2}
        self.french_idx2word = {0: '<PAD>', 1: '< SOS >', 2: '<EOS>'}
        self.english_word2idx = {'<PAD>': 0, '< SOS >': 1, '<EOS>': 2}
        self.english_idx2word = {0: '<PAD>', 1: '< SOS >', 2: '<EOS>'}
        
        self.french_idx = 3  # Starting index for new words
        self.english_idx = 3
    
    def read_docx_file(self, file_path):
        """
        Read content from a .docx file
        
        Args:
            file_path: Path to the .docx file
            
        Returns:
            String content of the file
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract text from .docx
            content = docx2txt.process(file_path)
            return content
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def parse_dataset(self, content):
        """
        Parse the dataset from content
        
        Args:
            content: String content from the file
            
        Returns:
            List of (french, english) sentence pairs
        """
        content = content.replace("\\[", "").replace("\\]", "")
        
        # Extract pairs using regex pattern matching
        pattern = r'\("([^"]+)", "([^"]+)"\)'
        matches = re.findall(pattern, content)
        
        if not matches:
            raise ValueError("No valid sentence pairs found in the content. Check the format.")
        
        # Swap the english and french pairs since we want French to English translation
        # Original format: (english, french) -> Swapped: (french, english)
        swapped_matches = [(match[1], match[0]) for match in matches]
        
        return swapped_matches
    
    def build_vocabulary(self, sentences, is_english=False):
        """
        Build vocabulary from sentences
        
        Args:
            sentences: List of sentences
            is_english: Whether this is English vocabulary
            
        Returns:
            Updated word2idx and idx2word dictionaries
        """
        word2idx = self.english_word2idx if is_english else self.french_word2idx
        idx2word = self.english_idx2word if is_english else self.french_idx2word
        next_idx = self.english_idx if is_english else self.french_idx
        
        for sentence in sentences:
            for word in sentence.split():
                if word not in word2idx:
                    word2idx[word] = next_idx
                    idx2word[next_idx] = word
                    next_idx += 1
        
        if is_english:
            self.english_idx = next_idx
        else:
            self.french_idx = next_idx
        
        return word2idx, idx2word
    
    def sentence_to_indices(self, sentence, word2idx):
        """
        Convert a sentence to a list of indices
        
        Args:
            sentence: Input sentence
            word2idx: Word to index mapping
            
        Returns:
            List of indices
        """
        return [word2idx.get(word, word2idx['< SOS >']) for word in sentence.split()]
    
    def prepare_data(self, file_path=None, content=None):
        """
        Prepare the entire dataset for the translation task
        
        Args:
            file_path: Path to the .docx file (if provided)
            content: String content (if file_path not provided)
            
        Returns:
            Dictionary containing all necessary data for training and evaluation
        """
        # Get content either from file or directly
        if file_path:
            content = self.read_docx_file(file_path)
            if not content:
                raise ValueError(f"Failed to read content from {file_path}")
        elif not content:
            raise ValueError("Either file_path or content must be provided")
        
        # Parse the dataset
        pairs = self.parse_dataset(content)
        french_sentences = [pair[0] for pair in pairs]
        english_sentences = [pair[1] for pair in pairs]
        
        print(f"Loaded {len(pairs)} sentence pairs")
        
        # Build vocabularies
        self.french_word2idx, self.french_idx2word = self.build_vocabulary(french_sentences)
        self.english_word2idx, self.english_idx2word = self.build_vocabulary(english_sentences, is_english=True)
        
        self.french_vocab_size = len(self.french_word2idx)
        self.english_vocab_size = len(self.english_word2idx)
        
        print(f"French vocabulary size: {self.french_vocab_size}")
        print(f"English vocabulary size: {self.english_vocab_size}")
        
        # Convert sentences to sequences of indices
        french_indices = [self.sentence_to_indices(sentence, self.french_word2idx) for sentence in french_sentences]
        english_indices = [self.sentence_to_indices(sentence, self.english_word2idx) for sentence in english_sentences]
        
        # Determine max sequence lengths if not provided
        if not self.max_french_length:
            self.max_french_length = max([len(seq) for seq in french_indices]) + 1  # +1 for EOS
        if not self.max_english_length:
            self.max_english_length = max([len(seq) for seq in english_indices]) + 1  # +1 for EOS
            
        print(f"Max French sequence length: {self.max_french_length}")
        print(f"Max English sequence length: {self.max_english_length}")
        
        # Prepare encoder inputs (French)
        encoder_inputs = []
        for seq in french_indices:
            padded_seq = seq + [self.french_word2idx['<EOS>']]
            padded_seq = padded_seq + [self.french_word2idx['<PAD>']] * (self.max_french_length - len(padded_seq))
            encoder_inputs.append(padded_seq)
        
        # Prepare decoder inputs (English with SOS token)
        decoder_inputs = []
        for seq in english_indices:
            padded_seq = [self.english_word2idx['< SOS >']] + seq
            padded_seq = padded_seq + [self.english_word2idx['<PAD>']] * (self.max_english_length - len(padded_seq))
            decoder_inputs.append(padded_seq)
        
        # Prepare decoder outputs (English with EOS token)
        decoder_outputs = []
        for seq in english_indices:
            padded_seq = seq + [self.english_word2idx['<EOS>']]
            padded_seq = padded_seq + [self.english_word2idx['<PAD>']] * (self.max_english_length - len(padded_seq))
            decoder_outputs.append(padded_seq)
        
        # Convert to numpy arrays
        encoder_inputs = np.array(encoder_inputs)
        decoder_inputs = np.array(decoder_inputs)
        decoder_outputs = np.array(decoder_outputs)
        
        # Create dataset with the full data
        self.full_dataset = FrenchEnglishDataset(
            encoder_inputs,
            decoder_inputs,
            decoder_outputs
        )
        
        # Create PyTorch DataLoader for full dataset
        self.data_loader = DataLoader(
            self.full_dataset, 
            batch_size=self.batch_size, 
            shuffle=True  # Still shuffle for training
        )
        
        print(f"Full dataset size: {len(encoder_inputs)}")
        print(f"Full dataset batches: {len(self.data_loader)}")
        
        return {
            'data_loader': self.data_loader,
            'french_vocab_size': self.french_vocab_size,
            'english_vocab_size': self.english_vocab_size,
            'french_word2idx': self.french_word2idx,
            'french_idx2word': self.french_idx2word,
            'english_word2idx': self.english_word2idx,
            'english_idx2word': self.english_idx2word,
            'max_french_length': self.max_french_length,
            'max_english_length': self.max_english_length,
            'all_french_sentences': french_sentences,
            'all_english_sentences': english_sentences
        }
    
    def indices_to_sentence(self, indices, idx2word):
        """
        Convert a sequence of indices back to a sentence
        
        Args:
            indices: List of indices
            idx2word: Index to word mapping
            
        Returns:
            Reconstructed sentence
        """
        # Filter out padding, SOS, and EOS tokens
        words = []
        for idx in indices:
            if idx == 0:  # <PAD>
                continue
            if idx == 1:  # < SOS >
                continue
            if idx == 2:  # <EOS>
                break
            words.append(idx2word[idx])
        
        return ' '.join(words)


# Example usage:
def main():
    # Initialize data loader
    translation_loader = TranslationDataLoader(batch_size=16)
    
    # Prepare data from file
    file_path = "/data1/Sadia/Hamed_GRU/Dataset _English_to _French.docx"
    
    try:
        data = translation_loader.prepare_data(file_path=file_path)
        
        # Now you can access the data loader and other information
        print("Data preparation complete!")
        print(f"Full dataset batches: {len(data['data_loader'])}")
        
        # Example: iterate through a batch
        for batch_idx, batch in enumerate(data['data_loader']):
            print(f"Batch {batch_idx}:")
            print(f"Encoder input shape: {batch['encoder_input'].shape}")
            print(f"Decoder input shape: {batch['decoder_input'].shape}")
            print(f"Decoder output shape: {batch['decoder_output'].shape}")
            break  # Just show the first batch
    
    except Exception as e:
        print(f"Error preparing data: {e}")


if __name__ == "__main__":
    main()
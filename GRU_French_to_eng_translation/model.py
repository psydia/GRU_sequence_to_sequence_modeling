import torch
import torch.nn as nn

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Encoder for the translation model
        
        Args:
            input_size: Size of the input vocabulary (French vocab size)
            embedding_size: Size of the embedding vectors
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(EncoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Forward pass of the encoder
        
        Args:
            src: Source sequence [batch_size, seq_len]
            
        Returns:
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
        """
        # src: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, embedding_size]
        
        _, hidden = self.gru(embedded)
        # hidden: [num_layers, batch_size, hidden_size]
        
        return hidden


class DecoderGRU(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Decoder for the translation model
        
        Args:
            output_size: Size of the output vocabulary (English vocab size)
            embedding_size: Size of the embedding vectors
            hidden_size: Hidden size of the GRU
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(DecoderGRU, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(
            embedding_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        """
        Forward pass of the decoder for a single time step
        
        Args:
            input: Input token indices [batch_size]
            hidden: Previous hidden state [num_layers, batch_size, hidden_size]
            
        Returns:
            output: Output probabilities [batch_size, output_size]
            hidden: Updated hidden state [num_layers, batch_size, hidden_size]
        """
        # input: [batch_size]
        # hidden: [num_layers, batch_size, hidden_size]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        
        # Embed input
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_size]
        
        # Pass through GRU
        output, hidden = self.gru(embedded, hidden)
        # output: [batch_size, 1, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        
        # Feed through linear layer to predict next token
        output = output.squeeze(1)  # [batch_size, hidden_size]
        prediction = self.fc_out(output)  # [batch_size, output_size]
        
        return prediction, hidden


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        Sequence-to-sequence model combining encoder and decoder
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
            device: Device to run the model on
        """
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the sequence-to-sequence model
        
        Args:
            src: Source sequence [batch_size, src_len]
            trg: Target sequence [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder outputs for all timesteps [batch_size, trg_len, output_size]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode the source sequence
        hidden = self.encoder(src)
        
        # First input to the decoder is the < SOS > token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden = self.decoder(input, hidden)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token from output
            top1 = output.argmax(1)
            
            # Use either ground truth or predicted token as next input
            input = trg[:, t] if teacher_force else top1
        
        return outputs
    
    def translate(self, src, max_len, sos_idx, eos_idx):
        """
        Generate translation for a source sentence
        
        Args:
            src: Source sequence [batch_size, src_len]
            max_len: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            translations: List of token indices for translated sentences [batch_size, max_len]
        """
        batch_size = src.shape[0]
        
        # Tensor to store translations
        translations = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
        translations[:, 0] = sos_idx
        
        # Encode the source sequence
        hidden = self.encoder(src)
        
        # First input to the decoder is the < SOS > token
        input = torch.tensor([sos_idx] * batch_size).to(self.device)
        
        # Track which sequences have ended
        ended = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        for t in range(1, max_len):
            # Exit if all sequences have ended
            if ended.all():
                break
                
            # Pass through decoder
            with torch.no_grad():
                output, hidden = self.decoder(input, hidden)
            
            # Get the predicted token
            top1 = output.argmax(1)
            translations[:, t] = top1
            
            # Mark sequences that have generated EOS
            ended = ended | (top1 == eos_idx)
            
            # Use predicted token as next input
            input = top1
        
        return translations


def create_translation_model(french_vocab_size, english_vocab_size, 
                           embedding_size=256, hidden_size=512, 
                           num_layers=2, dropout=0.3, device='cuda'):
    """
    Create a GRU-based encoder-decoder model for translation from French to English
    
    Args:
        french_vocab_size: Size of the French vocabulary
        english_vocab_size: Size of the English vocabulary 
        embedding_size: Embedding dimension
        hidden_size: Hidden size of the GRU
        num_layers: Number of GRU layers
        dropout: Dropout probability
        device: Device to run the model on
        
    Returns:
        model: Seq2Seq model
    """
    # Create encoder (now takes French input)
    encoder = EncoderGRU(
        input_size=french_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create decoder (now produces English output)
    decoder = DecoderGRU(
        output_size=english_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create sequence-to-sequence model
    model = Seq2SeqGRU(encoder, decoder, device)
    model = model.to(device)
    
    return model


def indices_to_sentence(indices, idx2word):
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
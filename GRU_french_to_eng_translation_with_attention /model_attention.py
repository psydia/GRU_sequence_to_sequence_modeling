import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderGRU(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout=0.1):
        """
        Encoder for the translation model
        
        Args:
            input_size: Size of the input vocabulary (French vocabulary size)
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
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        """
        Forward pass of the encoder
        
        Args:
            src: Source sequence [batch_size, seq_len]
            
        Returns:
            outputs: Outputs from all timesteps [batch_size, seq_len, hidden_size*2]
            hidden: Final hidden state [num_layers, batch_size, hidden_size]
        """
        # src: [batch_size, seq_len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, seq_len, embedding_size]
        
        # Pass through bidirectional GRU
        outputs, hidden = self.gru(embedded)
        # outputs: [batch_size, seq_len, hidden_size*2]
        # hidden: [num_layers*2, batch_size, hidden_size]
        
        # Concatenate forward and backward hidden states
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat((hidden[:, 0], hidden[:, 1]), dim=2)
        # hidden: [num_layers, batch_size, hidden_size*2]
        
        # Project hidden state from hidden_size*2 to hidden_size
        hidden = self.fc(hidden)
        # hidden: [num_layers, batch_size, hidden_size]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        """
        Attention mechanism
        
        Args:
            enc_hidden_dim: Encoder hidden dimension
            dec_hidden_dim: Decoder hidden dimension
        """
        super(Attention, self).__init__()
        
        self.attn = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Calculate attention weights
        
        Args:
            hidden: Decoder hidden state [batch_size, dec_hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_dim*2]
            mask: Mask for padded elements [batch_size, src_len]
            
        Returns:
            attention: Attention weights [batch_size, src_len]
            context: Context vector [batch_size, enc_hidden_dim*2]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, dec_hidden_dim]
        
        # Create energy by concatenating encoder outputs and decoder hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, dec_hidden_dim]
        
        # Calculate attention scores
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)
        # attention: [batch_size, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        # context: [batch_size, enc_hidden_dim*2]
        
        return attention, context


class DecoderGRU(nn.Module):
    def __init__(self, output_size, embedding_size, enc_hidden_size, dec_hidden_size, num_layers=1, dropout=0.1):
        """
        Decoder with attention for the translation model
        
        Args:
            output_size: Size of the output vocabulary (English vocabulary size)
            embedding_size: Size of the embedding vectors
            enc_hidden_size: Hidden size of the encoder
            dec_hidden_size: Hidden size of the decoder
            num_layers: Number of GRU layers
            dropout: Dropout probability
        """
        super(DecoderGRU, self).__init__()
        self.output_size = output_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.attention = Attention(enc_hidden_size, dec_hidden_size)
        
        # Input to GRU will be embedding + context vector
        self.gru = nn.GRU(
            embedding_size + (enc_hidden_size * 2), 
            dec_hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer combines embedding, context, and GRU output
        self.fc_out = nn.Linear(embedding_size + (enc_hidden_size * 2) + dec_hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask=None):
        """
        Forward pass of the decoder for a single time step
        
        Args:
            input: Input token indices [batch_size]
            hidden: Previous hidden state [num_layers, batch_size, dec_hidden_size]
            encoder_outputs: Encoder outputs [batch_size, src_len, enc_hidden_size*2]
            mask: Mask for padding [batch_size, src_len]
            
        Returns:
            output: Output probabilities [batch_size, output_size]
            hidden: Updated hidden state [num_layers, batch_size, dec_hidden_size]
            attention: Attention weights [batch_size, src_len]
        """
        # input: [batch_size]
        # hidden: [num_layers, batch_size, dec_hidden_size]
        # encoder_outputs: [batch_size, src_len, enc_hidden_size*2]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        
        # Embed input
        embedded = self.dropout(self.embedding(input))
        # embedded: [batch_size, 1, embedding_size]
        
        # Get the attention and context vector
        # Use the top layer of the decoder hidden state
        attention, context = self.attention(hidden[-1], encoder_outputs, mask)
        # attention: [batch_size, src_len]
        # context: [batch_size, enc_hidden_size*2]
        
        # Expand context for concatenation
        context = context.unsqueeze(1)
        # context: [batch_size, 1, enc_hidden_size*2]
        
        # Concatenate embedding and context
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [batch_size, 1, embedding_size + enc_hidden_size*2]
        
        # Pass through GRU
        output, hidden = self.gru(rnn_input, hidden)
        # output: [batch_size, 1, dec_hidden_size]
        # hidden: [num_layers, batch_size, dec_hidden_size]
        
        # Concatenate embedding, context, and GRU output for prediction
        output = output.squeeze(1)  # [batch_size, dec_hidden_size]
        context = context.squeeze(1)  # [batch_size, enc_hidden_size*2]
        embedded = embedded.squeeze(1)  # [batch_size, embedding_size]
        
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        # prediction: [batch_size, output_size]
        
        return prediction, hidden, attention


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx=0):
        """
        Sequence-to-sequence model combining encoder and decoder with attention
        
        Args:
            encoder: Encoder module
            decoder: Decoder module
            device: Device to run the model on
            pad_idx: Padding token index
        """
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
        
    def create_mask(self, src):
        """Create mask for padded elements in source sequence"""
        mask = (src != self.pad_idx).to(self.device)
        return mask
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        Forward pass of the sequence-to-sequence model with attention
        
        Args:
            src: Source sequence (French) [batch_size, src_len]
            trg: Target sequence (English) [batch_size, trg_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            outputs: Decoder outputs for all timesteps [batch_size, trg_len, output_size]
            attentions: Attention weights for all timesteps [batch_size, trg_len, src_len]
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs and attentions
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Create mask for padded elements
        mask = self.create_mask(src)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the < SOS > token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Pass through decoder with attention
            output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Store output and attention
            outputs[:, t, :] = output
            attentions[:, t, :] = attention
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token from output
            top1 = output.argmax(1)
            
            # Use either ground truth or predicted token as next input
            input = trg[:, t] if teacher_force else top1
        
        return outputs, attentions
    
    def translate(self, src, max_len, sos_idx, eos_idx):
        """
        Generate translation for a source sentence with attention
        
        Args:
            src: Source sequence (French) [batch_size, src_len]
            max_len: Maximum length of generated sequence
            sos_idx: Start of sequence token index
            eos_idx: End of sequence token index
            
        Returns:
            translations: List of token indices for translated sentences [batch_size, max_len]
            attention_weights: Attention weights for visualization [batch_size, max_len, src_len]
        """
        batch_size = src.shape[0]
        
        # Tensor to store translations and attention weights
        translations = torch.zeros(batch_size, max_len, dtype=torch.long).to(self.device)
        translations[:, 0] = sos_idx
        attention_weights = torch.zeros(batch_size, max_len, src.shape[1]).to(self.device)
        
        # Create mask for padded elements
        mask = self.create_mask(src)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to the decoder is the < SOS > token
        input = torch.tensor([sos_idx] * batch_size).to(self.device)
        
        # Track which sequences have ended
        ended = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        for t in range(1, max_len):
            # Exit if all sequences have ended
            if ended.all():
                break
                
            # Pass through decoder with attention
            with torch.no_grad():
                output, hidden, attention = self.decoder(input, hidden, encoder_outputs, mask)
            
            # Store attention weights for visualization
            attention_weights[:, t, :] = attention
            
            # Get the predicted token
            top1 = output.argmax(1)
            translations[:, t] = top1
            
            # Mark sequences that have generated EOS
            ended = ended | (top1 == eos_idx)
            
            # Use predicted token as next input
            input = top1
        
        return translations, attention_weights


def create_translation_model(french_vocab_size, english_vocab_size, 
                            embedding_size=256, hidden_size=512, 
                            num_layers=2, dropout=0.3, device='cuda', pad_idx=0):
    """
    Create a GRU-based encoder-decoder model with attention for translation
    
    Args:
        french_vocab_size: Size of the French vocabulary
        english_vocab_size: Size of the English vocabulary 
        embedding_size: Embedding dimension
        hidden_size: Hidden size of the GRU
        num_layers: Number of GRU layers
        dropout: Dropout probability
        device: Device to run the model on
        pad_idx: Padding token index
        
    Returns:
        model: Seq2Seq model with attention
    """
    # Create encoder (bidirectional)
    encoder = EncoderGRU(
        input_size=french_vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create decoder with attention
    decoder = DecoderGRU(
        output_size=english_vocab_size,
        embedding_size=embedding_size,
        enc_hidden_size=hidden_size,
        dec_hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Create sequence-to-sequence model with attention
    model = Seq2SeqGRU(encoder, decoder, device, pad_idx)
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
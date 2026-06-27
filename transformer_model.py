import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return x

def generate_square_subsequent_mask(sz):
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=2048):
        super(MusicTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # 3. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Final LayerNorm (Standard GPT practice)
        self.ln_f = nn.LayerNorm(d_model)
        
        # 5. Output Head
        self.output_head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        # Standard GPT initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
        
        self.output_head.bias.data.zero_()

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, causal mask
        Returns:
            output: Tensor, shape [batch_size, seq_len, vocab_size]
        """
        # Embed and add positional encoding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if src_mask is None:
            device = src.device
            sz = src.size(1)
            src_mask = generate_square_subsequent_mask(sz).to(device)
            
        output = self.transformer_decoder(src, mask=src_mask, is_causal=True)
        
        # Apply final LayerNorm
        output = self.ln_f(output)
        
        # Map back to vocabulary
        logits = self.output_head(output)
        
        return logits

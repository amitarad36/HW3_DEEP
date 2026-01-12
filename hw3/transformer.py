import torch
import torch.nn as nn
import math


# Yuval


def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    '''
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    '''
    assert window_size%2 == 0, "window size must be an even number"
    seq_len = q.shape[-2]
    embed_dim = q.shape[-1]
    batch_size = q.shape[0]

    values, attention = None, None

    # ====== YOUR CODE: ======
    device = q.device
    
    # 1. Setup window indices
    # We create a relative range [-w/2, ..., +w/2]
    w_radius = window_size // 2
    window_shifts = torch.arange(-w_radius, w_radius + 1, device=device)
    
    # Broadcast to create a grid of indices: [SeqLen, WindowSize]
    # Each row 'i' contains indices [i-w/2, ..., i+w/2]
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(1)
    window_indices = seq_indices + window_shifts
    
    # 2. Mask out-of-bounds indices (left of 0 or right of seq_len)
    mask_valid_indices = (window_indices >= 0) & (window_indices < seq_len)
    
    # Clamp indices so 'index_select' doesn't crash (we will mask invalid ones later)
    window_indices = window_indices.clamp(min=0, max=seq_len - 1)
    
    # 3. Gather Keys (K)
    # We flatten the window indices to use index_select efficiently
    flat_indices = window_indices.flatten()
    k_subset = k.index_select(-2, flat_indices)
    # Reshape back to [Batch, ..., SeqLen, WindowSize, Dim]
    k_subset = k_subset.view(*k.shape[:-2], seq_len, len(window_shifts), -1)
    
    # 4. Compute Scores (Scaled Dot Product)
    # Q: [..., L, 1, D] * K_subset: [..., L, W, D] -> Sum last dim -> [..., L, W]
    attn_scores = (q.unsqueeze(-2) * k_subset).sum(dim=-1) / math.sqrt(embed_dim)
    
    # Apply valid-index mask (handle edges of the sliding window)
    attn_scores = attn_scores.masked_fill(~mask_valid_indices, float('-inf'))

    # 5. Apply Padding Mask
    if padding_mask is not None:
        # Expand valid indices to batch dimension to gather mask values
        gather_indices = window_indices.view(1, -1).expand(batch_size, -1)
        mask_subset = padding_mask.gather(1, gather_indices)
        mask_subset = mask_subset.view(batch_size, seq_len, len(window_shifts))
        
        # Broadcast to head dimension if necessary (for Multi-Head Attention)
        while mask_subset.dim() < attn_scores.dim():
            mask_subset = mask_subset.unsqueeze(1)
            
        attn_scores = attn_scores.masked_fill(mask_subset == 0, float('-inf'))
    
    # 6. Softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)
    
    # Handle NaNs (e.g. rows that are entirely masked out)
    if torch.isnan(attn_probs).any():
        attn_probs = attn_probs.clone()
        attn_probs[torch.isnan(attn_probs)] = 0.0
    
    # 7. Compute Values (V)
    v_subset = v.index_select(-2, flat_indices)
    v_subset = v_subset.view(*v.shape[:-2], seq_len, len(window_shifts), -1)
    
    # Weighted Sum: Probs * Values
    weighted_values = attn_probs.unsqueeze(-1) * v_subset
    values = weighted_values.sum(dim=-2)

    # Force outputs for padding tokens to be exactly 0.0
    if padding_mask is not None:
        out_mask = padding_mask
        # Broadcast mask to match value shape [Batch, (Heads), SeqLen, Dim]
        while out_mask.dim() < values.dim() - 1:
            out_mask = out_mask.unsqueeze(1)
        out_mask = out_mask.unsqueeze(-1)
        
        values = values.masked_fill(out_mask == 0, 0.0)

    # 8. Reconstruct Full Dense Attention Matrix (for return value)
    if attn_probs.dim() == 4:
        # Multi-head case: [Batch, Heads, SeqLen, SeqLen]
        dense_shape = (batch_size, q.shape[1], seq_len, seq_len)
        scatter_indices = window_indices.view(1, 1, seq_len, -1).expand(*attn_probs.shape)
    else:
        # Single-head case: [Batch, SeqLen, SeqLen]
        dense_shape = (batch_size, seq_len, seq_len)
        scatter_indices = window_indices.view(1, seq_len, -1).expand(*attn_probs.shape)
        
    full_attention = torch.zeros(dense_shape, device=device)
    # Scatter the sparse window probs back into the dense matrix
    full_attention.scatter_add_(dim=-1, index=scatter_indices, src=attn_probs)
    
    attention = full_attention
    # ========================

    return values, attention

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        
        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no teoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)
        
        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, 3*Dims]
        
        q, k, v = qkv.chunk(3, dim=-1) #[Batch, Head, SeqLen, Dims]
        
        # Determine value outputs
        # call the sliding window attention function you implemented
        # ====== YOUR CODE: ======
        values, attention = sliding_window_attention(q, k, v, self.window_size, padding_mask)
        # ========================

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim) #concatination of all heads
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o
        
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000): 
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model) 
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x
    
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        '''
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        '''
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, embed_dim, num_heads, window_size)
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask):
        '''
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        '''

        # ====== YOUR CODE: ======
        res_connection = x
        attn_output = self.self_attn(x, padding_mask)
        x = self.dropout(attn_output)
        x = self.norm1(res_connection + x)
        res_connection = x
        ff_output = self.feed_forward(x)
        x = self.dropout(ff_output)
        x = self.norm2(res_connection + x)
        # ========================
        
        return x
    
    
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, max_seq_length, window_size, dropout=0.1):
        '''
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        '''
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout) for _ in range(num_layers)])

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        '''
        output = None

        # ====== YOUR CODE: ======
        x = self.encoder_embedding(sentence)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        cls_token = x[:, 0, :] 
        output = self.classification_mlp(cls_token)
        # ========================
        
        
        return output  
    
    def predict(self, sentence, padding_mask):
        '''
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        '''
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds

    
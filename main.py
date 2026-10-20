import torch 
import torch.nn as nn
import math

def main():
    print(tokenizer("Hello world"))

# Transformer (feed forward pass)
# - Tokenizer
# - Encoder
#   - Embedding
#   - Positional Encoding
#   - Repeat Nx
#       - Encoder Layer
#           - MHA Layer + Add + Norm
#           - FFN Layer + Add + Norm
# - Decoder
#   - Embedding
#   - Positional Encoding
#   - Repeat Nx
#       - Decoder Layer 
#           - MMHA Layer + Add + Norm
#           - Cross Attention + Add + Norm
#           - FFN Layer + Add + Norm
# - Linear
# - Softmax

class Embedding(nn.Module):
    """
    Look up the embeddings for the given token ids

    Arg:
        torch.Tensor: token ids of shape (seq_len)

    Returns:
        torch.Tensor: embeddings of shape (seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding_matrix = nn.Parameter(init_weights((vocab_size, d_model)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, initial_seq_len: int):
        """
        Generates a table of size [initial_seq_len] that implements
        the following formula

        for all (pos x i)
        where   pos $\in$ [0, seq_len)
        and       i $\in$ [0, d_model)

        $PE_(pos, 2i)       = sin(pos * 10000^{-(2i / d_{model})})$
        $PE_(pos, 2i + 1)   = cos(pos * 10000^{-(2i / d_{model})})$
        """

        assert d_model % 2 == 0

        super().__init__()
        self.initial_seq_len = initial_seq_len

        # compute -(2i / d_model)
        # shape -> [d_model/2]
        exponents = torch.arange(0, d_model, 2, dtype=torch.float32) / -d_model

        # compute 10000^(-2i / d_model)
        # shape -> [d_model/2]
        denoms = torch.pow(10000, exponents)

        # shape -> [initial_seq_len, 1]
        pos = torch.arange(0, initial_seq_len).unsqueeze(1)

        # shape -> [initial_seq_len, d_model]
        self.register_buffer("encoding_table", torch.empty((initial_seq_len, d_model)))

        # even dimensions
        # $PE_(pos, 2i)       = sin(pos * 10000^{-(2i / d_{model})})$
        self.encoding_table[:, 0::2] = torch.sin(pos * denoms)

        # odd dimensions
        # $PE_(pos, 2i + 1)   = cos(pos * 10000^{-(2i / d_{model})})$
        self.encoding_table[:, 1::2] = torch.cos(pos * denoms)

    def forward(self, token_embeddings: torch.Tensor):
        """
        Add the positional information to the token embedding representations

        Args:
            torch.Tensor: token emebeddings of shape (seq_len, d_model)

        Returns:
            torch.Tensor: embeddings infused with positional information (seq_len, d_model)
        """

        seq_len = len(token_embeddings)

        # ensure we have enough precomputed positional encoding values
        # TODO: if we have less, we should be able to dynamically add more
        assert seq_len <= self.initial_seq_len

        positional_encodings = self.encoding_table[:seq_len]

        return token_embeddings + positional_encodings

class SublayerConnection(nn.Module):
    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, token_embeddings: torch.Tensor):
        """
        Computes LayerNorm(x + SubLayer(x))

        Args:
            torch.Tensor: current token_embeddings of shape (seq_len, d_model)

        Returns:
            torch.Tensor: infused token embeddings of shape (seq_len, d_model)
        """
        return self.layer_norm(token_embeddings + self.sublayer(token_embeddings))

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        # upward projection
        self.linear_1 = nn.Linear(d_model, d_ff)
        # downward projection
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, token_embeddings: torch.Tensor):
        """
        Computes FFN(x) = max(0, x * W_1 + B_1) * W_2 + B_2

        Args:
            torch.Tensor: current token embeddings of shape (seq_len, d_model)

        Returns:
            torch.Tensor: infused token embeddings of shape (seq_len, d_model)
        """
        return self.linear_2(torch.relu(self.linear_1(token_embeddings)))

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # contains W_q, W_k and W_v
        self.projection_matrix = nn.Linear(d_model, 3 * d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(self, token_embeddings: torch.Tensor):
        projection = self.projection_matrix(token_embeddings)
        q, k, v = projection.chunk(3, dim = -1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        context = merge_heads(attention(q, k, v))
        return self.output_matrix(context)

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # contains W_k and W_v weights
        self.kv_projection_matrix = nn.Linear(d_model, 2 * d_model)
        # contains W_q weights
        self.q_projection_matrix = nn.Linear(d_model, d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(self, src_token_embeddings: torch.Tensor, target_token_embeddings: torch.Tensor):
        kv_projection = self.kv_projection_matrix(src_token_embeddings)
        q = self.q_projection_matrix(target_token_embeddings)
        k, v = kv_projection.chunk(2, dim = -1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        context = merge_heads(attention(q, k, v))
        return self.output_matrix(context)

class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # contains W_q, W_k and W_v
        self.qkv_projection_matrix = nn.Linear(d_model, 3 * d_model)
        self.output_matrix = nn.Linear(d_model, d_model)

    def forward(self, token_embeddings: torch.Tensor):
        projection = self.qkv_projection_matrix(token_embeddings)
        q, k, v = projection.chunk(3, dim = -1)

        q = split_heads(q, self.n_heads, self.d_k)
        k = split_heads(k, self.n_heads, self.d_k)
        v = split_heads(v, self.n_heads, self.d_k)

        seq_len = token_embeddings.size(0)
        mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=token_embeddings.device)
        )

        context = merge_heads(attention(q, k, v, mask))
        return self.output_matrix(context)


def split_heads(x: torch.Tensor, n_heads: int, d_k: int) -> torch.Tensor:
    """Convert [seq_len, d_model] into [n_heads, seq_len, d_k]."""
    seq_len = x.size(0)
    return x.reshape(seq_len, n_heads, d_k).transpose(0, 1)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert [n_heads, seq_len, d_k] into [seq_len, d_model]."""
    n_heads, seq_len, d_k = x.shape
    return x.transpose(0, 1).reshape(seq_len, n_heads * d_k)

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask = None) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim = -1)
    return weights @ v

# Helpers

def tokenizer(str):
    """
    Fake tokenizer that just returns a random integer
    for every word in the input string
    
    Returns: 
        torch.Tensor: token ids of shape (seq_len)
    """
    n_words = len(str.split())
    return torch.randint(0, 100, (n_words,))

def init_weights(shape):
    # TODO: implement different shape initialization approaches
    # actually, this won't work because things like LayerNorm will not be initialized 
    # this way, so it makes sense to go through all the named parameters
    return torch.empty(shape)

if __name__ == "__main__":
    main()

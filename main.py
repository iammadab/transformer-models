import torch 
import torch.nn as nn

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

def tokenizer(str):
    """
    Fake tokenizer that just returns a random integer
    for every word in the input string

    Returns: 
        torch.Tensor: token ids of shape (seq_len)
    """
    n_words = len(str.split())
    return torch.randint(0, 100, (n_words,))

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        # initialize the embedding matrix
        self.embedding_matrix = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the embeddings for the given token ids

        Returns:
            torch.Tensor: Embeddings of shape (seq_len, d_model)
        """
        return self.embedding_matrix[token_ids]

if __name__ == "__main__":
    main()

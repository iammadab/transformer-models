import torch 
import torch.nn

def main():
    print(tokenizer("Hello world"))
    print("Hello from transformer-models!")

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

# Tokenizer
# No need for a tokenizer yet, just use rand int
# then during training replace the fake tokenizer

def tokenizer(str):
    """
    Fake tokenizer that just returns a random integer
    for every word in the input string
    """
    n_words = len(str.split())
    return torch.randint(0, 100, (n_words,))

if __name__ == "__main__":
    main()

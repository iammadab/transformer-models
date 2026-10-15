import torch 
import torch.nn

def main():
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
    # return a random number for each word in the str

if __name__ == "__main__":
    main()

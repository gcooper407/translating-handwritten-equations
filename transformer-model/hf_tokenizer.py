from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from pathlib import Path

class EquationTokenizer:
  def __init__(self, vocab_size=30000):
    self.vocab_size = vocab_size
    self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    
    # Initialize a Byte-Pair Encoding (BPE) tokenizer
    self.tokenizer = Tokenizer(models.BPE())
    
    # Set up pre-tokenizer to split on whitespace and punctuation
    self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # Set up a decoder for the BPE tokenizer
    self.tokenizer.decoder = decoders.ByteLevel()
    
    # Add a post-processor to handle special tokens
    self.tokenizer.post_processor = processors.TemplateProcessing(
      single="[CLS] $A [SEP]",
      pair="[CLS] $A [SEP] $B:1 [SEP]:1",
      special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
      ],
    )
    
    # Prepare a trainer for the BPE tokenizer
    self.trainer = trainers.BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=self.special_tokens
    )
  
  def train(self, files):
    """Train the tokenizer on a dataset."""
    self.tokenizer.train(files=files, trainer=self.trainer)

  def encode(self, text):
    """Encode a string into token IDs."""
    return self.tokenizer.encode(text).ids

  def decode(self, token_ids):
    """Decode token IDs back into a string."""
    return self.tokenizer.decode(token_ids)
  
  def save(self, path):
    """Save the trained tokenizer to a file."""
    self.tokenizer.save(path)
    print(f"Tokenizer saved to {path}")
  
  def load(self, path):
    """Load a tokenizer from a file."""
    self.tokenizer = Tokenizer.from_file(path)
    print(f"Tokenizer loaded from {path}")

# Example usage:
if __name__ == "__main__":
  train_txt_files = list(Path("./CACHED_CROHME/TXT/train").glob("*.txt"))


  latex_files = [str(file) for file in train_txt_files]
  
  tokenizer = EquationTokenizer()
  tokenizer.train(files=latex_files)
  # tokenizer.save("latex_tokenizer.json")

  # Example encoding and decoding
  example_text = "E = mc^2"
  encoded = tokenizer.encode(example_text)
  print(f"Encoded: {encoded}")

  decoded = tokenizer.decode(encoded)
  print(f"Decoded: {decoded}")

import re
import collections
from typing import List


class EquationTokenizer:
    def __init__(self):
        # Special tokens for padding, sequence control, and unknown characters
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        self.vocab_size = 0
        self.id_to_token = {}
        self.token_to_id = {}

    def tokenize(self, formula):
        # Regex to identify and isolate letters, numbers, formatting characters
        # (e.g. ^, *, etc.), and LaTeX commands (i.e. string of characters
        # prefixed by \)
        expression = r"\\[a-zA-Z]+|\\.|[a-zA-Z0-9]|\S"
        return re.findall(expression, formula)

    def train(self, expressions: List[str]):
        # assign id to each special token (i.e. add special tokens to vocab)
        for token in self.special_tokens:
            self.token_to_id[token] = self.vocab_size
            self.vocab_size += 1

        # Keep track of # of occurrences of each character across all LaTeX expressions
        frequency = collections.Counter()
        for expr in expressions:
            frequency.update(self.tokenize(expr))

        # assign id to each token in order of frequency (i.e. add tokens to vocab)
        # (we want frequent tokens to come up first, it makes accessing faster/better)
        for token, _ in frequency.most_common():
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.vocab_size += 1

        # for encoding, we also want a way to map from ids to tokens
        for key, val in self.token_to_id.items():
            self.id_to_token[val] = key

    def encode(self, expression):
        # Prefix and suffix with BOS (beginning of sequence) and EOS (end of sequence) markers
        sequence = ["[BOS]"] + self.tokenize(expression) + ["[EOS]"]

        encoded_sequence = []

        # Convert tokens to IDs, fallback to UNK for unknown symbols
        for tok in sequence:
            if tok in self.token_to_id:
                encoded_sequence.append(self.token_to_id[tok])
            else:
                encoded_sequence.append(self.token_to_id["[UNK]"])

        return encoded_sequence

    def decode(self, ids):
        decoded_sequence = []

        # Convert ID list back to token list
        for id in ids:
            if id in self.id_to_token:
              decoded_sequence.append(self.id_to_token[id])
            else:
              decoded_sequence.append("[UNK]")

        # Stop at EOS token, drop all tokens afterward
        if "[EOS]" in decoded_sequence:
            decoded_sequence = decoded_sequence[:decoded_sequence.index("[EOS]")]

        # Replace [UNK] with a readable fallback character (?) and remove special tokens
        final_tokens = ["?" if t == "[UNK]" else t for t in decoded_sequence if t not in self.special_tokens]

        # Join characters together to reform the original expression
        return "".join(final_tokens)

import re
from collections import Counter


class Vocabulary:
    # Reserve IDs for padding, caption boundaries, and unknown words so
    # captions can be represented consistently in training and inference.
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold

        # Special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        # Token-to-ID mapping
        self.stoi = {
            self.pad_token: 0,
            self.start_token: 1,
            self.end_token: 2,
            self.unk_token: 3
        }

        # ID-to-token mapping
        self.itos = {
            idx: token for token, idx in self.stoi.items()
        }

    # Split text into lowercase words and punctuation so captions can be
    # counted, converted to IDs, or prepared for display consistently.
    def tokenize(self, text):
        """Convert a sentence into lowercase tokens."""
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    # Build the word-to-ID mapping from training captions; rare words are
    # left out and will use the unknown-token ID.
    def build_vocabulary(self, captions):
        """Build vocabulary from a collection of captions."""
        frequencies = Counter()

        for caption in captions:
            tokens = self.tokenize(caption)
            frequencies.update(tokens)

        for token, frequency in frequencies.items():
            if frequency >= self.freq_threshold:
                if token not in self.stoi:
                    index = len(self.stoi)
                    self.stoi[token] = index
                    self.itos[index] = token

    # Convert a training caption into model input IDs, marking where the
    # caption starts and ends so the decoder can learn those boundaries.
    def numericalize(self, text):
        """Convert a sentence into token IDs."""
        tokens = self.tokenize(text)

        numericalized = [
            self.stoi.get(token, self.stoi[self.unk_token])
            for token in tokens
        ]

        return (
            [self.stoi[self.start_token]]
            + numericalized
            + [self.stoi[self.end_token]]
        )

    # During inference, convert the decoder's generated IDs into readable
    # text and stop at the end token.
    def decode(self, token_ids):
        """Convert token IDs back into words."""
        words = []

        for idx in token_ids:
            token = self.itos.get(int(idx), self.unk_token)

            if token == self.end_token:
                break

            if token not in (
                self.start_token,
                self.pad_token,
                self.unk_token
            ):
                words.append(token)

        return " ".join(words)

    # Provide the vocabulary size when configuring the decoder's output
    # layer, which predicts one token from this vocabulary at each step.
    def __len__(self):
        return len(self.stoi)


'''
What this file provides
Method	            Purpose
tokenize()	        Sentence → tokens
build_vocabulary()	Build word dictionary
numericalize()	    Sentence → IDs
decode()	        IDs → sentence
__len__()	        Vocabulary size
'''

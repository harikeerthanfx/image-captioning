
from src.vocabulary import Vocabulary


def main():
    # Sample captions (temporary test data)
    captions = [
        "A dog is running on grass.",
        "A dog is playing outside.",
        "A cat is running on grass."
    ]

    # Create vocabulary
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(captions)

    print("Vocabulary size:", len(vocab))

    # Test tokenization
    sentence = "A dog is running on grass."
    tokens = vocab.tokenize(sentence)
    print("\nTokens:", tokens)

    # Test numericalization
    numericalized = vocab.numericalize(sentence)
    print("Token IDs:", numericalized)

    # Test decoding
    decoded = vocab.decode(numericalized)
    print("Decoded sentence:", decoded)

    # Test unknown word
    unknown_sentence = "A dog is flying."
    unknown_ids = vocab.numericalize(unknown_sentence)
    print("\nUnknown word test:", unknown_ids)

    # Test special tokens
    print("\nSpecial token IDs:")
    print("PAD:", vocab.stoi["<pad>"])
    print("START:", vocab.stoi["<start>"])
    print("END:", vocab.stoi["<end>"])
    print("UNK:", vocab.stoi["<unk>"])

    print("\nVocabulary test passed!")


if __name__ == "__main__":
    main()


from src.dataset import Flickr8kDataset, create_dataloader
from src.vocabulary import Vocabulary

# Dataset paths
IMAGE_DIR = "data/images/Flicker8k_Dataset"
CAPTIONS_FILE = "data/annotations/Flickr8k.token.txt"
TRAIN_FILE = "data/annotations/Flickr_8k.trainImages.txt"

# Build vocabulary from training captions
vocabulary = Vocabulary(freq_threshold=2)

from src.dataset import load_captions

captions_dict = load_captions(CAPTIONS_FILE, TRAIN_FILE)
all_captions = [
    caption
    for captions in captions_dict.values()
    for caption in captions
]

vocabulary.build_vocabulary(all_captions)

# Create Dataset
train_dataset = Flickr8kDataset(
    IMAGE_DIR,
    CAPTIONS_FILE,
    TRAIN_FILE,
    vocabulary
)

print("Number of training samples:", len(train_dataset))
print("Vocabulary size:", len(vocabulary))

# Create DataLoader
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Fetch one batch
images, captions = next(iter(train_loader))

print("\nImage batch shape:", images.shape)
print("Caption batch shape:", captions.shape)
print("First caption IDs:", captions[0])

print("\nDataset and DataLoader test passed!")

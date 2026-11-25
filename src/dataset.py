
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Reuse our preprocessing function from preprocessing.py.
# It loads an image, resizes it and converts it into a PyTorch tensor.
# The resulting image tensor will later be passed to our CNN encoder.
from src.preprocessing import preprocess_image


# ============================================================
# 1. LOAD AND GROUP CAPTIONS
# ============================================================

def load_captions(captions_file, split_file):
    """
    Reads the Flickr8k caption file and groups captions by image.

    Input:
        captions_file -> Contains all image-caption pairs.
        split_file    -> Contains image filenames for train/val/test.

    Output:
        A dictionary:
        {
            "image1.jpg": [caption1, caption2, ...],
            "image2.jpg": [caption1, caption2, ...]
        }
    """

    # Read the selected split (for example, training images).
    # A set allows us to efficiently check whether an image belongs
    # to this split using: image_name in image_names.
    with open(split_file, "r") as file:
        image_names = set(
            line.strip() for line in file if line.strip()
        )

    # This dictionary will connect each image filename
    # to all its corresponding captions.
    captions_dict = {}

    # Read the complete caption file line by line.
    with open(captions_file, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            # Each line looks like:
            # 1000268201_693b08cb0e.jpg#0    A child is climbing stairs.
            #
            # Separate the image identifier from the caption.
            image_caption, caption = line.split("\t", 1)

            # Remove the caption number (#0, #1, etc.).
            # We only need the actual image filename.
            image_name = image_caption.split("#")[0]

            # Keep only images belonging to the selected split.
            if image_name in image_names:

                # Create an empty list the first time we encounter
                # a particular image.
                if image_name not in captions_dict:
                    captions_dict[image_name] = []

                # Add this caption to that image's list.
                # Repeated appearances of the same filename
                # automatically collect its other captions.
                captions_dict[image_name].append(caption)

    return captions_dict


# ============================================================
# 2. FLICKR8K DATASET
# ============================================================

class Flickr8kDataset(Dataset):

    def __init__(self, image_dir, captions_file, split_file, vocabulary):

        # Location of the actual Flickr8k image files.
        self.image_dir = Path(image_dir)

        # Our Vocabulary object converts words into integer IDs.
        # This is necessary because the Transformer works with numbers,
        # not raw text.
        self.vocabulary = vocabulary

        # Load the dictionary:
        # image filename -> list of its captions.
        captions_dict = load_captions(captions_file, split_file)

        # Convert the dictionary into individual training samples.
        #
        # Each sample contains:
        # (image filename, ONE caption)
        #
        # Therefore, an image with five captions produces five samples.
        self.samples = []

        for image_name, captions in captions_dict.items():

            for caption in captions:

                self.samples.append((image_name, caption))

    # PyTorch uses this to know how many samples are in the dataset.
    def __len__(self):
        return len(self.samples)

    # PyTorch calls this whenever it requests a particular sample.
    # For example, dataset[0] requests the first image-caption pair.
    def __getitem__(self, index):

        # Retrieve one image filename and its corresponding caption.
        image_name, caption = self.samples[index]

        # Construct the complete path to the image.
        image_path = self.image_dir / image_name

        # Load and preprocess the image.
        # Output shape: (3, 224, 224)
        image = preprocess_image(image_path)

        # Convert the caption into token IDs using our vocabulary.
        # Example: "a dog is running" -> [START, a, dog, is, running, END]
        caption_ids = self.vocabulary.numericalize(caption)

        # Convert the list of token IDs into a PyTorch tensor.
        # Data type is long because these are integer token IDs.
        caption_tensor = torch.tensor(
            caption_ids, dtype=torch.long
        )

        # Return one image and its caption.
        # These will later be combined into batches by the DataLoader.
        return image, caption_tensor


# ============================================================
# 3. BATCH PREPARATION (COLLATE FUNCTION)
# ============================================================

def collate_fn(batch):

    # Separate the images and captions from the samples.
    # batch is a list of (image, caption) pairs.
    images, captions = zip(*batch)

    # Stack individual image tensors into one batch.
    # Example: 32 images -> (32, 3, 224, 224)
    images = torch.stack(images, dim=0)

    # Captions have different lengths.
    # Neural networks need equal-length sequences within a batch.
    #
    # pad_sequence adds PAD token ID (0) to shorter captions
    # so that every caption in the batch has the same length.
    #
    # batch_first=True gives shape:
    # (batch_size, maximum_caption_length)
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


# ============================================================
# 4. CREATE DATALOADER
# ============================================================

def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
):

    # DataLoader automatically:
    # 1. Selects samples from the Dataset.
    # 2. Groups them into batches.
    # 3. Calls collate_fn to prepare each batch.
    #
    # shuffle=True randomizes sample order during training.
    # This helps prevent the model from learning from a fixed order.
    #
    # num_workers=0 means image loading happens in the main process.
    # This is a simple starting point for our Mac setup.

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )


from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from src.encoder import get_device
from src.model import ImageCaptioningModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_PATH = Path(__file__).parent / "dog.jpg"

VOCAB_SIZE = 4460

PAD_ID = 0
START_ID = 1


def main():

    # Select Apple GPU (MPS) if available
    device = get_device()
    print("Using device:", device)

    # --------------------------------------------------
    # Step 1: Load and preprocess a real image
    # --------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # --------------------------------------------------
    # Step 2: Create sample caption token IDs
    # --------------------------------------------------

    # Dummy token IDs for testing the decoder.
    # These are not an actual caption.
    caption_ids = torch.tensor(
        [[START_ID, 4, 5, 6, 2, PAD_ID, PAD_ID]],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------
    # Step 3: Create complete CNN + Transformer model
    # --------------------------------------------------

    model = ImageCaptioningModel(
        vocab_size=VOCAB_SIZE
    ).to(device)

    model.eval()

    # --------------------------------------------------
    # Step 4: Run complete forward pass
    # --------------------------------------------------

    with torch.inference_mode():
        output = model(image, caption_ids)

    # --------------------------------------------------
    # Step 5: Verify output
    # --------------------------------------------------

    print("\nImage shape:", image.shape)
    print("Caption IDs shape:", caption_ids.shape)
    print("Model output shape:", output.shape)

    expected_shape = (1, 7, VOCAB_SIZE)

    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    print("\nFull CNN + Transformer integration test passed!")


if __name__ == "__main__":
    main()

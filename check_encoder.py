
import sys
import torch

from src.preprocessing import preprocess_image
from src.encoder import ResNet18Encoder, get_device


def main():

    device = get_device()

    print("Using device:", device)

    # Use an actual image if its path is provided.
    if len(sys.argv) > 1:
        image_tensor = preprocess_image(sys.argv[1])
        print("Loaded and preprocessed actual image.")

    else:
        # Random tensor for an initial shape test.
        image_tensor = torch.randn(3, 224, 224)
        print("Using random tensor for shape testing.")

    # Add batch dimension.
    images = image_tensor.unsqueeze(0).to(device)

    # Create encoder.
    encoder = ResNet18Encoder(d_model=512).to(device)
    encoder.eval()

    # Run the encoder.
    with torch.no_grad():
        output = encoder(images)

    print("\n--- Results ---")
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(output.shape))

    # Check whether the architecture behaves as expected.
    assert output.shape == (1, 49, 512)

    frozen = sum(
        p.numel()
        for p in encoder.backbone.parameters()
        if not p.requires_grad
    )

    trainable = sum(
        p.numel()
        for p in encoder.projection.parameters()
        if p.requires_grad
    )

    print("Frozen CNN parameters:", frozen)
    print("Trainable projection parameters:", trainable)

    print("\nEncoder test passed!")


if __name__ == "__main__":
    main()

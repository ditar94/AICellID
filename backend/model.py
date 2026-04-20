from PIL import Image

# Placeholder until a trained model is wired up.
# Replace with: load checkpoint, run torchvision transforms, return argmax + softmax score.

CLASSES = [
    "neutrophil",
    "lymphocyte",
    "monocyte",
    "eosinophil",
    "basophil",
    "blast",
]


def classify_image(image: Image.Image) -> tuple[str, float]:
    return ("lymphocyte", 0.0)

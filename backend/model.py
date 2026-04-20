from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CHECKPOINT_PATH = Path(__file__).parent / "models" / "cell_classifier.pt"

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model: Optional[nn.Module] = None
_classes: Optional[List[str]] = None


def _load() -> Tuple[Optional[nn.Module], Optional[List[str]]]:
    global _model, _classes
    if _model is not None:
        return _model, _classes
    if not CHECKPOINT_PATH.exists():
        return None, None

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _model, _classes = model, classes
    return model, classes


def classify_image(image: Image.Image) -> Tuple[str, float]:
    model, classes = _load()
    if model is None:
        return ("unknown (no model loaded)", 0.0)

    x = _transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return (classes[idx], float(probs[idx]))

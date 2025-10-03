# models/image_model.py
from typing import List, Dict, Union
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from core.decorators import timeit

_MODEL_NAME = "google/vit-base-patch16-224"
_clf = None  # encapsulated singleton


def _get_pipeline():
    global _clf
    if _clf is None:
        processor = AutoImageProcessor.from_pretrained(_MODEL_NAME, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(
            _MODEL_NAME,
            use_safetensors=True
        )
        _clf = pipeline(
            task="image-classification",
            model=model,
            feature_extractor=processor,
            framework="pt",
        )
    return _clf


@timeit
def classify_image(image: Union[str, Path, Image.Image], top_k: int = 5) -> List[Dict]:
    """
    image: path or PIL.Image
    Returns: [{'label': str, 'score': float}, ...]
    """
    clf = _get_pipeline()

    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            return [{"error": f"Image not found: {p}"}]
        img = Image.open(p).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        return [{"error": "Unsupported image input"}]

    return clf(img, top_k=top_k)

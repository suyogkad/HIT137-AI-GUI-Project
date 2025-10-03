# this file handles image classification using a huggingface vision transformer model

from typing import List, Dict, Union
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from core.decorators import timeit  # this is for measuring how long a function takes to run

# name of the pretrained model we are using (vision transformer trained on ImageNet)
_MODEL_NAME = "google/vit-base-patch16-224"
_clf = None  # we keep the pipeline object here so we don’t reload it again and again


def _get_pipeline():
    """
    this function creates the image classification pipeline only once
    if it's already created, it just returns the existing one
    """
    global _clf
    if _clf is None:
        # load processor (does resizing, normalization etc.)
        processor = AutoImageProcessor.from_pretrained(_MODEL_NAME, use_fast=False)
        
        # load the actual model weights (classification head etc.)
        model = AutoModelForImageClassification.from_pretrained(
            _MODEL_NAME,
            use_safetensors=True  # safer format than pytorch bin
        )
        
        # build the pipeline (easier way to do preprocessing + inference + postprocessing)
        _clf = pipeline(
            task="image-classification",
            model=model,
            feature_extractor=processor,  # processor handles image preprocessing
            framework="pt",  # pytorch backend
        )
    return _clf


@timeit  # this decorator will print how long classify_image takes
def classify_image(image: Union[str, Path, Image.Image], top_k: int = 5) -> List[Dict]:
    """
    main function to classify an image
    - input can be a path string, a Path object, or a PIL image
    - it will return a list like: [{'label': 'cat', 'score': 0.98}, ...]
    """
    clf = _get_pipeline()

    # if the input is a path, check if it exists and then open the image
    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.exists():
            return [{"error": f"Image not found: {p}"}]
        img = Image.open(p).convert("RGB")  # always convert to rgb
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")  # if already a PIL image, just ensure rgb
    else:
        return [{"error": "Unsupported image input"}]

    # run classification and return top-k predictions
    return clf(img, top_k=top_k)

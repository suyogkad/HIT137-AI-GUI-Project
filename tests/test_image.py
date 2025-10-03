# tests/test_image.py
# this file is just for testing the image classifier model by itself
# we load one sample image from assets/ and print the top-5 labels

from pathlib import Path
import sys
from PIL import Image
from transformers import (
    AutoImageProcessor,          # image preprocessor (resize/normalize)
    AutoModelForImageClassification,  # vit model with imagenet head
    pipeline,                    # simple wrapper to run everything
)

# model name on hugging face (vit base, 224px, imagenet-1k)
# if internet is blocked, you can download the model and set this to a local folder like "./models/vit"
MODEL_NAME = "google/vit-base-patch16-224"


def project_path(*parts) -> Path:
    """
    this is a small helper to resolve paths relative to project root.
    our tests/ folder is one level deeper, so we go up one.
    """
    here = Path(__file__).resolve()
    root = here.parents[1]  # go to project root
    return root.joinpath(*parts)


def load_image() -> Image.Image:
    """
    this function try to load assets/sample.png first, then assets/sample.jpg.
    this lets us support either filename without editing the code.
    """
    png_path = project_path("assets", "sample.png")
    jpg_path = project_path("assets", "sample.jpg")

    if png_path.exists():
        img_path = png_path
    elif jpg_path.exists():
        img_path = jpg_path
    else:
        # friendly error if the file is missing
        raise FileNotFoundError(
            f"Could not find sample image at:\n"
            f" - {png_path}\n"
            f" - {jpg_path}\n"
            f"put your image in assets/ and name it sample.png or sample.jpg"
        )

    print(f"📷 using image: {img_path}")
    # always convert to rgb so the model gets consistent input
    return Image.open(img_path).convert("RGB")


def build_pipeline():
    """
    this function create the image-classification pipeline.
    we force use_fast=False to avoid torch.compiler calls (works better on older torch).
    we also force safetensors to avoid torch.load security gate.
    """
    print("Loading image processor and model (safetensors, slow processor)…")

    # slow processor is fine for our assignment and avoids a torch.compiler attr error
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    # use_safetensors=True avoids loading legacy .bin with torch.load
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
    )

    # pipeline stitches preprocess/postprocesses
    clf = pipeline(
        task="image-classification",
        model=model,
        feature_extractor=processor,  # older/newer transformers both accept this
        framework="pt",             
    )
    return clf


def main():
    # first make sure the image exists and opens
    try:
        image = load_image()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # now build the pipeline (downloads on first run, then cached)
    try:
        classifier = build_pipeline()
    except Exception as e:
        print(f"Failed to build pipeline: {e}")
        sys.exit(1)

    print("Image classification pipeline ready.\n")

    # ask for the top-5 labels so we can see nearby classes (imagenet has 1000 classes)
    results = classifier(image, top_k=5)

    print("🔎 top-5 predictions:")
    for i, r in enumerate(results, 1):
        label = r.get("label", "unknown")
        score = r.get("score", 0.0)
        print(f"{i}. {label:<25}  score={score:.4f}")


if __name__ == "__main__":
    main()
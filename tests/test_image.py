from pathlib import Path
import sys
from PIL import Image
from transformers import (
    AutoImageProcessor,          # slow processor (we will force use_fast=False)
    AutoModelForImageClassification,
    pipeline,
)

MODEL_NAME = "google/vit-base-patch16-224"


def project_path(*parts) -> Path:
    """
    Resolve a path relative to the project root (one level up from /tests).
    """
    here = Path(__file__).resolve()
    root = here.parents[1]
    return root.joinpath(*parts)


def load_image() -> Image.Image:
    """
    Load the sample image from assets/. Tries PNG first, then JPG.
    """
    png_path = project_path("assets", "sample.png")
    jpg_path = project_path("assets", "sample.jpg")

    if png_path.exists():
        img_path = png_path
    elif jpg_path.exists():
        img_path = jpg_path
    else:
        raise FileNotFoundError(
            f"⚠️ Could not find sample image at:\n"
            f" - {png_path}\n"
            f" - {jpg_path}\n"
            f"Make sure your image is in assets/ as sample.png or sample.jpg."
        )

    print(f"📷 Using image: {img_path}")
    img = Image.open(img_path).convert("RGB")
    return img


def build_pipeline():
    print("🔄 Loading image processor and model (safetensors, slow processor)…")

    # Force the **slow** processor to avoid torch.compiler path
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)

    # Force safetensors to bypass torch.load restriction
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
    )

    clf = pipeline(
        task="image-classification",
        model=model,
        feature_extractor=processor,   # works across transformers versions
        framework="pt",                # ensure PyTorch
    )
    return clf


def main():
    try:
        image = load_image()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    classifier = build_pipeline()
    print("✅ Image classification pipeline ready.\n")

    results = classifier(image, top_k=5)

    print("🔎 Top-5 predictions:")
    for i, r in enumerate(results, 1):
        label = r.get("label", "unknown")
        score = r.get("score", 0.0)
        print(f"{i}. {label:<25}  score={score:.4f}")


if __name__ == "__main__":
    main()

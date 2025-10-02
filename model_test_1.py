import sys
from transformers import pipeline

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_test_1.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]

    # Force safetensors; pick a model that has safetensors weights
    clf = pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224",
        model_kwargs={"use_safetensors": True}
    )

    preds = clf(image_path, top_k=5)
    print("\nTop-5 predictions:")
    for p in preds:
        print(f"- {p['label']}: {p['score']:.4f}")

if __name__ == "__main__":
    main()

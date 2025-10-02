import sys
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_test_2.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]

    image = Image.open(path).convert("RGB")

    # Force safetensors here too
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True   # <— important
    )

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print("\nCaption:", caption)

if __name__ == "__main__":
    main()

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

MODEL_NAME = "siebert/sentiment-roberta-large-english"

def build_pipeline():
    # Load tokenizer (fast tokenizer where available)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Force safetensors; this avoids torch.load and the >=2.6 hard block
    # Requires the 'safetensors' package installed.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            use_safetensors=True,   # <- key line
        )
    except Exception as e:
        # Helpful fallback hint if safetensors isn’t installed or TF version is old
        raise RuntimeError(
            "Failed to load model with safetensors. "
            "Make sure you ran: pip install -U transformers safetensors"
        ) from e

    return pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
    )

def main():
    print("🔄 Building sentiment pipeline (safetensors)…")
    nlp = build_pipeline()
    print("✅ Pipeline ready.\n")

    samples = [
        "I really love working on this project!",
        "This assignment is making me stressed.",
        "The apple looks delicious.",
    ]
    for s in samples:
        out = nlp(s)
        print(f"Input: {s}\nOutput: {out}\n")

if __name__ == "__main__":
    main()

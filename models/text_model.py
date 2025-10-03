# models/text_model.py
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from core.decorators import timeit

_MODEL_NAME = "siebert/sentiment-roberta-large-english"
_nlp = None  # encapsulated singleton


def _get_pipeline():
    global _nlp
    if _nlp is None:
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME,
            use_safetensors=True
        )
        _nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return _nlp


@timeit
def analyze_sentiment(text: str) -> List[Dict]:
    """
    Returns: [{'label': 'POSITIVE'|'NEGATIVE', 'score': float}]
    """
    text = (text or "").strip()
    if not text:
        return [{"error": "Please enter some text."}]
    return _get_pipeline()(text)

# this file handles text sentiment analysis using huggingface roberta model

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from core.decorators import timeit  # to measure how long a function runs

# name of the pretrained model we are using (roberta sentiment model trained on english text)
_MODEL_NAME = "siebert/sentiment-roberta-large-english"
_nlp = None  # we keep the pipeline here so we don’t reload it again and again


def _get_pipeline():
    """
    this function builds the sentiment pipeline only once
    if it's already made, it just reuses it
    """
    global _nlp
    if _nlp is None:
        # load tokenizer (splits text into tokens that roberta understands)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)

        # load the roberta model with a classification head for sentiment
        model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL_NAME,
            use_safetensors=True  # safer format to store weights
        )

        # build the pipeline (text → tokens → model → label + score)
        _nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return _nlp


@timeit  # measure how long the function takes
def analyze_sentiment(text: str) -> List[Dict]:
    """
    main function to run sentiment analysis
    - takes in a string
    - returns something like: [{'label': 'POSITIVE', 'score': 0.99}]
    """
    text = (text or "").strip()  # clean up input text
    if not text:
        return [{"error": "Please enter some text."}]  # edge case for empty input
    return _get_pipeline()(text)  # run pipeline and return results

# HIT137 – AI GUI Project

This is our Assignment 3 project for the Software Now module (Masters in Data Science, Semester 1).
The project shows how to use Hugging Face models inside a simple Tkinter GUI application.

## Features

- Text Sentiment Analysis
- Image Classification
- Tkinter GUI

## How to Run

1. Create & activate conda environment

```python
conda create -n hit137_env python=3.10 -y
conda activate hit137_env
```

2. Install requirements
```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers pillow
```
3. Run tests
```python
python tests/test_sentiment.py
python tests/test_image.py
```
4. Run GUI
```python
python main.py
```

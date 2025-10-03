# core/base_classes.py
from abc import ABC, abstractmethod

class LoggingMixin:
    def log(self, msg: str):
        print(f"[ModelLog] {msg}")

class ModelBase(ABC):
    def __init__(self, name: str):
        self._name = name  # encapsulation by convention

    @property
    def name(self):
        return self._name

    @abstractmethod
    def run(self, x):
        pass  # overridden by subclasses

class TextModel(LoggingMixin, ModelBase):  # multiple inheritance with mixin
    def run(self, x):
        self.log(f"Running text model {self.name}")
        return {"info": f"{self.name} would run on text: {str(x)[:40]}..."}

class ImageModel(LoggingMixin, ModelBase):  # multiple inheritance with mixin
    def run(self, x):
        self.log(f"Running image model {self.name}")
        return {"info": f"{self.name} would run on image path: {x}"}

from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
import nodetails as nd

TrainingSet = namedtuple("TrainingSet", [
    "x_train", "y_train", "x_val", "y_val"])

Lexicon = namedtuple("Lexicon", [
    "x_tkn", "y_tkn", "x_len", "y_len"])

AbstractiveModel = namedtuple("AbstractiveModel", [
    "training", "encoder", "decoder", "lexicon"])

ExtractiveSummary = namedtuple("ExtractiveSummary", [
    "summary", "reference", "sentences", "paragraphs"])

class BaseEngine(ABC):
    def __init__(self, name="Unnamed_NDEngine", config=None):
        self.name = name
        self.ready = True
        self.config = defaultdict(lambda: None, config)

        if not self.config["disable_vram_growth"]:
            nd.enable_vram_growth()

    @abstractmethod
    def load(self):
        self.ready = True

    @abstractmethod
    def unload(self):
        self.ready = False

    @abstractmethod
    def summarize(self, text_body, **kwargs) -> dict:
        """
        This method must return a dict with at least
        a ``summary`` field.
        """
        if self.ready:
            return {"summary": text_body}

    def __del__(self):
        self.unload()

# END OF _types.py

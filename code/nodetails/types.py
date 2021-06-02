from collections import namedtuple

TrainingSet = namedtuple("TrainingSet", [
    "x_train", "y_train", "x_val", "y_val"])

Lexicon = namedtuple("Lexicon", [
    "x_tkn", "y_tkn", "x_len", "y_len"])

AbstractiveModel = namedtuple("AbstractiveModel", [
    "training", "encoder", "decoder", "lexicon"])

ExtractiveSummary = namedtuple("ExtractiveSummary", [
    "summary", "reference", "sentences", "paragraphs"])

# END OF types.py

if __name__ == "__main__":
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from nodetails import abstractive
from nodetails.abstractive import InferenceParameters
from nodetails.extractive import *

ExtractiveSummary = namedtuple(
    "ExtractiveSummary",
    ["summary", "reference", "sentences", "paragraphs"])


class IntegratedSummarizer:
    def __init__(self, abstractive_model: InferenceParameters):
        self.abst_model = abstractive_model

    @staticmethod
    def _extractive_wikipedia(article_url, length=7):
        html = fetch_article(article_url)
        paragraphs = split_paragraphs(html, preset="wikipedia")
        sentences = tag_sentences(paragraphs)

        scores = score_sentences(sentences)
        reference = get_best_items(scores, length)

        summary = " ".join([sentences[it] for it in reference])

        return ExtractiveSummary(summary, reference, sentences, paragraphs)

    def from_wikipedia_article(self, article_url):
        extsum = self._extractive_wikipedia(article_url, 10)
        abstsum = abstractive.make_inference(self.abst_model, extsum.summary)

        return abstsum


if __name__ == "__main__":
    model_dir = "../../data/_models"
    model_name = f"nodetails--food_reviews--80-10--100000"
    infr_params = abstractive.load(f"{model_dir}/{model_name}.model")
    
    summarizer = IntegratedSummarizer(infr_params)
    summary = summarizer.from_wikipedia_article("https://en.wikipedia.org/wiki/Citation_needed")

# END OF integration.py
from nodetails import abstractive, extractive
from nodetails.abstractive import InferenceParameters


class IntegratedSummarizer:
    def __init__(self, abstractive_model: InferenceParameters):
        self.abst_model = abstractive_model

    def from_wikipedia_article(self, article_url):
        extsum = extractive.get_summary_from_url(article_url, 10)
        abstsum = abstractive.make_inference(self.abst_model, extsum.summary)

        return abstsum


if __name__ == "__main__":
    model_dir = "../../data/_models"
    model_name = f"nodetails--food_reviews--80-10--100000"
    infr_params = abstractive.load(f"{model_dir}/{model_name}.model")
    
    summarizer = IntegratedSummarizer(infr_params)
    summary = summarizer.from_wikipedia_article("https://en.wikipedia.org/wiki/Citation_needed")

# END OF integration.py
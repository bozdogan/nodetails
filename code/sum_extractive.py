from nodetails.extractive import *

if __name__ == "__main__":
    html = fetch_article("https://en.wikipedia.org/wiki/Citation_needed")
    paragraphs = split_paragraphs(html)
    sentences = tag_sentences(paragraphs)

    scores = score_sentences(sentences)
    # NOTE(bora): It's an ordered dictionary such as
    #     scores = {(para_no, sent_no): score, ...}

    summary = []
    for sent_id in get_best_items(scores, 7):
        summary.append("%s %s" % (sent_id, sentences[sent_id]))

    print("\n\n   == SUMMARY ==\n")
    print("\n".join(summary))

# END OF sum_abstractive.py

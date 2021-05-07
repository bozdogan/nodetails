from collections import OrderedDict
import re
from urllib import request
from bs4 import BeautifulSoup
import nltk

def fetch_article(articleurl):
    urlhandle = request.urlopen(articleurl)
    return urlhandle.read()


def split_paragraphs(html, preset="wikipedia"):
    article_parsed = BeautifulSoup(html, "lxml")

    if preset == "wikipedia":
        # TODO(bora): Stuff inside <li> tags aren't wrapped by <p> tags. This
        # query doesn't catch list items.
        p_tags = article_parsed.find_all("p")
        
        paragraphs = []
        for i, p in enumerate(p_tags):
            para = p.text
            para = re.sub(r"\[[0-9]*\]", " ", para)

            # NOTE(bora): This line messes "Citation Needed" article and 
            # doesn't really help with general articles that much, so.
            #para = re.sub(r"\[citation needed\]", " ", para)

            para = re.sub(r"\s+", " ", para)

            paragraphs.append((i, para))

        return paragraphs

    return None



def tag_sentences(paragraphs):
    result = OrderedDict()  # NOTE(bora): This needs to be an ordered list as it is sorted later in the code.
    for para_no, para in paragraphs:
        sentences = nltk.sent_tokenize(para)
        for sent_no, sent in enumerate(sentences):
            result[(para_no, sent_no)] = sent

    return result


def calc_frequencies(text, stopwords=None):
    if stopwords == None: stopwords = []
    assert isinstance(text, str)

    # NOTE(bora): Remove non-alphabetic characters. Languages other than
    # English require more detailed implementation, though.
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)

    result = {}
    for word in nltk.word_tokenize(text):
        if word.lower() not in stopwords:
            if word not in result:
                result[word] = 1
            else:
                result[word] += 1

    max_freq = max(result.values())
    for word in result:
        result[word] = result[word]/max_freq

    return result


def score_sentences(sentences):
    if isinstance(sentences, dict):
        tagged = True
        text = " ".join(sentences.values())
    else:
        assert isinstance(sentences, list), \
               "A list rexpected, '%s' found" % type(sentences)
        text = " ".join(sentences)

    freq = calc_frequencies(text,
                            stopwords=nltk.corpus.stopwords.words("english"))

    def update_score(scores, key, word, sent):
        if word in freq and len(sent.split(" ")) < 30:
            if key not in scores:
                scores[key] = freq[word]
            else:
                scores[key] += freq[word]

    scores = {}
    if tagged:
        for key, sent in sentences.items():
            for word in nltk.word_tokenize(sent.lower()):
                update_score(scores, key, word, sent)
    else:
        for key, sent in enumerate(sentences):
            for word in nltk.word_tokenize(sent.lower()):
                update_score(scores, key, word, sent)


    return scores


def get_best_items(a_dict, n: int):
    return list(reversed(sorted(a_dict, key=a_dict.get)))[:n]


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


from collections import OrderedDict
import re
import urllib.request
import bs4
import nltk


def read_from_web(articleurl):
    urlhandle = urllib.request.urlopen(articleurl)
    return urlhandle.read()


def get_article_paragraphs(article_html):
    parsed_article = bs4.BeautifulSoup(article_html, "lxml")
    p_tags = parsed_article.find_all("p")

    paragraphs = OrderedDict([(i, p.text) for i, p in enumerate(p_tags)])

    # NOTE(bora): Remove brackets and extra spaces
    for i in paragraphs:
        paragraphs[i] = re.sub(r"\[[0-9]*\]", " ", paragraphs[i])
        paragraphs[i] = re.sub(r"\s+", " ", paragraphs[i])

    return paragraphs


def get_frequencies(paragraphs, stopwords=None):
    if stopwords is None: stopwords = []
    if isinstance(paragraphs, str):
        article_text = paragraphs
    else:
        assert isinstance(paragraphs, dict), \
               "Dictionary expected, '%s' found"%type(paragraphs)
        article_text = "".join(paragraphs.values())

    # NOTE(bora): Remove non-alphabetic characters
    article_text = re.sub("[^a-zA-Z]", " ", article_text)
    article_text = re.sub(r"\s+", " ", article_text)

    result = {}
    for word in nltk.word_tokenize(article_text):
        if word.lower() not in stopwords:
            if word not in result.keys():
                result[word] = 1
            else:
                result[word] += 1

    maximum = max(result.values())
    for word in result:
        result[word] = (result[word]/maximum)

    return result


def get_sentences(paragraphs):
    sentence_list = OrderedDict()

    for parano, para in enumerate(paragraphs.values()):
        sentences = nltk.sent_tokenize(para)
        for sentno, sentence in enumerate(sentences):
            sentence_list[(parano, sentno)] = sentence

    return sentence_list


def get_scores(sentences, word_frequencies):
    scores = {}
    for key, sentence in sentences.items():
        for token in nltk.word_tokenize(sentence.lower()):
            if token in word_frequencies:
                if len(sentence.split(" ")) < 30:
                    if key not in scores:
                        scores[key] = word_frequencies[token]
                    else:
                        scores[key] += word_frequencies[token]

    return scores


def extract(article_text):
    paragraphs = get_article_paragraphs(article_text)
    stopwords = nltk.corpus.stopwords.words("english")
    word_freq = get_frequencies(paragraphs, stopwords)

    sentences = get_sentences(paragraphs)
    scores = get_scores(sentences, word_freq)

    return sentences, scores, word_freq


def best_n_items(d: dict, n: int):
    return list(reversed(sorted(d, key=d.get)))[:n]


if  __name__ == "__main__":
    article_url = "https://en.wikipedia.org/wiki/Object-oriented_programming"
    print("Fetching article")
    html = read_from_web(article_url)
    # print("Cleaning article body")
    # paragraphs = get_article_paragraphs(html)
    #
    # print("Counting words")
    # stopwords_english = nltk.corpus.stopwords.words("english")
    # word_frequencies = get_frequencies(paragraphs, stopwords_english)
    #
    # print("Generating sentence scores")
    # sentences = get_sentences(paragraphs)
    # scores = get_scores(sentences, word_frequencies)
    # summary = list(reversed(sorted(scores, key=scores.get)))[:7]

    sentences, scores, word_freq = extract(html)
    summary = best_n_items(scores, 7)

    # NOTE(bora): Sort by order of occurrence
    summary = sorted(summary)

    print("\n\n   == SUMMARY ==\n")
    print(*summary)
    print("\n\n".join([sentences[it] for it in summary]))

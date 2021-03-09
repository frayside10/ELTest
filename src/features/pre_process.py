import nltk; nltk.download('stopwords')
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from nltk.corpus import stopwords
#from gensim.models import CoherenceModel
import gensim

def set_stop_words(extra_words):
     stop_words = stopwords.words('english')
     stop_words.extend(extra_words)
     return stop_words


def sent_to_words(sentences):
    for sentence in sentences:
        yield (simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts, added_stop_words):
    return [[word for word in simple_preprocess(str(doc)) if word not in set_stop_words(added_stop_words)] for doc in texts]


def bigrams(words, bi_min=2, tri_min=10):
    bigram = gensim.models.Phrases(words, min_count=bi_min)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return bigram_mod


def get_corpus(df, added_stop_words):
    """
    Get Bigram Model, Corpus, id2word mapping
    """
    words = list(sent_to_words(df.text))
    words = remove_stopwords(words, added_stop_words)
    bigram = bigrams(words)
    bigram = [bigram[review] for review in words]
    id2word = corpora.Dictionary(bigram)
    id2word.filter_extremes(no_below=10, no_above=0.35)
    id2word.compactify()
    corpus = [id2word.doc2bow(text) for text in bigram]
    return corpus, id2word, bigram

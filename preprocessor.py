from nltk.corpus import stopwords
import re
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    ""

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        return [self._preprocess(doc) for doc in data]
    
    def _preprocess(self, doc):
        ""
        
        doc = self._clean(doc)
        doc = self._tokenize(doc)
        # TODO lowercase?
        doc = self._remove_stopwords(doc)
        # TODO lemmatization/stemming?
        return doc
    
    def _clean(self, doc):
        ""
        
        # Remove "RT" (retweeted)
        doc = re.sub(r"RT ", " ", doc)
        # Replace @username with MENTION
        doc = re.sub(r"(\@)\S+", "MENTION", doc)
        # Replace #atheism with HASHTAG atheism
        doc = re.sub(r"(\#)", "HASHTAG ", doc)
        # Replace Quran quotes with QURAN_QUOTE
        doc = re.sub(r"(Quran \(*[0-9]+[:|\.][0-9]+\)*)", "Quran QURANQUOTE", doc)
        # Replace Bible quotes with BIBLE_QUOTE
        # It is assumed that Quran quotes are preceded by "Quran ",
        # while Bible quotes are preceded by the respective book
        doc = re.sub(r"[0-9]+[:|\.][0-9]+", "BIBLEQUOTE", doc)
        return doc
    
    def _tokenize(self, doc):
        ""
        
        # Tokenizer taken from sklearn
        # TODO replace with nltk?
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return re.findall(token_pattern, doc)
    
    def _remove_stopwords(self, doc):
        ""
        
        sw = set(stopwords.words("english"))
        return [tok for tok in doc if tok not in sw]
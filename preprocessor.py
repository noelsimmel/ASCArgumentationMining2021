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
        return doc
    
    def _tokenize(self, doc):
        ""
        
        # Custom tokenizer (based on sklearn's)
        # TODO replace with nltk?
        token_pattern = re.compile(r"""(?u)             # Unicode
                                    [0-9]+[:|\.][0-9]+  # Match Bible/Quran quotes 
                                                        # (e.g. 24:2 or 24.2)
                                    |
                                    \b\w\w+\b           # Match tokens with len>1""", 
                                   re.X)
        return re.findall(token_pattern, doc)
    
    def _remove_stopwords(self, doc):
        ""
        
        sw = set(stopwords.words("english"))
        return [tok for tok in doc if tok not in sw]
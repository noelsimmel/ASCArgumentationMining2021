# preprocessor.py
# Preprocesses a corpus for classification

import re

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    """Class for custom preprocessing (cleaning, tokenization, stemming, 
    removing stop words). Inherits basic functionality from sklearn base classes. 
    """

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        """Applies preprocessing to all documents in the corpus.

        Args:
            data (list): List of documents as strings.
            y (list, optional): List of class labels. Currently not implemented. 
            Defaults to None.

        Returns:
            list: List preprocessed strings or list of lists of tokens (if tokenized).
        """
        
        return [self._preprocess(doc) for doc in data]
    
    def _preprocess(self, doc):
        """Preprocessing pipeline. (Un)comment steps in exclude/include them. 
        
        NOTE: Custom transformers rely on tokenized input!

        Args:
            doc (string): Input text.

        Returns:
            string|list: Cleaned string or list of string tokens.
        """
        
        # _clean is needed for TwitterFeaturesExtractor()
        doc = self._clean(doc)
        # Tokenization is needed for all custom features except average word length
        doc = self._tokenize(doc)
        ### All steps from here on out rely on tokenized doc! ###
        doc = self._remove_stopwords(doc)
        # Stemming is disabled by default as it decreased accuracy
        # doc = self._stem(doc)
        return doc
    
    def _clean(self, doc):
        """Applies basic preprocessing using regular expressions, i.e. removing/
        masking Twitter related data and converting to lowercase.

        Args:
            doc (string): Input text.

        Returns:
            [string]: Cleaned input text.
        """
        
        # Remove "RT" (retweeted)
        doc = re.sub(r"RT ", " ", doc)
        doc = doc.lower()
        # Replace @username with "mention" (remove username)
        doc = re.sub(r"(\@)\S+", "mention", doc)
        # Replace #atheism with "hashtag" atheism
        doc = re.sub(r"(\#)", "hashtag ", doc)
        # Replace Quran quotes with "quranquote"
        doc = re.sub(r"(quran \(*[0-9]+[:|\.][0-9]+\)*)", "quran quranquote", doc)
        # Replace Bible quotes with "biblequote"
        # It is assumed that Quran quotes are preceded by "Quran ",
        # while Bible quotes are preceded by the respective book
        doc = re.sub(r"[0-9]+[:|\.][0-9]+", "biblequote", doc)
        return doc
    
    def _tokenize(self, doc):
        """Tokenization using the regex pattern from sklearn. This worked better 
        on the ASC than nltk tokenizers.

        Args:
            doc (string): Input text.

        Returns:
            list: List of tokens.
        """
        
        # Token pattern taken from sklearn
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        token_pattern = re.compile(r"(?u)\b\w\w+\b")
        return re.findall(token_pattern, doc)
    
    def _remove_stopwords(self, tokens):
        """Removes stop words from the nltk English stop words list.

        Args:
            tokens (list): List of tokens (1 document).

        Returns:
            list: List of tokens without stop words.
        """
        
        sw = set(stopwords.words("english"))
        return [tok for tok in tokens if tok not in sw]
    
    def _stem(self, tokens):
        """Stems input using nltk's snowball stemmer.

        Args:
            tokens (list): List of tokens (1 document).

        Returns:
            list: List of stemmed tokens.
        """
        
        stemmer = SnowballStemmer("english")
        return [stemmer.stem(tok) for tok in tokens]
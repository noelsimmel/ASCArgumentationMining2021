# transformers.py
# Custom transformers (feature extractors) for use in an sklearn pipeline

from abc import ABC, abstractmethod

from nltk import ne_chunk, pos_tag
from numpy import mean
from sklearn.base import BaseEstimator, TransformerMixin

class BaseCustomTransformer(ABC, BaseEstimator, TransformerMixin):
    """Abstract base class for custom transformers (feature extractors). 
    Inherits basic functionality from sklearn base classes. 
    
    Derived classes need to implement a make_feature(data) method that 
    extracts the feature and returns a list of lists.
    """

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        # Uncomment for pandas version:
        # data = data["text"].apply(self.make_feature)
        
        return self.make_feature(data)
    
    @abstractmethod
    def make_feature(self, data):
        """Extracts features. Must be implemented in derived classes (abstract method).

        Args:
            data (list): List of lists, each inner list contains a (tokenized) 
            document.

        Returns:
            list: The same list, documents replaced with feature values (ints/floats).
        """
        
        return []

class AtheismPolarityExtractor(BaseCustomTransformer):
    """Custom transformer class.
    
    Compares the input against a hardcoded set of pro and anti atheism words. 
    """
    
    def __init__(self):
        """Constructor. 
        
        Contains sets of 5 pro and anti atheism words each + their stemmed forms 
        according to the nltk snowball stemmer. Selection is based on the most 
        discriminative tokens in the ASC favor/against atheism classes.
        """
        
        self.pro_atheism  = {"freethinker", "evidence", "atheist", "freethink", "evid"}
        self.anti_atheism = {"teamjesus", "holy", "lord", "holi", "amen"}
        
    def make_feature(self, data):
        """Calculates pro/anti atheism polarity for all documents."""
        
        return [[self._get_polarity(doc)] for doc in data]
    
    def _get_polarity(self, doc):
        """Calculates pro/anti atheism polarity for one document. 

        Args:
            doc (list): List of string tokens.

        Returns:
            int: Sum of pro/anti atheism token counts. +1 for each pro atheism 
            token in the document, -1 for each anti atheism token.
        """
        
        s = 0
        for tok in doc:
            if tok in self.pro_atheism: s += 1
            elif tok in self.anti_atheism: s -= 1
        return s 

class AverageWordLengthExtractor(BaseCustomTransformer):
    """Custom transformer class.
    
    Extracts the average word/token length.
    """

    def make_feature(self, data):
        """Calculates average token length (numpy.mean()) for all documents. 

        Args:
            data (list): List of untokenized strings or list of list of string tokens. 
            If untokenized, naive .split() tokenization will be applied.

        Returns:
            list: List of lists, each list contains mean as float.
        """
        
        # For untokenized text
        try:
            return [[mean([len(token) for token in doc.split()])] for doc in data]
        # For tokenized text
        except AttributeError:
            return [[mean([len(token) for token in doc])] for doc in data]
        
class NamedEntityExtractor(BaseCustomTransformer):
    """Custom transformer class.
    
    Extracts the number of named entities.
    """
    
    def make_feature(self, data):
        """Calculates the number of named entities for all documents using nltk 
        POS tagging and chunking. Does not work well on the ASC due to its 
        non-standard vocabulary.

        Args:
            data (list): List of tokenized documents.

        Returns:
            list: List of lists, each list contains number of NEs as an int.
        """
        
        # POS tagging
        # Exclude masks as they confuse the parser
        masks = {"MENTION", "HASHTAG", "BIBLEQUOTE", "QURANQUOTE"}
        data = [[t for t in doc if t not in masks] for doc in data]
        tagged = [pos_tag(doc) for doc in data]
        # NER using nltk
        chunked = [ne_chunk(doc) for doc in tagged]
        # Quick and dirty counting named entities:
        # Flatten list (= syntax tree) and count leftover tuples, 
        # since NEs are the innermost branches
        # Based on https://stackoverflow.com/a/952952
        flattened = [[item for sublist in tree for item in sublist] for tree in chunked]
        return [[sum(type(t) == tuple for t in doc)] for doc in flattened]
    
class TwitterFeaturesExtractor(BaseCustomTransformer):
    """Custom transformer class.
    
    Extracts the number of Twitter-specific tokens, i.e. hashtags and mentions.
    """
    
    def make_feature(self, data):
        """Calculates the sum of mentions and hashtags for all documents. 
        This relies on a cleaning step in preprocessing that masks mentions/@ as 
        "mention" and hashtags/# as "hashtag".

        Args:
            data (list): List of tokenized documents.

        Returns:
            list: List of lists, each list contains Twitter token count as an int.
        """
        
        return [[doc.count("mention") + doc.count("hashtag")] for doc in data]
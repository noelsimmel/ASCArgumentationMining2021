# classifier.py
# The classifier

from collections import namedtuple
import logging
from time import time

import joblib
from nltk import pos_tag
from pandas import read_table
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

# File imports
from transformers import AtheismPolarityExtractor, AverageWordLengthExtractor, \
                         NamedEntityExtractor, TwitterFeaturesExtractor 
from preprocessor import Preprocessor

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(funcName)s:%(message)s")
file_handler = logging.FileHandler("logfile.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def dummy(arg):
    """Dummy function, simply returns the argument.
    
    Needed for using sklearn transformers on preprocessed input: https://stackoverflow.com/a/52855200
    """
    
    return arg

class ASCClassifier:
    """Classifier for the Atheism Stance Corpus."""
    def __init__(self):
        """Constructor.
        
        Implements list of possible transformers (feature extractors) and 
        estimators (classifiers). (Un)comment to exclude/include in training.
        """

        logger.info("\n")
        model_attributes = ["clf", "accuracy", "micro_f1", "cv_mean", "cv_std"]
        self.model = namedtuple("Model", model_attributes)
        
        # Use this when using custom preprocessing/tokenizing
        countvec = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
        # TF-IDF on POS tags
        pos_tfidf = TfidfVectorizer(tokenizer=self._pos_tagger, preprocessor=dummy)
        # List of available transformers/features
        self.transformers = [
                                ("bow", countvec),
                                ("pos_tfidf", pos_tfidf),
                                ("polarity", AtheismPolarityExtractor()),
                                ("length", AverageWordLengthExtractor()),
                                ("ner", NamedEntityExtractor()),
                                ("twitter", TwitterFeaturesExtractor())
                            ]
        self.estimator = svm.SVC(kernel="linear", class_weight="balanced")
        
    def train(self, train_file, target, test_model=False):
        """Trains (and tests) a model for one class.

        Args:
            train_file (string): Path to the corpus file.
            target (string): Class name, e.g. atheism.
            test_model (bool, optional): Whether part of the input data should 
            be used for testing/evaluation. Defaults to False.

        Returns:
            # TODO
            namedtuple: model tuple (sklearn.Pipeline, accuracy, micro f1, 
                                     cross validation mean, cross validation std)
        """

        # Import data
        train_data = self._read_file(train_file, target)
        # FIXME smallest possible dataset for splitting => 8:3, classes [1, 0, -1]
        # train_data = train_data[:11]
        X_test = y_test = None
        
        # Split dataset if classifier should be tested
        if test_model:
            # FIXME remove seed
            train_data, test_data = train_test_split(train_data, test_size=0.2, 
                                                     stratify=train_data[target], random_state=21)
            X_test = list(test_data["text"])
            y_test = list(test_data[target])
        X_train = list(train_data["text"])
        y_train = list(train_data[target])
        
        logger.info(f"Training start")
        
        baseline_model = self.train_baseline_model(X_train, y_train, X_test, y_test)
        
        ppl = self._build_pipeline(self.estimator, self.transformers)
        ppl = self.evaluate_gridsearch(ppl, X_train, y_train)
        # ppl.fit(X_train, y_train)
        # cv_scores = self.evaluate_cv(ppl, X_train, y_train)
        # print(cv_scores.mean(), cv_scores.std())
        return ppl
        
    def train_baseline_model(self, X_train, y_train, X_test=None, y_test=None):
        """Trains a baseline model as described in Wojatzki & Zesch (2016): 
        Linear SVM with word and character bag of words features. #TODO
        
        Evalutes the model if test data supplied.

        Args:
            X_train (list): List of lists of document strings. #FIXME
            y_train (list): List of class labels, e.g. -1/0/1.
            X_test (list optional): See X_train. Defaults to None.
            y_test (list, optional): See y_train. Defaults to None.

        Returns:
            namedtuple: model tuple (sklearn.Pipeline, accuracy, micro f1, 
                                     cross validation mean, cross validation std)
        """
        
        logger.info("Training baseline model...")
        # Baseline classifier: Linear SVM
        baseline_clf = svm.SVC(kernel="linear")
        # Baseline features: Simple word/character n-grams
        baseline_transformers = [("bow", CountVectorizer(ngram_range=(1,3))),
                                 ("boc", CountVectorizer(ngram_range=(2,5), 
                                                         analyzer="char"))]
        # Build and train model
        baseline_ppl = self._build_pipeline(baseline_clf, baseline_transformers, 
                                            preprocessing=False)
        baseline_ppl.fit(X_train, y_train)
        
        # Cross validation
        cv_scores = self.evaluate_cv(baseline_ppl, X_train, y_train)
        
        # If testing model, evaluate
        accuracy = f1 = None
        if X_test and y_test:
            accuracy, f1 = self.evaluate_metrics(baseline_ppl, X_test, y_test)
        baseline_model = self.model(baseline_ppl, accuracy, f1, cv_scores.mean(), cv_scores.std())
        logger.info("... Training baseline model finished")
        return baseline_model
    
    def evaluate_metrics(self, pipeline, X_test, y_test):
        """Evaluates a fitted estimator/pipeline on a test data set.

        Args:
            pipeline (sklearn estimator|pipeline): A fitted model.
            X_test (list): List of document strings. #FIXME
            y_test (list): List of class labels, e.g. -1/0/1.

        Returns:
            float, float: Accuracy and micro f1 scores.
        """

        pred = pipeline.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Accuracy on test set {accuracy:.3f}, micro f1 {f1:.3f}")
        return accuracy, f1
    
    def evaluate_cv(self, pipeline, X, y, k=10):
        """Evaluates a fitted estimator/pipeline using cross-validation.

        Args:
            pipeline (sklearn estimator|pipeline): A fitted or unfitted model.
            X (list): List of document strings. #FIXME
            y (list): List of class labels, e.g. -1/0/1.
            k (int, optional): Number of folds. Defaults to 10.

        Returns:
            ndarray: Output of sklearn.model_selection.cross_val_score()
        """
        
        # FIXME k=10
        cv_scores = cross_val_score(pipeline, X, y, cv=k, scoring="f1_micro")
        logger.info(f"{k}-fold Cross-validation micro f1 mean {cv_scores.mean():.3f}, std {cv_scores.std():.3f}")
        return cv_scores
    
    def evaluate_gridsearch(self, pipeline, X, y, k=10):
        """Implements sklearn gridsearch to find the best model given the 
        parameters in the constructor.
        
        Args:
            pipeline (sklearn Pipeline): Pipeline to be searched.
            X (list): List of document strings. #FIXME
            y (list): List of class labels, e.g. -1/0/1.
            k (int, optional): Number of folds. Defaults to 10.

        Returns:
            # FIXME
            sklearn Pipeline: The best pipeline.
        """
        
        features = pipeline["features"].transformer_list
        features_search_space = [features[:i] for i in range(1,len(features)+1)]
        search_space = {"features__bow__ngram_range": [(1,1), (1,2), (1,3)],
                        "features__bow__min_df": [0.01, 0.05, 0.1],
                        "features__bow__max_df": [0.75, 0.9, 1.0],
                        "features__transformer_list": features_search_space}
        
        if len(features_search_space) > 3 or len(search_space) > 3 or k > 3:
            print(f"WARNING: Fitting {k} folds on more than 3 parameter candidates. \
                    This could take a long time.")
        # FIXME cv=10?
        # FIXME verbose=0
        grid_search = GridSearchCV(pipeline, search_space, scoring="f1_micro", cv=k, verbose=2)
        t0 = time()
        grid_search.fit(X, y)
        # TODO logging
        print(f"Grid search done in {(time()-t0):.3f}")
        print()

        print(f"Best micro f1 score: {grid_search.best_score_:.3f}")
        print("Best parameters:")
        best_params = grid_search.best_estimator_.get_params()
        for param_name in sorted(search_space.keys()):
            print("\t%s: %r" % (param_name, best_params[param_name]))
            
        # FIXME
        return grid_search.best_estimator_
    
    def save_model(self, estimator, fn):
        """Pickles a model.

        Args:
            estimator (sklearn.pipeline.Pipeline): The estimator/pipeline to pickle.
            fn (string): Path where it should be saved.
        """
        
        joblib.dump(estimator, fn, compress = 1)
        logger.info(f"Dumped model in {fn}")
        
    def load_model(self, fn):
        """Loads a pickled model.

        Args:
            fn (string): Path to the pickle jar.

        Returns:
            sklearn.pipeline.Pipeline: The model
        """
        
        logger.info(f"Loaded model from {fn}")
        return joblib.load(fn)
    
    def _read_file(self, fn, target):
        """Uses pandas to read and validate the input data from a file.

        Args:
            fn (string): Path to input data.
            target (string): Class name of the target class, e.g. "atheism".

        Raises:
            ValueError: If target is not a column in the data.

        Returns:
            pandas.DataFrame: DF with columns "id", "text", target.
        """

        df = read_table(fn)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' is not a column in the data")
        logger.info(f"Read data from {fn}: {df.shape} dataframe")
        # Drop all columns except id, text, atheism stance
        return df[["id", "text", target]]
    
    def _pos_tagger(self, tokenized_data):
        """Uses nltk.pos_tag() to tag a tokenized document.

        Args:
            tokenized_data (list): List of tokens.

        Returns:
            list: List of tokens concatenated with list of POS tags.
        """
        
        # https://stackoverflow.com/a/33305005
        return tokenized_data + [tag for _, tag in pos_tag(tokenized_data)]
    
    def _build_pipeline(self, clf, transformers, preprocessing=True):
        ""
        
        if not transformers:
            raise TypeError("Need transformer to build a pipeline")
        
        if preprocessing:
            return Pipeline([("preprocessing", Preprocessor()),
                             ("features", FeatureUnion(transformers)), 
                             ("clf", clf)])
        return Pipeline([("features", FeatureUnion(transformers)), 
                         ("clf", clf)])
    

if __name__ == "__main__":
    f = "data/corpus.txt"
    clf = ASCClassifier()
    
    # Run automatically for different targets
    # targets = ["atheism", "secularism", "religious_freedom", "freethinking",
    #            "no_evidence", "supernatural", "christianity", "afterlife", 
    #            "usa", "islam", "conservatism", "same_sex_marriage"]
    # targets = ["atheism", "supernatural", "christianity", "islam"]
    targets = ["atheism"]
    testing = True
    for t in targets:
        print(t.upper())
        model = clf.train(f, t, test_model=testing)
        print() 
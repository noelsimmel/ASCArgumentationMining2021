from collections import namedtuple
import logging
from nltk import pos_tag
from pandas import read_table
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time

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
    ""
    # https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
    
    return arg

class ASCClassifier:
    def __init__(self):
        ""

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
        ""

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
        ppl.fit(X_train, y_train)
        # 10-fold cross validation on all available data
        cv_scores = self.evaluate_cv(ppl, X_train, y_train)
        print(cv_scores.mean(), cv_scores.std())
        return ppl
        
    def train_baseline_model(self, X_train, y_train, X_test=None, y_test=None):
        ""
        
        logger.info("Training baseline model...")
        # Baseline classifier: Linear SVM with bag of words features
        baseline_clf = svm.SVC(kernel="linear", class_weight="balanced")
        # Baseline feature: Simple bag of words
        baseline_transformer = [("bow", CountVectorizer())]
        # Build and train model
        baseline_ppl = self._build_pipeline(baseline_clf, baseline_transformer, preprocessing=False)
        baseline_ppl.fit(X_train, y_train)
        logger.info(f"Baseline model: {baseline_ppl}")
        
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
        ""

        pred = pipeline.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Accuracy on test set {accuracy:.3f}, micro f1 {f1:.3f}")
        return accuracy, f1
    
    def evaluate_cv(self, pipeline, X, y, k=3):
        ""
        
        # FIXME k=10
        cv_scores = cross_val_score(pipeline, X, y, cv=k, scoring="f1_micro")
        logger.info(f"{k}-fold Cross-validation micro f1 mean {cv_scores.mean():.3f}, std {cv_scores.std():.3f}")
        return cv_scores
    
    def evaluate_gridsearch(self, pipeline, X, y):
        ""
        
        features = pipeline["features"].transformer_list
        features_search_space = [features[:i] for i in range(1,len(features)+1)]
        search_space = {#"features__bow__lowercase": [True, False],
                        "features__transformer_list": features_search_space}
        
        # FIXME cv=10?
        # FIXME verbose=0
        grid_search = GridSearchCV(pipeline, search_space, scoring="f1_micro", cv=2, verbose=2)
        t0 = time()
        grid_search.fit(X, y)
        print(f"Grid search done in {(time()-t0):.3f}")
        print()

        print(f"Best micro f1 score: {grid_search.best_score_:.3f}")
        print("Best parameters:")
        best_params = grid_search.best_estimator_.get_params()
        for param_name in sorted(search_space.keys()):
            print("\t%s: %r" % (param_name, best_params[param_name]))
            
        return grid_search.best_estimator_
    
    def _read_file(self, fn, target):
        ""

        df = read_table(fn)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' is not a column in the data")
        logger.info(f"Read data from {fn}: {df.shape} dataframe")
        # Drop all columns except id, text, atheism stance
        return df[["id", "text", target]]

        return train, test
    
    def _pos_tagger(self, tokenized_data):
        ""
        
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
    

if __name__ == '__main__':
    f = 'data/corpus.txt'
    clf = ASCClassifier()
    
    # Run automatically for different targets
    # targets_all = ["atheism", "secularism", "religious_freedom", "freethinking",
    #                "no_evidence", "supernatural", "christianity", "afterlife", 
    #                "usa", "islam", "conservatism", "same_sex_marriage"]
    # targets = ["atheism", "supernatural", "christianity", "islam"]
    targets = ["atheism"]
    testing = True
    for t in targets:
        print(t.upper())
        clf.train(f, t, test_model=testing)
        print()
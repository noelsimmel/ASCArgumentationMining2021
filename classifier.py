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
            train_data, test_data = self._split_dataset(train_data, target, seed=21)
            X_test = list(test_data["text"])
            y_test = list(test_data[target])
        X_train = list(train_data["text"])
        y_train = list(train_data[target])
        
        logger.info(f"Training start")
        # Use this when using custom preprocessing/tokenizing
        countvec = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
        # TF-IDF on POS tags
        pos_tfidf = TfidfVectorizer(tokenizer=self._pos_tagger, preprocessor=dummy)
        
        # # Baseline classifier: Linear SVM with bag of words features
        # baseline_clf = svm.SVC(kernel="linear", class_weight="balanced")
        # baseline_transformers = [("bow", countvec)]
        # baseline_ppl = self.build_pipeline(X_train, y_train, baseline_transformers, baseline_clf)
        # # 10-fold cross validation on all available data
        # cv_scores = self.evaluate_cv(baseline_ppl, X_train, y_train)
        
        # # If testing model, evaluate
        # accuracy = f1 = None
        # if X_test and y_test:
        #     accuracy, f1 = self.evaluate_metrics(baseline_ppl, X_test, y_test)
        # baseline_model = self.model(baseline_ppl, accuracy, f1, cv_scores.mean(), cv_scores.std())
        # print(baseline_model)
        
        
        # another classifier goes here ...
        transformers = [("bow", countvec),
                        ("polarity", AtheismPolarityExtractor()),
                        ("length", AverageWordLengthExtractor()),
                        ("ner", NamedEntityExtractor()),
                        ("twitter", TwitterFeaturesExtractor())]
        clf = svm.SVC(kernel="linear", class_weight="balanced")
        
        ppl = self.build_pipeline(transformers, clf)
        self.evaluate_gridsearch(ppl, X_train, y_train)
        # pipeline.fit(X_train, y_train)
        # logger.info("Fitted pipeline to training data")
        # 10-fold cross validation on all available data
        # cv_scores = self.evaluate_cv(ppl, X_train, y_train)
        # print(cv_scores.mean(), cv_scores.std())
        
    def build_pipeline(self, transformers, clf):
        ""
        
        if transformers == []:
            pipeline = Pipeline([("preprocessing", Preprocessor()),
                                 ("clf", clf)])
        else:
            pipeline = Pipeline([("preprocessing", Preprocessor()),
                                 ("features", FeatureUnion(transformers)), 
                                 ("clf", clf)])
        logger.info(f"Created pipeline: {pipeline.get_params()}")
        
        return pipeline
    
    def evaluate_metrics(self, pipeline, X_test, y_test):
        ""

        pred = pipeline.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, pred)
        f1 = metrics.f1_score(y_test, pred, average="micro")
        logger.info(f"Accuracy on test set {accuracy:.3f}, micro f1 {f1:.3f}")
        return accuracy, f1
    
    def evaluate_cv(self, pipeline, X, y):
        ""
        
        # FIXME cv=10
        cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring="f1_micro")
        logger.info(f"Cross-validation micro f1 mean {cv_scores.mean():.3f}, std {cv_scores.std():.3f}")
        return cv_scores
    
    def evaluate_gridsearch(self, pipeline, X, y):
        ""
        
        transformers = pipeline["features"].transformer_list
        search_space = {#"features__bow__lowercase": [True, False]},
                        "features__transformer_list": [transformers, transformers[:1], transformers[:2], transformers[:3],
                                                       transformers[:4]]}
        
        # FIXME cv=10?
        grid_search = GridSearchCV(pipeline, search_space, scoring="f1_micro", cv=2, verbose=1)
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

    def _split_dataset(self, df, target, seed=None):
        ""

        # Split in train and test sets (80:20)
        train, test = train_test_split(df, test_size=0.2, random_state=seed,
                                           stratify=df[target])
        return train, test
    
    def _pos_tagger(self, tokenized_data):
        ""
        
        # Not included since it raises cv std
        return [token+"/"+tag for token, tag in pos_tag(tokenized_data)]
    

if __name__ == '__main__':
    f = 'data/corpus.txt'
    clf = ASCClassifier()
    # clf.train(f, "atheism", test_model=False)
    
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
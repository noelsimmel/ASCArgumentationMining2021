# classifier.py
# Classification. Using Python 3.7.3.

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
        
        Implements list of possible transformers (feature extractors), estimators 
        (classifiers) and parameters. (Un)comment to exclude/include in training.
        """

        # Use this CountVectorizer when using custom preprocessing/tokenizing
        # You may also add more parameters
        bow = CountVectorizer(tokenizer=dummy, preprocessor=dummy)
        # Feature: TF-IDF on POS tags
        pos_tfidf = TfidfVectorizer(tokenizer=self._pos_tagger, preprocessor=dummy)
        
        # List of available transformers/features
        self.transformers = [
                                ("bow", bow),
                                ("pos_tfidf", pos_tfidf),
                                ("polarity", AtheismPolarityExtractor()),
                                ("length", AverageWordLengthExtractor()),
                                ("ner", NamedEntityExtractor()),
                                ("twitter", TwitterFeaturesExtractor())
                            ]
        
        # Parameters for grid search. Only concern BOW/CountVectorizer
        self.search_space = {
                             "features__bow__ngram_range": [(1,1), (1,2), (1,3)],
                             "features__bow__min_df": [0.01, 0.05, 0.1],
                             "features__bow__max_df": [0.75, 0.9, 1.0]
                             }
        
        # Classifier
        self.estimator = svm.SVC(kernel="linear", class_weight="balanced")
        
        # The trained model (= fitted pipeline) (result of self.train())
        self.model = None
        
    def train(self, train_file, target, test_model=False, gridsearch=False):
        """Trains (and tests) a model for one class. Saves the model as self.model.

        Args:
            train_file (string): Path to the corpus file.
            target (string): Class name, e.g. atheism.
            test_model (bool, optional): Whether part of the input data should 
            be used for testing/evaluation. Defaults to False.

        Returns:
            sklearn.pipeline.Pipeline: The fitted baseline model.
        """

        # Import data
        X_train, y_train, X_test, y_test = self._split_dataset(train_file, target, test_model)
        logger.info(f"Training start...")
        # Build and fit model
        ppl = self._build_pipeline(self.estimator, self.transformers)
        ppl.fit(X_train, y_train)
        logger.info("...Training finished")
        
        # Cross-validate
        cv_scores = self.evaluate_cv(ppl, X_train, y_train)
        
        # Grid search (warning: may take a long time if using many transformers)
        if gridsearch:
            gs_ppl = self.gridsearch(ppl, X_train, y_train)
            if X_test and y_test:
                logger.info("Evaluating grid search model:")
                self.evaluate_metrics(gs_ppl, X_test, y_test)
                logger.info("Evaluating actual model:")
        
        # Evaluate model on test set
        if X_test and y_test:
            self.evaluate_metrics(ppl, X_test, y_test)
            # Re-fit on all data
            ppl.fit(X_train + X_test, y_train + y_test)
            
        self.model = ppl
        return ppl
        
    def train_baseline_model(self, train_file, target, test_model=False):
        """Trains a baseline model as described in Wojatzki & Zesch (2016): 
        Linear SVM with word and character ngram features.
        
        Evalutes the model if test data supplied.

        Args:
            X_train (list): List of document strings.
            y_train (list): List of class labels, e.g. -1/0/1.
            X_test (list optional): See X_train. Defaults to None.
            y_test (list, optional): See y_train. Defaults to None.

        Returns:
            sklearn.pipeline.Pipeline: The fitted baseline model.
        """
        
        logger.info("Training baseline model...")
        # Import data
        X_train, y_train, X_test, y_test = self._split_dataset(train_file, target, test_model)
            
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
        
        # Cross-validate
        cv_scores = self.evaluate_cv(baseline_ppl, X_train, y_train)
        
        # If testing model, evaluate
        if X_test and y_test:
            accuracy, f1 = self.evaluate_metrics(baseline_ppl, X_test, y_test)
            # Re-fit on all data
            baseline_ppl.fit(X_train + X_test, y_train + y_test)
            
        logger.info("... Training baseline model finished")
        return baseline_ppl
    
    def evaluate_metrics(self, pipeline, X_test, y_test):
        """Evaluates a fitted estimator/pipeline on a test data set.

        Args:
            pipeline (sklearn estimator|pipeline): A fitted model.
            X_test (list): List of document strings.
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
            X (list): List of document strings.
            y (list): List of class labels, e.g. -1/0/1.
            k (int, optional): Number of folds. Defaults to 10.

        Returns:
            ndarray: Output of sklearn.model_selection.cross_val_score()
        """
        
        cv_scores = cross_val_score(pipeline, X, y, cv=k, scoring="f1_micro")
        mean = cv_scores.mean()
        if not mean <= 0 and not mean > 0:
            raise ValueError("Cross-validation failed: Class underpopulated.",
                             "Try using less folds or more class instances.")
        logger.info(f"{k}-fold Cross-validation micro f1 mean {mean:.3f}, std {cv_scores.std():.3f}")
        return cv_scores
    
    def gridsearch(self, pipeline, X, y, k=5):
        """Implements sklearn grid search to find the best model given the 
        parameters in the constructor.
        
        Args:
            pipeline (sklearn.pipeline.Pipeline): Pipeline to be searched.
            X (list): List of document strings.
            y (list): List of class labels, e.g. -1/0/1.
            k (int, optional): Number of folds. Defaults to 5.

        Returns:
            sklearn.pipeline.Pipeline: The best performing pipeline.
        """
        
        logger.info("Grid search...")
        # Features search space implements forward elimination:
        # Train + evaluate on one transformer, then two, etc.
        features = pipeline["features"].transformer_list
        features_search_space = [features[:i] for i in range(1,len(features)+1)]
        # Merge constructor search space + features search space
        search_space = {**self.search_space, 
                        **{"features__transformer_list": features_search_space}}
        
        if (len(features_search_space) > 3 or len(search_space) > 3) and k > 5:
            print(f"WARNING: Fitting {k} folds on more than 3 parameter candidates. \
                             This could take a long time.")
            
        time0 = time()
        grid_search = GridSearchCV(pipeline, search_space, scoring="f1_micro", cv=k)
        grid_search.fit(X, y)
        
        logger.info(f"... Grid search finished in {(time()-time0):.3f} seconds")
        logger.info(f"Best micro f1 score: {grid_search.best_score_}")
        logger.info("Best estimator:")
        logger.info(grid_search.best_estimator_)
        
        return grid_search.best_estimator_
    
    def predict(self, data, fn=None):
        ""
        
        if not self.model:
            raise TypeError("Model is not fitted yet. \
                              Call train() on training data first \
                              or load a pickled model with load_model().")
            
        logger.info("Making predictions...")
        predictions = self.model.predict(data)
        assert len(data) == len(predictions)
        if fn: 
            with open(fn, mode="w+") as f:
                f.write("ID\tPREDICTION\tTEXT")
                for idx, pred in enumerate(predictions):
                    f.write(f"{str(idx)}\t{str(pred)}\t{data[idx]}\n")
        logger.info("... Predictions done")
        return predictions
    
    def save_model(self, fn, estimator=None):
        """Pickles a model.

        Args:
            fn (string): Path where it should be saved (e.g. .pkl file).
            estimator (sklearn.pipeline.Pipeline, optional): The estimator/pipeline 
            to pickle. Saves self.model by default.

        Raises:
            TypeError: If no estimator is supplied self.model=None.
        """
        
        if not estimator:
            estimator = self.model
        if not estimator:
            raise TypeError("Nothing to pickle. \
                             Please supply an estimator or train self.model.")
        joblib.dump(estimator, fn, compress = 1)
        logger.info(f"Dumped model in {fn}")
        
    def load_model(self, fn):
        """Loads a pickled model and saves it in self.model.

        Args:
            fn (string): Path to the pickle jar.

        Returns:
            sklearn.pipeline.Pipeline: The model
        """
        
        logger.info(f"Loaded model from {fn}")
        model = joblib.load(fn)
        self.model = model
        return model
    
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
        if len(df) == 0:
            raise TypeError(f"File {fn} is empty.")
        if target not in df.columns:
            raise ValueError(f"Target '{target}' is not a column in the data.\nColumns are: {list(df.columns)}")
        logger.info(f"Read data from {fn}: {len(df)} instances")
        # Drop all columns except id, text, atheism stance
        return df[["id", "text", target]]
    
    def _split_dataset(self, fn, target, test_model=False):
        """Read file and make X and y datasets. Also split in train/test set 
        if desired.

        Args:
            fn (string): Path to input data.
            target (string): Classification target/column name, e.g. "atheism".
            test_model (bool, optional): Whether to make a train/test split. 
            Defaults to false.

        Returns:
            list, list, list, list: Train and test sets. Test set may be None.
        """
        
        train_data = self._read_file(fn, target)
        X_train = list(train_data["text"])
        y_train = list(train_data[target])
        n_classes = len(set(y_train))
        X_test = y_test = None
        if n_classes < 2:
            raise ValueError(f"Target '{target}' only has {n_classes} class ({y_train[0]}). \
                               Needs at least 2 for classification.")
        
        # Split dataset if classifier should be tested
        if test_model:
            # FIXME remove seed
            train_data, test_data = train_test_split(train_data, test_size=0.2, 
                                                    stratify=train_data[target], random_state=21)
            X_train = list(train_data["text"])
            y_train = list(train_data[target])
            X_test = list(test_data["text"])
            y_test = list(test_data[target])
        return X_train, y_train, X_test, y_test
    
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
        """Builds an sklearn pipeline consisting of preprocessor, transformers, 
        and an estimator. Does NOT perform fit or transform.

        Args:
            clf (estimator): An sklearn estimator such as sklearn.svm.SVC.  
            transformers (list): List of transformers, see constructor.
            preprocessing (bool, optional): Whether the pipeline should include 
            preprocessing from the Preprocessor class. Defaults to True.

        Raises:
            TypeError: If no transformers are supplied. (Data needs to be 
            transformed to numeric values before classification)

        Returns:
            sklearn.pipeline.Pipeline: The unfitted pipeline.
        """
        
        if not transformers:
            raise TypeError("Need transformer to build a pipeline")
        
        if preprocessing:
            # FeatureUnion means that the transformations are applied 
            # simultaneously instead of consecutively
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
        model = clf.train(f, t, test_model=testing, gridsearch=True)
        print() 
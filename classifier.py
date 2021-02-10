import logging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
file_handler = logging.FileHandler('logfile.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class ASCClassifier:
    def __init__(self, fn):
        """
        Constructor
        """

        self.data = self._read_data(fn)

    def split_dataset(self, df):
        """
        Split data into training and test sets (80:20). 
        Save datasets to tsv files.

        - df (dataframe): [Preprocessed] data

        Return training and test sets as dataframes.
        """

        # Atheism value counts: 
        # none 220, favor 147, against 348, mean -0.281119 (n=715)
        # Split in train and test sets (80:20)
        train, test = train_test_split(df, test_size=0.2, stratify=df.atheism)
        train.to_csv('data/dataset_training.tsv', sep="\t")
        test.to_csv('data/dataset_test.tsv', sep="\t")
        logger.info("Saved datasets to data folder in tsv format")

    def train(self):
        """
        """

        # Daten einlesen, wenn nötig
        if type(input_data) == str: input_data = self._preprocess(input_data)
        logger.info(f"Beginn Training ({input_data.shape[0]} Zeilen)")
        # Features extrahieren und an den DF anhängen
        features_only = self._extract_features(input_data)
        features = pd.merge(input_data, features_only, on=['user_id'])
        logger.info(f"Ende Training: {features.shape[0]} Zeilen, {features.shape[1]} Spalten")

        # Features aggregieren, d.h. für jede Klasse über alle Instanzen mitteln
        agg_features = self._aggregate_features(features)
        # Aggregierte Features in tsv-Datei schreiben
        # Habe tsv statt json gewählt, weil es für Menschen besser lesbar ist
        agg_features.to_csv(output_filename, sep='\t')
        logger.info(f"Aggregierte Features in {output_filename} geschrieben")
        return agg_features
    
    def predict(self):
        """
        """

        pass

    def evaluate(self):
        """
        """

        pass

    def _read_data(self, fn):
        """
        Read data from tab-separated txt/csv/tsv file and preprocess.

        - fn (str): Path to [preprocessed] corpus file

        Return dataframe
        """

        df = pd.read_csv(fn, sep="\t")
        # Replace columns for easier access
        df.columns = ["id", "text", "atheism", "secularism", "religious_freedom", "freethinking", 
                    "no_evidence", "supernatural", "christianity", "afterlife", "usa", "islam",
                    "conservatism", "same_sex_marriage"]
        # Represent stances as ints 0, 1, -1
        df.fillna(0, inplace=True)
        df.replace(["none", "favor", "against"], [0, 1, -1], inplace=True)
        logger.info(f"Read data from {fn}: {df.shape} dataframe")
        return df

    def _extract_features(self):
        """
        """

        pass


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    clf = MBTIClassifier()
    f = 'data/corpus_preprocessed'
    clf.split_dataset(f)
    # clf.train('data/dataset_training.json', 'model.tsv')
    # clf.evaluate('data/dataset_validation.json', 'model.tsv')
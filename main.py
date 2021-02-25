# main.py
# Main file for interactive use

import sys
from time import time
from classifier import ASCClassifier

def start_train_mode(data_filename, target, model_filename, test_model=False, gridsearch=False):
    """Trains and pickles the model.

    Args:
        data_filename (string): Path to input data.
        target (string): Class label to train on, e.g. "atheism".
        model_filename (string): Path where model should be saved.
        test_model (bool, optional): Whether model should be tested (Flag -t). Defaults to False.
        gridsearch (bool, optional): Whether grid search should be performed to 
        find the best model (Flag -g). Warning: This may take a long time. Defaults to False.
    """

    _validate_filenames([data_filename, model_filename])
    time0 = time()
    clf = ASCClassifier()
    model = clf.train(data_filename, target, 
                      test_model=test_model, gridsearch=gridsearch)
    clf.save_model(model_filename, model)
    print(f"Executed in {time()-time0} seconds")

def start_baseline_mode(data_filename, target, model_filename, test_model=False):
    """Trains and pickles the baseline model specified in Mohammad et al. (2016).

    Args:
        data_filename (string): Path to input data.
        target (string): Class label to train on, e.g. "atheism".
        model_filename (string): Path where model should be saved.
        test_model (bool, optional): Whether model should be tested (Flag -t). Defaults to False.
    """
    
    _validate_filenames([data_filename, model_filename])
    time0 = time()
    clf = ASCClassifier()
    model = clf.train_baseline_model(data_filename, target, test_model=test_model)
    clf.save_model(model_filename, model)
    print(f"Executed in {time()-time0} seconds")
    
def start_predict_mode(data_filename, model_filename, output_filename=None):
    """Loads a pickled model and performs predictions.

    Args:
        data_filename (string): Path to the input data.
        model_filename (string): Path to the pickled model.
        output_filename (string, optional): Path where the predictions should be 
        saved (as .txt). If None, predictions are printed to console. Defaults to None.
    """

    _validate_filenames([data_filename, model_filename])
    if output_filename:
        _validate_filenames([data_filename, model_filename, output_filename])
    time0 = time()
    clf = ASCClassifier()
    clf.load_model(model_filename)
    predictions = clf.predict(data_filename, output_filename)
    if not output_filename:
        print(predictions)
    print(f"Executed in {time()-time0} seconds")
    
def _validate_filenames(filenames):
    """Performs a quick check to see if the supplied filenames are valid. 
    Prints usage information and exits if not.

    Args:
        filenames (list): Iterable of filename strings to check.
    """
    
    for f in filenames:
        if len(f) < 3 or "." not in f:
            print(f"'{f}' is not a valid filename")
            _print_usage()
            
def _print_usage():
    """Print usage information to the console and exit the program.
    """
    
    print("TRAIN MODE: python3 main.py data target model -b -t -g")
    print("PREDICT MODE: python3 main.py data model [output] -p")
    sys.exit()
    
def main(argv):
    """Main function. Validates the user input and makes function calls.

    Args:
        args (list): List of sys.argv arguments.
    """
    
    argc = len(argv)
    if argc < 4: _print_usage()
        
    data = argv[1]
    target = argv[2]
    model = argv[3]
    
    if "-p" in argv:
        print("Starting predict mode (ignoring other flags)")
        model = argv[2]
        if argc > 4: 
            output = argv[3]
            start_predict_mode(data, model, output)
        else:
            print("No output path supplied, will print predictions to console")
            start_predict_mode(data, model)
    elif "-b" in argv and argc > 4:
        print("Starting baseline mode")
        if "-g" in argv:
            print("ERROR: Grid search is not available for baseline model")
            sys.exit()
        if "-t" in argv:
            print("Testing enabled")
            start_baseline_mode(data, target, model, test_model=True)
        else:
            start_baseline_mode(data, target, model)
    elif "-t" in argv and argc > 4:
        print("Starting train mode")
        print("Testing enabled")
        if "-g" in argv:
            print("Grid search enabled")
            start_train_mode(data, target, model, test_model=True, gridsearch=True)
        else:
            start_train_mode(data, target, model, test_model=True)
    elif "-g" in argv and argc > 4:
        print("Starting train mode")
        print("Grid search enabled")
        start_train_mode(data, target, model, gridsearch=True)
    elif argc == 4:
        print("Starting train mode")
        start_train_mode(data, target, model)
    else:
        _print_usage()
        
if __name__ == "__main__":
    main(sys.argv)
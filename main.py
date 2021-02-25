# main.py
# Main file for interactive use

import sys
from classifier import ASCClassifier

def start_train_mode(data_filename, target, model_filename, test_model=False, gridsearch=False):
    ""

    clf = ASCClassifier()
    model = clf.train(data_filename, target, 
                      test_model=test_model, gridsearch=gridsearch)
    clf.save_model(model_filename, model)

def start_baseline_mode(data_filename, target, model_filename, test_model=False):
    ""
    
    clf = ASCClassifier()
    model = clf.train_baseline_model(data_filename, target, test_model=test_model)
    clf.save_model(model_filename, model)
    
def start_predict_mode(data_filename, output_filename, model_filename):
    ""

    clf = ASCClassifier()
    clf.load_model(model_filename)
    clf.predict(data_filename, output_filename)
    
def main(argv):
    """Main function. Validates the user input and makes function calls.

    Args:
        args (list): List of sys.argv arguments.
    """
    
    argc = len(argv)
    # Catch invalid input
    if argc < 4:
        print("TRAIN MODE: python3 main.py data target model -b -t -g")
        print("PREDICT MODE: python3 main.py data output model -p")
        sys.exit()
        
    data = argv[1]
    target = argv[2]
    model = argv[3]
    
    if "-p" in argv:
        print("Starting predict mode (ignoring other flags)")
        # target = output file
        start_predict_mode(data, target, model)
    elif "-b" in argv:
        print("Starting baseline mode")
        if "-g" in argv:
            print("ERROR: Grid search is not available for baseline model")
            sys.exit()
        if "-t" in argv:
            print("Testing enabled")
            start_baseline_mode(data, target, model, test_model=True)
        else:
            start_baseline_mode(data, target, model)
    elif "-t" in argv:
        print("Starting train mode")
        print("Testing enabled")
        if "-g" in argv:
            print("Grid search enabled")
            start_train_mode(data, target, model, test_model=True, gridsearch=True)
        else:
            start_train_mode(data, target, model, test_model=True)
    elif "-g" in argv:
        print("Starting train mode")
        print("Grid search enabled")
        start_train_mode(data, target, model, gridsearch=True)
    elif argc == 4:
        print("Starting train mode")
        start_train_mode(data, target, model)
    else:
        print("TRAIN MODE: python3 main.py data target model -b -t -g")
        print("PREDICT MODE: python3 main.py data model output -p")
        sys.exit()
        
if __name__ == "__main__":
    main(sys.argv)
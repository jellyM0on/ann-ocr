from utils import load_training_data
from gui import run_gui

if __name__ == "__main__":
    print("[LOG] Loading training data and launching GUI...")

    training_file = "training.xlsx"

    X_train, Y_train, ascii_to_index, index_to_ascii = load_training_data(training_file)

    run_gui(X_train, Y_train, ascii_to_index, index_to_ascii)

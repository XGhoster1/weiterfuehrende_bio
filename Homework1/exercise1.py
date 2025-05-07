import numpy as np
import json
from typing import Dict, List, Tuple
import os, sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

AMINO_ACIDS = "ACEDGFIHKMLNQPSRTWVY"
LABELS = {"H": 0, "E": 1, "C": 2}


#input_file = "./data/input.jsonl"

def load_jsonl(filename: str)-> List[Dict]:
    '''This function accept a file/pathname to a jsonl file and return a list of dictionary, where every dictionary represent a json document. 
    I. e. the key/value from the json documents are converted to a Python dictionary'''
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data

def single_one_hot_encode(amino_acid: "str") -> np.array:
    '''A single amino acid is encoded in a numpy array as shown in the lecture. The index for the amino acid in the given string above (AMINO_ACIDS)
    is set to 1 all other value to 0.'''
    if len(amino_acid) != 1:
        raise ValueError("single_one_hot_encode expects exactly one amino‑acid letter.")
    aa = amino_acid.upper()
    index = AMINO_ACIDS.find(aa)
    if index == -1:
        raise ValueError(f"Invalid amino acid: {aa}")
    vector = np.zeros(20, dtype=np.int8)
    vector[index] = 1
    return vector

def one_hot_encode_sequence(sequence:str, window_size=5) -> np.array:
    '''This function takes a sequence of amino acids and converts it into a 2D Numpy array
    representing the one-hot encoding. It has len(sequence) - 2*window_size rows and 20*(window_size*2 + 1) columns.
    '''
    seq = sequence.upper()
    L = len(seq)
    window_len = 2 * window_size + 1
    encoded = []

    for i in range(window_size, L - window_size):
        window = seq[i - window_size : i + window_size + 1]
        win_vec = np.zeros((20, window_len), dtype=np.int8)
        for j, aa in enumerate(window):
            idx = AMINO_ACIDS.find(aa)
            if idx == -1:
                raise ValueError(f"Invalid amino acid '{aa}' in window '{window}'")
            win_vec[idx, j] = 1

        encoded.append(win_vec.flatten(order="F"))

    return np.asarray(encoded, dtype=np.int8)

def one_hot_encode_labeled_sequence(entry: Dict, window_size=5) -> Tuple[np.array, np.array]:
    '''
    Returns:
        Tuple:
            - X: 2D one-hot encoded array of shape (n_valid_residues, 20 * window_length)
            - y: 1D array of labels (0, 1, 2) for H, E, C
    '''
    seq   = entry["sequence"].upper()
    label = entry["label"].upper()
    res   = entry["resolved"]              
    window_len = 2 * window_size + 1

    X, y = [], []

    for i in range(window_size, len(seq) - window_size):
        centre_resolved = (res[i] == '1')
        if not centre_resolved or label[i] not in LABELS:
            continue

        win_vec = np.zeros((20, window_len), dtype=np.int8)
        for j, aa in enumerate(seq[i - window_size : i + window_size + 1]):
            idx = AMINO_ACIDS.find(aa)
            if idx == -1:
                break
            win_vec[idx, j] = 1
        else:                                    
            X.append(win_vec.flatten(order="F")) 
            y.append(LABELS[label[i]])

    return np.asarray(X, dtype=np.int8), np.asarray(y, dtype=np.int8)

def predict_secondary_structure(input: np.array, labels: np.array, size_hidden=10) -> Tuple[float, float, float]:
    '''
    Trains an MLPClassifier on the input features and labels.
    
    Returns:
        Tuple of (accuracy, precision, recall), all macro-averaged.
    '''
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.2, random_state=42)

    # Create and train the MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(size_hidden,), random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    return accuracy, precision, recall


def calculate_Q3(prediction: str, truth: str) -> Tuple[float, float, float]:
    '''
    Computes the per-class accuracy (Q3 score) for H, E, and C states.
    
    Parameters:
        prediction (str): predicted secondary structure string (H/E/C)
        truth (str): ground-truth secondary structure string (H/E/C)

    Returns:
        Tuple[float, float, float]: (Q3_H, Q3_E, Q3_C)
    '''
    if len(prediction) != len(truth):
        raise ValueError("Prediction and truth must have the same length.")

    scores = {}
    for state in "HEC":
        total = sum(1 for t in truth if t == state)
        correct = sum(1 for p, t in zip(prediction, truth) if t == state and p == t)
        scores[state] = correct / total if total else float("nan")

    return scores["H"], scores["E"], scores["C"]

if __name__ == "__main__":
    input_file = "./data/input.jsonl"
    #entries = load_jsonl(input_file)
    #print(len(entries))
    seq = "ACDEFGHIKLMNPQRSTVWY"  # All 20 amino acids
    result = one_hot_encode_sequence(seq, window_size=2)
    print(result.shape)
    
    # extend as you need
    pass

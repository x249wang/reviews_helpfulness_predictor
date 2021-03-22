import numpy as np


def load_labels(filepath):
    with open(filepath, "r") as f:
        labels = f.readlines()
        labels = [int(label.strip()) for label in labels]
    return labels


def generate_predicted_labels(act, scores):
    positive_proportion = np.mean(np.array(act) == 1)
    score_threshold = np.quantile(scores, q=1 - positive_proportion)

    pred = scores > score_threshold
    return pred
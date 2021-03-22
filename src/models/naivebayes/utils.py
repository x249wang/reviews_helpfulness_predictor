import numpy as np

# Loads labels from a text file as a list, converting the "0" and "1"'s into integer type
def load_labels(filepath):
    with open(filepath, "r") as f:
        labels = f.readlines()
        labels = [int(label.strip()) for label in labels]
    return labels


# Computes the predicted label from model scores, using threshold picked out to
# match the actual class distribution
def generate_predicted_labels(actual_labels, model_scores):
    positive_proportion = np.mean(np.array(actual_labels) == 1)
    score_threshold = np.quantile(model_scores, q=1 - positive_proportion)

    pred = model_scores > score_threshold
    return pred
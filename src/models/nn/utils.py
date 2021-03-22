import numpy as np

# Load embeddings data (in matrix form, .npy file format)
def load_embeddings_data(filepath):
    return np.load(open(filepath, "rb"))


# Loads labels from a text file as a list, converting the "0" and "1"'s into integer type
def load_labels(filepath):
    with open(filepath, "r") as f:
        labels = f.readlines()
        labels = [int(label.strip()) for label in labels]
    return labels


# Computes the predicted label from model scores, using threshold picked out to
# match the actual class distribution
def generate_predicted_labels(act, scores):
    positive_proportion = np.mean(np.array(act) == 1)
    score_threshold = np.quantile(scores, q=1 - positive_proportion)

    pred = scores > score_threshold
    return pred

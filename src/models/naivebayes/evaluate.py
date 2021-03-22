import joblib
import argparse
from scipy.sparse import load_npz
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import src.models.naivebayes.utils as utils
from src.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF naive bayes model")
    parser.add_argument(
        "--test_label_path", type=str, required=True, help="Path to test label data"
    )
    parser.add_argument(
        "--test_tfidf_path", type=str, required=True, help="Path to test tfidf data"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model object"
    )
    args = parser.parse_args()

    logger.info(f"Evaluating Naive Bayes model on test set from {args.test_tfidf_path}")
    test_features = load_npz(args.test_tfidf_path)
    test_labels = utils.load_labels(args.test_label_path)
    nb_classifier = joblib.load(args.model_path)

    predicted_scores = nb_classifier.predict_proba(test_features)
    predicted_labels = utils.generate_predicted_labels(
        test_labels, predicted_scores[:, 1]
    )

    test_cm = confusion_matrix(test_labels, predicted_labels)
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
    test_acc = accuracy_score(test_labels, predicted_labels)
    test_f1 = f1_score(test_labels, predicted_labels)
    test_auc = roc_auc_score(test_labels, predicted_scores[:, 1])

    logger.info(f"Testing confusion matrix: {test_cm}")
    logger.info(f"Testing accuracy: {test_acc}")
    logger.info(f"Testing F1: {test_f1}")
    logger.info(f"Testing AUC: {test_auc}")

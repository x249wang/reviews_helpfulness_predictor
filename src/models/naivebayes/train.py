# Trains Naive Bayes model and evaluates it on a validation set
#
import numpy as np
import argparse
from scipy.sparse import load_npz
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import joblib
import mlflow
import time
import src.models.naivebayes.utils as utils
import src.models.naivebayes.config as config
from src.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TF-IDF naive bayes model")
    parser.add_argument(
        "--train_label_path", type=str, required=True, help="Path to train label data"
    )
    parser.add_argument(
        "--train_tfidf_path", type=str, required=True, help="Path to train tfidf data"
    )
    parser.add_argument(
        "--val_label_path", type=str, required=True, help="Path to val label data"
    )
    parser.add_argument(
        "--val_tfidf_path", type=str, required=True, help="Path to val tfidf data"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to save model object"
    )
    args = parser.parse_args()

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run() as run:

        start = time.time()

        train_features = load_npz(args.train_tfidf_path)
        train_labels = utils.load_labels(args.train_label_path)

        val_features = load_npz(args.val_tfidf_path)
        val_labels = utils.load_labels(args.val_label_path)

        logger.info(f"Building naive bayes classifier")

        nb_classifier = MultinomialNB()
        nb_classifier.fit(train_features, train_labels)

        predicted_scores = nb_classifier.predict_proba(val_features)
        predicted_labels = utils.generate_predicted_labels(
            val_labels, predicted_scores[:, 1]
        )

        val_cm = confusion_matrix(val_labels, predicted_labels)
        val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
        val_acc = accuracy_score(val_labels, predicted_labels)
        val_f1 = f1_score(val_labels, predicted_labels)
        val_auc = roc_auc_score(val_labels, predicted_scores[:, 1])

        logger.info(f"Validation confusion matrix: {val_cm}")
        logger.info(f"Validation accuracy: {val_acc}")
        logger.info(f"Validation F1: {val_f1}")
        logger.info(f"Validation AUC: {val_auc}")

        joblib.dump(nb_classifier, args.model_path)
        logger.info(f"Model saved to {args.model_path}")

        end = time.time()
        training_time = end - start

        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("tn", val_tn)
        mlflow.log_metric("fp", val_fp)
        mlflow.log_metric("fn", val_fn)
        mlflow.log_metric("tp", val_tp)
        mlflow.log_metric("accuracy", val_acc)
        mlflow.log_metric("f1", val_f1)
        mlflow.log_metric("auc", val_auc)

        mlflow.log_artifact(args.model_path)

        logger.info(f"Took {round(training_time)} seconds for model development")

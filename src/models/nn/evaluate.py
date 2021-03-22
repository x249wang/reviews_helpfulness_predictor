# Evaluates trained feedforward classifier network on held-out test set
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.models.nn.model import ClassifierModel
from src.models.nn.dataset import CustomDataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import src.models.nn.config as config
import src.models.nn.utils as utils
from src.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate feedforward model with BERT embeddings"
    )
    parser.add_argument(
        "--test_label_path", type=str, required=True, help="Path to test label data"
    )
    parser.add_argument(
        "--test_embeddings_path",
        type=str,
        required=True,
        help="Path to test embeddings data",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model object"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dim to feedforward layer"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    test_embeddings = utils.load_embeddings_data(args.test_embeddings_path)
    test_labels = utils.load_labels(args.test_label_path)

    test_dataset = CustomDataset(test_embeddings, test_labels)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    logger.info(f"Evaluating {args.model_path} model on test set")

    model = ClassifierModel(
        config.bert_embeddings_dim,
        args.hidden_dim,
        config.num_labels,
        args.dropout_rate,
    )
    model.load_state_dict(torch.load(args.model_path))

    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    model.eval()
    predicted_scores = np.empty((0, config.num_labels))

    with torch.no_grad():

        for batch in test_dataloader:
            input_data = batch[0].to(device)
            labels = batch[1].to(device)

            output = model(input_data)
            predicted_scores = np.vstack(
                (predicted_scores, F.softmax(output, dim=1).detach().numpy())
            )

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

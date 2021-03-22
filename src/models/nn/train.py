import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import mlflow
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.models.nn.model import ClassifierModel
from src.models.nn.dataset import CustomDataset
import src.models.nn.utils as utils
import src.models.nn.config as config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build feedforward model with BERT embeddings"
    )
    parser.add_argument(
        "--train_label_path", type=str, required=True, help="Path to train label data"
    )
    parser.add_argument(
        "--train_embeddings_path",
        type=str,
        required=True,
        help="Path to train embeddings data",
    )
    parser.add_argument(
        "--val_label_path", type=str, required=True, help="Path to val label data"
    )
    parser.add_argument(
        "--val_embeddings_path",
        type=str,
        required=True,
        help="Path to val embeddings data",
    )
    parser.add_argument(
        "--model_dir_path",
        type=str,
        required=True,
        help="Directory to save model object",
    )
    parser.add_argument(
        "--tensorboard_log_dir_path",
        type=str,
        required=True,
        help="Directory to save tensorboard logs",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--hidden_dim", type=int, default=512, help="Hidden dim to feedforward layer"
    )
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    mlflow.set_experiment(config.experiment_name)
    run = mlflow.start_run()
    run_id = run.info.run_id

    train_embeddings = utils.load_embeddings_data(args.train_embeddings_path)
    train_labels = utils.load_labels(args.train_label_path)

    train_dataset = CustomDataset(train_embeddings, train_labels)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_embeddings = utils.load_embeddings_data(args.val_embeddings_path)
    val_labels = utils.load_labels(args.val_label_path)

    val_dataset = CustomDataset(val_embeddings, val_labels)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    model_dir = f"{args.model_dir_path}/{run_id}"
    model_path = f"{model_dir}/model.pth"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    tensorboard_log_path = f"{args.tensorboard_log_dir_path}/{run_id}"

    tbw = SummaryWriter(log_dir=tensorboard_log_path)

    model = ClassifierModel(
        config.bert_embeddings_dim,
        args.hidden_dim,
        config.num_labels,
        args.dropout_rate,
    )

    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
    else:
        device = "cpu"

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss(reduction="sum")

    # Training loop
    mlflow.log_param("hidden_dim", args.hidden_dim)
    mlflow.log_param("dropout_rate", args.dropout_rate)
    mlflow.log_param("learning_rate", args.lr)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("num_epochs", args.num_epochs)

    start = time.time()

    obs_seen = 0
    best_val_loss = float("inf")

    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        print("==========")

        model.train()
        for step, batch in enumerate(train_dataloader):
            total_train_loss, n_train = 0.0, 0

            optimizer.zero_grad()

            input_data = batch[0].to(device)
            labels = batch[1].to(device)

            n_train += input_data.size(0)
            obs_seen += input_data.size(0)

            output = model(input_data)
            loss = loss_function(output, labels)

            total_train_loss += loss.item()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            tbw.add_scalar("train_loss", loss.item() / input_data.size(0), obs_seen)

        print(f"Train loss: {total_train_loss / n_train}")

        with torch.no_grad():

            model.eval()
            correct_obs, total_val_loss, n_val = 0, 0.0, 0
            val_scores = np.empty((0, config.num_labels))

            for batch in val_dataloader:
                input_data = batch[0].to(device)
                labels = batch[1].to(device)

                n_val += input_data.size(0)

                output = model(input_data)
                val_scores = np.vstack(
                    (val_scores, F.softmax(output, dim=1).detach().cpu().numpy())
                )
                loss = loss_function(output, labels)

                total_val_loss += loss.item()
                pred_labels = output.argmax(dim=1)

                correct_obs += float((labels == pred_labels).sum().item())

            val_loss = total_val_loss / n_val
            tbw.add_scalar("validation_loss", val_loss, obs_seen)

            val_accuracy = correct_obs / n_val
            tbw.add_scalar("validation_accuracy", val_accuracy, obs_seen)

            print(f"Val loss: {val_loss}")
            print(f"Val accuracy: {val_accuracy}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_scores = val_scores
                torch.save(model.state_dict(), model_path)

    best_val_predicted_labels = utils.generate_predicted_labels(
        val_labels, best_val_scores[:, 1]
    )

    val_cm = confusion_matrix(val_labels, best_val_predicted_labels)
    val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
    val_acc = accuracy_score(val_labels, best_val_predicted_labels)
    val_f1 = f1_score(val_labels, best_val_predicted_labels)
    val_auc = roc_auc_score(val_labels, best_val_scores[:, 1])

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

    mlflow.log_artifact(model_path)

    mlflow.end_run()

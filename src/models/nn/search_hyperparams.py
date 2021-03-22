import argparse
from subprocess import check_call
import sys
from itertools import product
import mlflow
import src.models.nn.config as config
from src.logger import logger

PYTHON = sys.executable


def launch_training_job(
    train_label_path,
    train_embeddings_path,
    val_label_path,
    val_embeddings_path,
    model_dir_path,
    tensorboard_log_dir_path,
    batch_size,
    num_epochs,
    lr,
    hidden_dim,
    dropout_rate,
):
    cmd = (
        f"{PYTHON} -m src.models.nn.train "
        f"--train_label_path {train_label_path} "
        f"--train_embeddings_path {train_embeddings_path} "
        f"--val_label_path {val_label_path} "
        f"--val_embeddings_path {val_embeddings_path} "
        f"--model_dir_path {model_dir_path} "
        f"--tensorboard_log_dir_path {tensorboard_log_dir_path} "
        f"--batch_size {batch_size} "
        f"--num_epochs {num_epochs} "
        f"--lr {lr} "
        f"--hidden_dim {hidden_dim} "
        f"--dropout_rate {dropout_rate}"
    )
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        help="Directory to save model objects",
    )
    parser.add_argument(
        "--tensorboard_log_dir_path",
        type=str,
        required=True,
        help="Directory to save tensorboard logs",
    )
    args = parser.parse_args()

    mlflow.set_experiment(config.experiment_name)

    for batch_size, num_epoch, lr, hidden_dim, dropout_rate in product(
        config.batch_sizes,
        config.num_epochs,
        config.learning_rates,
        config.hidden_dims,
        config.dropout_rates,
    ):
        logger.info(
            f"Launching neural network model tuning job for batch_size {batch_size}, "
            f"hidden_dim {hidden_dim}, learning_rate {lr}, "
            f"dropout_rates {dropout_rate}, num_epochs {num_epoch}"
        )

        launch_training_job(
            args.train_label_path,
            args.train_embeddings_path,
            args.val_label_path,
            args.val_embeddings_path,
            args.model_dir_path,
            args.tensorboard_log_dir_path,
            batch_size,
            num_epoch,
            lr,
            hidden_dim,
            dropout_rate,
        )

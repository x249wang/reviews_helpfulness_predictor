import swifter
import pandas as pd
import fasttext
import random
import argparse
import src.data_prep.config as config
from src.logger import logger


def assign_partition(train_pct, val_pct, test_pct):
    rnd_num = random.random()
    if rnd_num < train_pct:
        return "train"
    elif rnd_num > 1 - val_pct:
        return "val"
    elif rnd_num < train_pct + test_pct:
        return "test"
    else:
        return None


def downsample(label, keep_ratio, minority_class):
    return label == minority_class or random.random() < keep_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into train, val and test sets"
    )
    parser.add_argument(
        "--cleaned_data_path", type=str, required=True, help="Path to save cleaned data"
    )
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size")
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to save training data"
    )
    parser.add_argument(
        "--train_label_path",
        type=str,
        required=True,
        help="Path to save training labels",
    )
    parser.add_argument(
        "--val_data_path", type=str, required=True, help="Path to save validation data"
    )
    parser.add_argument(
        "--val_label_path",
        type=str,
        required=True,
        help="Path to save validation labels",
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to save test data"
    )
    parser.add_argument(
        "--test_label_path", type=str, required=True, help="Path to save test labels"
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.05,
        help="Proportion of dataset set aside for validation",
    )
    parser.add_argument(
        "--test_percent",
        type=float,
        default=0.05,
        help="Proportion of dataset set aside for test",
    )
    parser.add_argument(
        "--train_percent",
        type=float,
        default=0.25,
        help="Proportion of dataset set aside for train",
    )
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument(
        "--downsample_ratio",
        type=float,
        default=0.12,
        help="Proportion of majority class to keep to address class imbalance (required if the downsample flag is set)",
    )
    args = parser.parse_args()

    logger.info(
        f"Loading cleaned data from {args.cleaned_data_path} for train/val/test split"
    )
    data_chunks = pd.read_csv(args.cleaned_data_path, chunksize=args.chunksize)

    logger.info(f"Using {args.train_percent*100}% as training data")
    logger.info(f"Using {args.val_percent*100}% as validation data")
    logger.info(f"Using {args.test_percent*100}% as testing data")
    if args.downsample:
        logger.info(
            f"Downsampling training set to {args.downsample_ratio*100}% of majority class"
        )

    for data_chunk in data_chunks:
        data_chunk["partition"] = data_chunk.label.swifter.progress_bar(False).apply(
            lambda x: assign_partition(
                args.train_percent, args.val_percent, args.test_percent
            )
        )

        if args.downsample:
            data_chunk["downsampled"] = data_chunk.label.swifter.progress_bar(
                False
            ).apply(lambda x: downsample(x, args.downsample_ratio, 1))

        train = data_chunk[data_chunk.partition == "train"]
        if args.downsample:
            train = train[train.downsampled == True]
        with open(args.train_data_path, "a") as f:
            train.to_csv(f, header=f.tell() == 0, index=False)
        with open(args.train_label_path, "a") as f:
            f.write("\n".join(train.label.astype(str)) + "\n")

        val = data_chunk[data_chunk.partition == "val"]
        with open(args.val_data_path, "a") as f:
            val.to_csv(f, header=f.tell() == 0, index=False)
        with open(args.val_label_path, "a") as f:
            f.write("\n".join(val.label.astype(str)) + "\n")

        test = data_chunk[data_chunk.partition == "test"]
        with open(args.test_data_path, "a") as f:
            test.to_csv(f, header=f.tell() == 0, index=False)
        with open(args.test_label_path, "a") as f:
            f.write("\n".join(test.label.astype(str)) + "\n")

    logger.info(f"Saved train data to {args.train_data_path}")
    logger.info(f"Saved validation data to {args.val_data_path}")
    logger.info(f"Saved test data to {args.test_data_path}")
    logger.info(f"Saved train labels to {args.train_label_path}")
    logger.info(f"Saved validation labels to {args.val_label_path}")
    logger.info(f"Saved test labels to {args.test_label_path}")

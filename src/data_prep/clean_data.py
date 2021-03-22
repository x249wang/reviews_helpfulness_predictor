import pandas as pd
import fasttext
import argparse
import src.data_prep.utils as utils
import src.data_prep.config as config
from src.logger import logger

lang_detection_model = fasttext.load_model(config.lang_detection_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean reviews dataset")
    parser.add_argument(
        "--raw_data_path", type=str, required=True, help="Path to csv data file"
    )
    parser.add_argument(
        "--cleaned_data_path", type=str, required=True, help="Path to save cleaned data"
    )
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size")
    args = parser.parse_args()

    logger.info(f"Loading parsed csv data from from {args.raw_data_path} for cleaning")

    data_chunks = pd.read_csv(
        args.raw_data_path,
        chunksize=args.chunksize,
    )

    for data_chunk in data_chunks:
        data_chunk["overall"] = utils.convert_to_int(data_chunk["overall"])
        data_chunk["verified"] = utils.convert_to_int(data_chunk.verified)
        data_chunk["unixReviewTime"] = utils.convert_to_int(
            data_chunk["unixReviewTime"]
        )
        data_chunk["has_images"] = utils.is_not_null(data_chunk.image)

        data_chunk["combined_text"] = utils.clean_text(
            data_chunk.reviewText, data_chunk.summary
        )
        data_chunk = data_chunk[utils.is_not_null(data_chunk["combined_text"])]
        data_chunk = data_chunk.drop(columns=config.columns_to_drop)

        data_chunk["language"] = utils.detect_language(
            lang_detection_model, data_chunk["combined_text"]
        )
        data_chunk = data_chunk[
            utils.filter_by_language(data_chunk["language"], ["en"])
        ]

        data_chunk["vote"] = utils.clean_vote_feature(data_chunk["vote"])
        data_chunk["label"] = utils.create_label(data_chunk["vote"])

        with open(args.cleaned_data_path, "a") as f:
            data_chunk.to_csv(f, header=f.tell() == 0, index=False)

    logger.info(f"Cleaned data saved to {args.cleaned_data_path}")

# Parses raw data (.json.gz format) to tabular format (csv file), in chunks
import src.data_prep.config as config
import argparse
import pandas as pd
from src.logger import logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse raw json.gz review data to csv")
    parser.add_argument(
        "--raw_data_path", type=str, required=True, help="Path to raw data file"
    )
    parser.add_argument(
        "--parsed_data_path", type=str, required=True, help="Path to save parsed data"
    )
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size")
    args = parser.parse_args()

    logger.info(f"Reading raw file {args.raw_data_path}")
    data_chunks = pd.read_json(
        args.raw_data_path,
        lines=True,
        chunksize=args.chunksize,
        compression="gzip",
    )

    for data_chunk in data_chunks:
        with open(args.parsed_data_path, "a") as f:
            data_chunk[config.columns_to_keep].to_csv(
                f, header=f.tell() == 0, index=False
            )

    logger.info(f"Parsed csv file saved to {args.parsed_data_path}")

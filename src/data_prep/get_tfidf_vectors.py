import re
import numpy as np
import pandas as pd
import swifter
import spacy
import argparse
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logger


def custom_lemmatized_tokenizer(texts):
    docs = nlp.pipe(texts)

    results = []
    for doc in docs:
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct
        ]
        results.append(tokens)

    return results


def make_corpus(filepath):
    for line in open(filepath, "r"):
        yield line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get tf-idf vector representations")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size")
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to save training data"
    )
    parser.add_argument(
        "--train_lemmatized_path",
        type=str,
        required=True,
        help="Path to save training lemmatized text",
    )
    parser.add_argument(
        "--train_tfidf_path",
        type=str,
        required=True,
        help="Path to save training tf-idf vector representations",
    )
    parser.add_argument(
        "--val_data_path", type=str, required=True, help="Path to save val data"
    )
    parser.add_argument(
        "--val_lemmatized_path",
        type=str,
        required=True,
        help="Path to save val lemmatized text",
    )
    parser.add_argument(
        "--val_tfidf_path",
        type=str,
        required=True,
        help="Path to save val tfidf vector representations",
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to save testing data"
    )
    parser.add_argument(
        "--test_lemmatized_path",
        type=str,
        required=True,
        help="Path to save testing lemmatized text",
    )
    parser.add_argument(
        "--test_tfidf_path",
        type=str,
        required=True,
        help="Path to save testing tfidf vector representations",
    )
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "ner", "parser"])

    for data_path, lemmatized_path, tfidf_path in zip(
        [args.train_data_path, args.val_data_path, args.test_data_path],
        [
            args.train_lemmatized_path,
            args.val_lemmatized_path,
            args.test_lemmatized_path,
        ],
        [
            args.train_tfidf_path,
            args.val_tfidf_path,
            args.test_tfidf_path,
        ],
    ):
        data_chunks = pd.read_csv(data_path, chunksize=args.chunksize)

        logger.info(f"Loaded data from {data_path} for text preprocessing")

        for data_chunk in data_chunks:
            tokens_list = custom_lemmatized_tokenizer(list(data_chunk.combined_text))

            with open(lemmatized_path, "a") as outfile:
                outfile.write(
                    "\n".join([" ".join(tokens) for tokens in tokens_list]) + "\n"
                )

        logger.info(
            f"Preprocessed data saved to {lemmatized_path} and available for vectorization"
        )

        corpus = make_corpus(lemmatized_path)

        if re.search("train", data_path, flags=re.I):
            vectorizer = TfidfVectorizer(min_df=5)
            tfidf_embeddings = vectorizer.fit_transform(corpus)
        else:
            tfidf_embeddings = vectorizer.transform(corpus)

        save_npz(tfidf_path, tfidf_embeddings)
        logger.info(f"tfidf vectors saved to {tfidf_path}")

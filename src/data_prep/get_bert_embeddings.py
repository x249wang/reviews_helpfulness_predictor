import numpy as np
import pandas as pd
import argparse
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import src.data_prep.config as config
from src.logger import logger


def get_sentence_embeddings(tokenizer, model, text_list):
    result = tokenizer.batch_encode_plus(text_list, padding=True, return_tensors="pt")
    result["input_ids"] = result["input_ids"][:, : config.max_sequence_length]
    result["attention_mask"] = result["attention_mask"][:, : config.max_sequence_length]

    outputs = model(**result)
    token_embeddings = outputs[1][-2]

    sentence_embeddings = torch.stack(
        [
            torch.mean(
                token_embeddings[i, result["attention_mask"][i].nonzero().squeeze(), :],
                dim=0,
            )
            for i in range(len(text_list))
        ],
        0,
    )
    return sentence_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get BERT sentence embeddings")
    parser.add_argument("--chunksize", type=int, default=64, help="Chunk size")
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to save training data"
    )
    parser.add_argument(
        "--train_embeddings_path",
        type=str,
        required=True,
        help="Path to save training embeddings",
    )
    parser.add_argument(
        "--val_data_path", type=str, required=True, help="Path to save validation data"
    )
    parser.add_argument(
        "--val_embeddings_path",
        type=str,
        required=True,
        help="Path to save validation embeddings",
    )
    parser.add_argument(
        "--test_data_path", type=str, required=True, help="Path to save test data"
    )
    parser.add_argument(
        "--test_embeddings_path",
        type=str,
        required=True,
        help="Path to save test embeddings",
    )
    args = parser.parse_args()

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased",
    )
    model = DistilBertModel.from_pretrained(
        "distilbert-base-uncased",
        output_hidden_states=True,
        output_attentions=False,
    )
    model.eval()

    for data_path, embeddings_path in zip(
        [args.train_data_path, args.val_data_path, args.test_data_path],
        [
            args.train_embeddings_path,
            args.val_embeddings_path,
            args.test_embeddings_path,
        ],
    ):
        logger.info(f"Retrieving BERT embeddings for {data_path}")

        data_chunks = pd.read_csv(data_path, chunksize=args.chunksize)

        embeddings = np.empty(shape=[0, config.bert_embeddings_dim])
        with torch.no_grad():
            for data_chunk in data_chunks:
                embeddings_batch = get_sentence_embeddings(
                    tokenizer, model, list(data_chunk.combined_text)
                )
                embeddings = np.vstack((embeddings, embeddings_batch))

        with open(embeddings_path, "wb") as f:
            np.save(f, embeddings)

        logger.info(f"Saved BERT embeddings to {embeddings_path}")

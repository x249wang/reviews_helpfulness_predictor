import re
import numpy as np
import torch
import torch.nn.functional as F
from langdetect import detect
from transformers import DistilBertTokenizer, DistilBertModel
from model import model as classifier

MAX_SEQUENCE_LENGTH = 512
THRESHOLD = 0.5

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased",
)
bert = DistilBertModel.from_pretrained(
    "distilbert-base-uncased",
    output_hidden_states=True,
    output_attentions=False,
)
bert.eval()


# Detects language of the review text
def detect_language(text):
    try:
        language = detect(text)
        return language
    except:
        return "N/A"


# Cleans review text by removing extra whitespace, newline characters and tabs,
# and converting everything to lower case
def clean_text(summary, review_text):
    combined_text = (
        re.sub("\n|\t|\s+", " ", summary.lower().strip())
        + " "
        + re.sub("\n|\t|\s+", " ", review_text.lower().strip())
    )
    combined_text = combined_text.strip()

    if combined_text == "":
        return None

    return combined_text


# Retrieve sentence embeddings for given (cleaned) text
def get_sentence_embeddings(text):
    result = tokenizer.batch_encode_plus([text], return_tensors="pt")
    result["input_ids"] = result["input_ids"][:, :MAX_SEQUENCE_LENGTH]
    result["attention_mask"] = result["attention_mask"][:, :MAX_SEQUENCE_LENGTH]

    outputs = bert(**result)
    token_embeddings = outputs[1][-2]

    return torch.mean(token_embeddings, dim=1)


# Get model predicted label from sentence embeddings
def inference_from_embeddings(embeddings, threshold):
    output = classifier(embeddings)
    predicted_probabilities = F.softmax(output, dim=1).detach().numpy()
    return "HELPFUL" if predicted_probabilities[0, 1] >= threshold else "NOT HELPFUL"


# Wrapper to generate model prediction from (cleaned) text
def generate_prediction_from_text(text):
    embeddings = get_sentence_embeddings(text)
    predicted_label = inference_from_embeddings(embeddings, THRESHOLD)
    return predicted_label

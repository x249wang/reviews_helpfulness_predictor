import numpy as np
import pandas as pd
import random


def convert_to_float(data):
    return data.astype(float)


def convert_to_int(data):
    return data.astype(int)


def is_not_null(data):
    return data.notnull()


def clean_vote_feature(votes):
    if votes.dtype == "O":
        votes_new = votes.str.replace(",", "").astype(float)
    else:
        votes_new = votes
    return votes_new.fillna(0)


def clean_text(review_text_list, summary_list):
    new_review_text_list = (
        review_text_list.fillna("")
        .str.replace("\n|\t|\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    new_summary_list = (
        summary_list.fillna("")
        .str.replace("\n|\t|\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    new_combined_list = (new_review_text_list + " " + new_summary_list).str.strip()

    return new_combined_list.replace("", np.nan)


def detect_language(model, text_list):
    raw_predictions = model.predict(list(text_list))[0]
    language_prediction = [x[0].split("__")[2] for x in raw_predictions]
    return language_prediction


def filter_by_language(language_list, languages_to_keep=["en"]):
    return language_list.isin(languages_to_keep)


def create_label(votes):
    return convert_to_int(votes > 0)

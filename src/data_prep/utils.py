# Helper functions for data cleaning
import numpy as np
import pandas as pd
import random


# Converts a pd.Series into float type
def convert_to_float(data):
    return data.astype(float)


# Converts a pd.Series into integer type
def convert_to_int(data):
    return data.astype(int)


# Checks for null values in a pd.Series
def is_not_null(data):
    return data.notnull()


# Converts string-format votes into numeric type and fill missing values with 0
# Also need to remove the thousand separators before parsing to float
def clean_vote_feature(votes):
    if votes.dtype == "O":
        votes[votes.str.contains(",") == True] = votes[
            votes.str.contains(",") == True
        ].str.replace(",", "")
        votes_new = convert_to_float(votes)
    else:
        votes_new = votes
    return votes_new.fillna(0)


# Cleans review text by removing newline characters, whitespace characters and
# applying lower case, then combine the summary and review content
# If the cleaned review ends up being empty, null is returned
def clean_text(review_text_list, summary_list):
    new_review_text_list = (
        review_text_list.fillna("")
        .str.replace(r"\n|\t|\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    new_summary_list = (
        summary_list.fillna("")
        .str.replace(r"\n|\t|\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )

    new_combined_list = (new_review_text_list + " " + new_summary_list).str.strip()

    return new_combined_list.replace("", np.nan)


# Uses a fasttext model to detect language of the review
def detect_language(model, text_list):
    raw_predictions = model.predict(list(text_list))[0]
    language_prediction = [x[0].split("__")[2] for x in raw_predictions]
    return language_prediction


# Checks whether language detected is part of the list of languages to keep
def filter_by_language(language_list, languages_to_keep=["en"]):
    return language_list.isin(languages_to_keep)


# Create classification label by checking whether a review has received any votes or not
def create_label(votes):
    return convert_to_int(votes > 0)

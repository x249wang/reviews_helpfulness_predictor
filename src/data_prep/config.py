columns_to_keep = [
    "reviewerID",
    "asin",
    "overall",
    "vote",
    "verified",
    "reviewTime",
    "reviewText",
    "summary",
    "unixReviewTime",
    "image",
]

lang_detection_model_path = "src/assets/lid.176.bin"

columns_to_drop = ["image", "reviewText", "summary"]

max_sequence_length = 512
bert_embeddings_dim = 768

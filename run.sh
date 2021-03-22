# Create folders
mkdir -p data
mkdir -p data/raw
mkdir -p data/intermediate
mkdir -p data/final

mkdir -p artifacts
mkdir -p artifacts/naivebayes
mkdir -p artifacts/nn
mkdir -p artifacts/nn/logs
mkdir -p artifacts/nn/models

# Download reviews data
curl http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Clothing_Shoes_and_Jewelry_5.json.gz -o data/raw/Clothing_Shoes_and_Jewelry_5.json.gz
# Source: https://nijianmo.github.io/amazon/index.html

# Download language detection model
curl https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o src/assets/lid.176.bin
# Source: https://towardsdatascience.com/benchmarking-language-detection-for-nlp-8250ea8b67c

# Prep data
## Parse zipped raw data 
python -m src.data_prep.parse_raw_data  --chunksize 100000 \
 --raw_data_path data/raw/Clothing_Shoes_and_Jewelry_5.json.gz --parsed_data_path data/intermediate/parsed_data.csv

## Clean data and create basic features
python -m src.data_prep.clean_data --chunksize 100000 \
  --raw_data_path data/intermediate/parsed_data.csv --cleaned_data_path data/intermediate/cleaned_data.csv

## Split data into train/val/test sets, and save the labels for each set
python -m src.data_prep.split_data \
  --cleaned_data_path data/intermediate/cleaned_data.csv --chunksize 100000 \
  --train_data_path data/intermediate/train_data.csv --train_label_path data/final/train_labels.txt \
  --val_data_path data/intermediate/val_data.csv --val_label_path data/final/val_labels.txt \
  --test_data_path data/intermediate/test_data.csv --test_label_path data/final/test_labels.txt \
  --train_percent 0.25 --val_percent 0.05 --test_percent 0.05 --downsample --downsample_ratio 0.12

## Retrieve tf-idf vector representations from lemmatized review text
python -m src.data_prep.get_tfidf_vectors --chunksize 10000 \
  --train_data_path data/intermediate/train_data.csv --train_lemmatized_path data/intermediate/train_lemmatized_text.txt --train_tfidf_path data/final/train_tfidf.npz \
  --val_data_path data/intermediate/val_data.csv --val_lemmatized_path data/intermediate/val_lemmatized_text.txt --val_tfidf_path data/final/val_tfidf.npz \
  --test_data_path data/intermediate/test_data.csv --test_lemmatized_path data/intermediate/test_lemmatized_text.txt --test_tfidf_path data/final/test_tfidf.npz

## Retrieve embeddings from pretrained DistilBert model 
python -m src.data_prep.get_bert_embeddings --chunksize 64 --train_data_path data/intermediate/train_data.csv --train_embeddings_path data/final/train_bert_embeddings.npy \
  --val_data_path data/intermediate/val_data.csv --val_embeddings_path data/final/val_bert_embeddings.npy \
  --test_data_path data/intermediate/test_data.csv --test_embeddings_path data/final/test_bert_embeddings.npy

# tf-idf Naive Bayes model
## Model development
python -m src.models.naivebayes.train --train_label_path data/final/train_labels.txt --train_tfidf_path data/final/train_tfidf.npz \
  --val_label_path data/final/val_labels.txt --val_tfidf_path data/final/val_tfidf.npz --model_path artifacts/naivebayes/nb_classifier.joblib

## Model evaluation
python -m src.models.naivebayes.evaluate --test_label_path data/final/test_labels.txt --test_tfidf_path data/final/test_tfidf.npz --model_path artifacts/naivebayes/nb_classifier.joblib

# DistilBERT embeddings model
## Model development
python -m src.models.nn.search_hyperparams --train_label_path data/final/train_labels.txt --train_embeddings_path data/final/train_bert_embeddings.npy \
  --val_label_path data/final/val_labels.txt --val_embeddings_path data/final/val_bert_embeddings.npy \
  --model_dir_path artifacts/nn/models --tensorboard_log_dir_path artifacts/nn/logs

## Model evaluation
python -m src.models.nn.evaluate --test_label_path data/final/test_labels.txt --test_embeddings_path data/final/test_bert_embeddings.npy \
  --model_path artifacts/nn/models/f5856df7215a4d89a07bbf787e8ecc5b/model.pth --lr 0.001 --hidden_dim 256 --dropout_rate 0.1 --batch_size 64

# To check results of experiments
# mlflow ui

# To check training loss progression
# tensorboard --logdir=artifacts/nn/logs/f5856df7215a4d89a07bbf787e8ecc5b/

import sys, os, argparse, subprocess

def pip_install(package):
    subprocess.call([sys.executable,"-m", "pip", "install", package])

pip_install("transformers==4.6.1")
pip_install("datasets==1.6.2")

import transformers
import datasets

from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--s3-bucket", type=str)
    parser.add_argument("--s3-prefix", type=str)

    args, _ = parser.parse_known_args()
    print("Received Arguments {}".format(args))

    threshold = args.threshold
    s3_bucket = args.s3_bucket
    s3_prefix = args.s3_prefix

    # Download huggingface dataset
    train_dataset, valid_dataset = load_dataset('generated_reviews_enth', split=['train', 'validation'])
    print('train dataset shape: {}'.format(train_dataset.shape))
    print('validation dataset shape: {}'.format(valid_dataset.shape))

    # Replace review_star rating with labels 0, 1
    def map_review_stars_to_sentiment(row):
        return {'labels': 1 if row['review_star'] >=threshold else 0}

    train_dataset = train_dataset.map(map_review_stars_to_sentiment)
    valid_dataset = valid_dataset.map(map_review_stars_to_sentiment)

    # To manage columns, flatten nested json dataset
    valid_dataset = valid_dataset.flatten()
    train_dataset = train_dataset.flatten()

    # Remove columns not required
    train_dataset = train_dataset.remove_columns(['correct', 'translation.th', 'review_star'])
    valid_dataset = valid_dataset.remove_columns(['correct', 'translation.th', 'review_star'])

    # Rename column
    train_dataset = train_dataset.rename_column('translation.en', 'text')
    valid_dataset = valid_dataset.rename_column('translation.en', 'text')

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
    valid_dataset = valid_dataset.map(tokenize, batched=True, batch_size=len(valid_dataset))

    # remove 'text' column
    train_dataset = train_dataset.remove_columns(['text'])
    valid_dataset = valid_dataset.remove_columns(['text'])


    # Save datasets locally in container which will be automatically copied by sagemaker to s3
    train_dataset.save_to_disk('/opt/ml/processing/output/training/')
    valid_dataset.save_to_disk('/opt/ml/processing/output/validation/')
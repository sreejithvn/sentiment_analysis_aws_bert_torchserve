# Sentiment Analysis on Customer Reviews
This project aims to perform `sentiment analysis` on customer reviews using `transfomer BERT model` with `end-to-end AWS workflow`.

## Introduction
Sentiment analysis is the process of identifying and categorizing subjective information in text data. This project is focused on analyzing customer reviews to determine the overall sentiment of the review, whether it is positive or negative and how to carry out an end-to-end deployment using AWS cloud.

## Preprocessing
The text dataset is preprocessed using SageMaker Processing jobs, using SKLearn docker container, which runs the accompanying preprocessing script. The `preprocessing_hf.py` script reads in the raw customer review data and applies a series of NLP techniques to clean and preprocess the data. The preprocessed data is split into training and validation sets and then stored in Amazon S3 buckets.

## Training
The training is done using Hugging-face's built-in framework in AWS using a HuggingFace Estimator and a training script which runs Hugging-Face's Trainer API code. The `train_hf.py` script sets up the environment for training, loads in the preprocessed datasets from s3, BERT model and it's Tokenizer and finally trains the sentiment analysis model.

## Deployment
The trained model is deployed as a real-time endpoint using both SageMaker's direct deployment using the Estimator, and alternatively using PyTorch's TorchServe. The `torchserve.py` script sets up the environment for deployment, loads in the trained model and the necessary inference code.

## Conclusion
This project demonstrates how to perform sentiment analysis on customer reviews using NLP techniques and Amazon SageMaker. The use of PyTorch and Hugging-Face Estimators enables us to train and deploy the model quickly and easily. By deploying the model as a real-time endpoint, we can process customer reviews in real-time and gain insights into the overall sentiment of the customer base.

# finalCapstone

# Capstone Project: Sentiment Analysis

## Description
This Python program performs sentiment analysis on a dataset of product reviews using the spaCy library. The program implements a sentiment analysis model to classify the sentiment of product reviews as positive, negative, or neutral.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Cleaning Text](#cleaning-text)
- [Sentiment Analysis](#sentiment-analysis)
- [Similarity Comparison](#similarity-comparison)
- [Credits](#credits)

# Usage
- Download the dataset of product reviews: Consumer Reviews of Amazon Products.
- Save the dataset as a CSV file named amazon_product_reviews.csv in the project directory.
- Run the sentiment_analysis.py script:
// python sentiment_analysis.py

# Follow the prompts to perform sentiment analysis on the product reviews.

# Cleaning Text
The program uses spaCy to preprocess the text data.
Stopwords are removed using the .is_stop attribute in spaCy.
Basic text cleaning methods such as .lower(), .strip(), and str() are used.

# Sentiment Analysis
The program analyzes the sentiment of reviews using spaCy.
It calculates the polarity score using the spaCy model.
The polarity score ranges from -1 (very negative) to 1 (very positive).
Reviews are classified as positive, negative, or neutral based on the polarity score.

# Similarity Comparison
The program compares the similarity between two product reviews.
It uses the .similarity() function in spaCy.
A similarity score of 1 indicates high similarity, while 0 indicates low similarity.

# Credits
This project was developed by Rashed

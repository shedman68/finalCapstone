# import spacy
# import pandas as pd
# from pathlib import Path
# import random
# 
# # Load the spaCy model
# print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_sm")
# print("SpaCy model loaded successfully!")
# 
# # Specify the full path to your CSV file
# csv_file_path = Path("C:/Users/rashe/Dropbox/RK23120012424/Data Science (Fundamentals)/T21 - Capstone Project - NLP Applications/amazon_product_reviews2.csv")
# 
# # Read the dataset from the CSV file using your DataFrame name (amazon_df)
# try:
    # print(f"Reading data from file: {csv_file_path}...")
    # amazon_df = pd.read_csv(csv_file_path)
    # print("Data read successfully!")
# except FileNotFoundError:
    # print(f"Error: File '{csv_file_path}' not found. Please provide the correct file path.")
    # exit(1)  # Exit the program if the file is not found
# 
# # Clean the 'reviews.text' column
# print("Cleaning 'reviews.text' column...")
# amazon_df['reviews.text'] = amazon_df['reviews.text'].apply(lambda x: ' '.join([token.text.lower().strip() for token in nlp(x) if not token.is_stop]))
# print("'reviews.text' column cleaned successfully!")
# 
# # Remove rows with empty strings after cleaning
# print("Removing rows with empty strings...")
# amazon_df = amazon_df[amazon_df['reviews.text'].astype(bool)]
# print("Rows with empty strings removed!")
# 
# def analyze_sentiment(review_text, review_date):
    # """
    # Analyzes the sentiment of a product review using spaCy.
    # Args:
        # review_text (str): The review text to analyze.
        # review_date (str): The date of the review.
    # Returns:
        # str: Sentiment label ('positive', 'negative', or 'neutral').
    # """
    # doc = nlp(review_text)
# 
    # # Function to calculate polarity
    # def polarity(doc):
        # # Initialize sentiment scores
        # positive = 0
        # negative = 0
        # total_tokens = 0
# 
        # # Iterate through tokens and calculate sentiment
        # for token in doc:
            # if token.sentiment > 0.5:  # Positive sentiment
                # positive += 1
            # elif token.sentiment < 0.5:  # Negative sentiment
                # negative += 1
            # total_tokens += 1
# 
        # # Calculate polarity score
        # if total_tokens > 0:
            # polarity_score = (positive - negative) / total_tokens
        # else:
            # polarity_score = 0
# 
        # return polarity_score
# 
    # # Set 'polarity' extension attribute to the doc
    # print("Setting 'polarity' extension attribute...")
    # spacy.tokens.Doc.set_extension('polarity', getter=polarity, force=True)
    # print("'polarity' extension attribute set successfully!")
# 
    # # Get the polarity score for the document
    # print("Calculating sentiment polarity...")
    # polarity = doc._.polarity
    # print("Sentiment polarity calculated successfully!")
# 
    # # Determine sentiment based on polarity
    # if polarity > 0:
        # sentiment = "positive"
    # elif polarity < 0:
        # sentiment = "negative"
    # else:
        # sentiment = "neutral"
# 
    # return sentiment, review_date
# 
# # Select two random reviews and their dates
# random_reviews = amazon_df.sample(2)
# for index, row in random_reviews.iterrows():
    # review_text = row['reviews.text']
    # review_date = row['reviews.date']
    # print(f"Analyzing sentiment for review from {review_date}: {review_text}")
    # sentiment, labeled_date = analyze_sentiment(review_text, review_date)
    # print(f"Review Date: {labeled_date}\nReview: {review_text}\nSentiment: {sentiment}\n")
# 
# # Compare similarity between two reviews
# print("Comparing similarity between two random reviews...")
# review1 = random_reviews.iloc[0]['reviews.text']
# review2 = random_reviews.iloc[1]['reviews.text']
# similarity_score = nlp(review1).similarity(nlp(review2))
# print(f"Similarity between reviews: {similarity_score:.2f}")

import spacy
import pandas as pd
from pathlib import Path
import random

# Load the spaCy model
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("SpaCy model loaded successfully!")

# Specify the full path to your CSV file
csv_file_path = Path("C:/Users/rashe/Dropbox/RK23120012424/Data Science (Fundamentals)/T21 - Capstone Project - NLP Applications/amazon_product_reviews_reduced.csv")

# Read the dataset from the CSV file using your DataFrame name (amazon_df)
try:
    print(f"Reading data from file: {csv_file_path}...")
    amazon_df = pd.read_csv(csv_file_path)
    print("Data read successfully!")
except FileNotFoundError:
    print(f"Error: File '{csv_file_path}' not found. Please provide the correct file path.")
    exit(1)  # Exit the program if the file is not found

# Clean the 'reviews.text' column
print("Cleaning 'reviews.text' column...")
def clean_text(text):
    doc = nlp(text)
    cleaned_text = ' '.join([token.text.lower().strip() for token in doc if not token.is_stop])
    return cleaned_text

amazon_df['cleaned_reviews'] = amazon_df['reviews.text'].apply(clean_text)
print("'reviews.text' column cleaned successfully!")

# Remove rows with empty strings after cleaning
print("Removing rows with empty strings...")
amazon_df = amazon_df[amazon_df['cleaned_reviews'].astype(bool)]
print("Rows with empty strings removed!")

def analyze_sentiment(review_text):
    """
    Analyzes the sentiment of a product review using spaCy.
    Args:
        review_text (str): The review text to analyze.
    Returns:
        str: Sentiment label ('positive', 'negative', or 'neutral').
    """
    doc = nlp(review_text)

    # Function to calculate polarity
    def polarity(doc):
        # Initialize sentiment scores
        positive = 0
        negative = 0
        total_tokens = 0

        # Iterate through tokens and calculate sentiment
        for token in doc:
            if token.sentiment > 0.5:  # Positive sentiment
                positive += 1
            elif token.sentiment < 0.5:  # Negative sentiment
                negative += 1
            total_tokens += 1

        # Calculate polarity score
        if total_tokens > 0:
            polarity_score = (positive - negative) / total_tokens
        else:
            polarity_score = 0

        return polarity_score

    # Set 'polarity' extension attribute to the doc
    spacy.tokens.Doc.set_extension('polarity', getter=polarity, force=True)

    # Get the polarity score for the document
    polarity = doc._.polarity

    # Determine sentiment based on polarity
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment

# Get two random reviews and their dates
random_indices = random.sample(range(len(amazon_df)), 2)
random_reviews = amazon_df.iloc[random_indices]['cleaned_reviews']
random_dates = amazon_df.iloc[random_indices]['reviews.date']

for review, date in zip(random_reviews, random_dates):
    print(f"Analyzing sentiment for review from {date}: {review}")
    sentiment = analyze_sentiment(review)
    print(f"Review Date: {date}")
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}\n")

# Compare similarity between two random reviews
print("Comparing similarity between two random reviews...")
random_review1, random_review2 = random_reviews
similarity_score = nlp(random_review1).similarity(nlp(random_review2))
print(f"Similarity between reviews: {similarity_score:.2f}")

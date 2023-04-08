import praw
from praw.models import MoreComments
import os
from datetime import datetime
import spacy
import json
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

## Download lexicons etc.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download('omw-1.4')

## Import Sentiment Intensity Analyzer 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
## Import sentence tokenizer
from nltk.tokenize import sent_tokenize

## Download the VADER dictionary
nltk.download('vader_lexicon')

## Initialize an instance of a sentiment analyzer
vader = SentimentIntensityAnalyzer()

## Ignore the following
client_id = os.getenv("Reddit_Client_Id")
client_secret = os.getenv('Reddit_Client_Secret')
password = os.getenv('Reddit_password')
user_agent = os.getenv('Reddit_User_Agent')
username = os.getenv('Reddit_Username')

def convert_date(date_float : float) -> str:
    """
    Takes a date in epoch time format and converts it into a string in human-readable date format.
    
    Parameters:
    -----------
        date_float (float): a float representing a date in epoch time format.
        
    Returns:
    --------
        (str) : a string representing a date in human-readable format.
    """
    return datetime.fromtimestamp(date_float).strftime('%d-%m-%Y %H:%M:%S')

## Connect to Reddit API
reddit = praw.Reddit(
    client_id=client_id,
    client_secret = client_secret,
    password=password,
    user_agent=user_agent,
    username=username,
    check_for_async=False
)

## Get all submissions from Reddit about multilingualparenting
subreddit = reddit.subreddit('multilingualparenting').top(time_filter="all", limit = None)

## Create a list of dictionaries with the submissions
submissions = [ { 'title' : line.title,
                  'id' : line.id,
                  'upvote_ratio' : line.upvote_ratio,
                  'selftext' : line.selftext,
                  'score' : line.score,
                  'flair' : line.link_flair_text,
                  'num_comments' : line.num_comments,
                  'is_self' : line.is_self,
                  'created' : convert_date(line.created_utc)} 
               for line in subreddit ]

## Create a json lines file with additional key - sentiment
with open('submissions.jl', 'w') as file:
  for submission in submissions:
    sentiment = vader.polarity_scores(submission['selftext'])
    adjusted_compound = sentiment['compound'] * (1 - sentiment['neu'])
    submission['sentiment'] = adjusted_compound
    file.write(json.dumps(submission) + '\n')
    
with open('comments.jl', 'w') as file:
    # Iterate over all the submissions
    for submission in submissions:
        submission_id = submission['id']

        # Fetch the submission using its ID
        submission = reddit.submission(submission_id)

        # Set the option to get all the comments
        submission.comments.replace_more(limit=None)

        # Iterate over all the comments
        for comment in tqdm(submission.comments.list()):
            temp_dict = {}
            temp_dict['body'] = comment.body

            if temp_dict['body'] == '[deleted]':
                continue

            temp_dict['score'] = comment.score
            temp_dict['link'] = comment.permalink

            try:
                temp_dict['author'] = {
                    'name': comment.author.name,
                    'karma': comment.author.comment_karma,
                    'created_utc': convert_date(comment.author.created_utc),
                    'has_verified_email': comment.author.has_verified_email,
                    'is_gold': comment.author.is_gold
                }
            except:
                pass

            temp_dict['created_utc'] = convert_date(comment.created_utc)
            temp_dict['edited'] = comment.edited
            temp_dict['is_submitter'] = comment.is_submitter
            temp_dict['submission_id'] = comment.submission.id

            file.write(json.dumps(temp_dict) + '\n')

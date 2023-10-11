# %%
import praw
from praw.models import MoreComments
import os
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

HERE = Path(__file__).absolute().parent.parent
DATA = HERE/'data'
RAW = DATA/'raw'
XLSX = DATA/'xlsx'
comment_id = 'uhcdh5'
client_id = os.getenv('Reddit_Client_Id')
client_secret = os.getenv('Reddit_Client_Secret')
password = os.getenv('Reddit_password')
user_agent = os.getenv('Reddit_User_Agent')
username = os.getenv('Reddit_Username')

reddit = praw.Reddit(
    client_id = client_id,
    client_secret = client_secret,
    password = password,
    user_agent = user_agent,
    username = username
)

# %%
def convert_date(date_float: float) -> str:
    """
    Takes a date in epoch time format and converts it into a string in
    human-readable date format.

    Args:
        date_float (float): a float representing a date in epoch time format.

    Returns:
        str: a string representing a date in human-redeable format.
    """
    return datetime.fromtimestamp(date_float).strftime('%d-%m-%Y %H:%M:%S')

def get_data(comment_id: str, path: str = RAW) -> None:
    """
    It downloads comment from a given submission/comment and 
    writes it out to JSON line file at path.

    Args:
        comment_id (str): a string with a submission/comment id from
        which you want to download comments.
        path (str): a path to the data folder.
    """
    submission = reddit.submission(comment_id)
    submission.comments.replace_more(limit = None)
    path = path/(comment_id + '_comments.jsonl')
    print(f'Processing submission {submission.title}.')
    with open(path, 'w') as file:
        for comment in tqdm(submission.comments.list()):
            temp_dict = {}
            temp_dict['body'] = comment.body
            if temp_dict['body'] == '[deleted]':
                continue
            temp_dict['score'] = comment.score
            temp_dict['link'] = comment.permalink
            try:
                temp_dict['author'] = { 'name' : comment.author.name,
                                      'karma' : comment.author.comment_karma,
                                      'created_utc' : convert_date(comment.author.created_utc),
                                      'has_verified_email' : comment.author.has_verified_email,
                                      'is_gold' : comment.author.is_gold
                }
            except:
                pass
            temp_dict['created_utc'] = convert_date(comment.created_utc)
            temp_dict['edited'] = comment.edited
            temp_dict['is_submitter'] = comment.is_submitter
          
            file.write(json.dumps(temp_dict) + '\n')
            
def convert_to_excel(comment_id: str, input_path: str = RAW, output_path: str = XLSX) -> None:
    """
    Converts a JSON line file into an Excel spreadsheet.

    Args:
        input_path (str): a path to a raw data folder.
        output_path (str): a path to a processed data folder.
        comment_id (str): name of the file to process without extension.
    """
    input_path = input_path/(comment_id + '_comments.jsonl')
    output_path = output_path/(comment_id + '_comments.xlsx')
    with open(input_path, 'r') as file:
        temp = [ json.loads(line) for line in file.readlines() ]
    pd.DataFrame.from_records(temp).to_excel(output_path)
      
# %%
if __name__ == "__main__":
    for comment_id in sys.argv[1:]:
        get_data(comment_id = comment_id, path = RAW)
        convert_to_excel(input_path=RAW, output_path = XLSX, comment_id = comment_id)

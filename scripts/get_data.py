## Load modules
import requests as rq
import json
import time
import sys
import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")
vader = SentimentIntensityAnalyzer()

## Define custom functions which will make our life easier


def parse_date(date, format="human"):
    """ "
    It takes a string and converts it into either human readable date format or epoch date format

    Parameteres:
    ============
    date: str
        A string with either epoch date format or human readable date format.
    format: str
        A string defining the format of the input string. By default, it takes the value 'human' and the other option is 'epoch'.
    """
    if format == "human":
        pattern = "%Y-%m-%d %H:%M:%S"
        return str(int(time.mktime(time.strptime(date, pattern))))
    elif format == "epoch":
        pattern = "%Y-%m-%d %H:%M:%S"
        return time.strftime(pattern, time.localtime(int(date)))


def collect_data(source_url, payload):
    """
    It takes the Pushift endpoint and payload as arguments and sends a request to the given URL. Depending on the status code it either returns
    the list of mappings, a status code, or sleeps for 60 seconds and tries again to scrape the data.

    Parameters:
    ===========
    source_url: str
        A string with url of the Pushshift endpoint.
    payload:
        A mapping with parameters passed to the Pushshift API.
    """
    if "after" not in payload:
        payload["after"] = parse_date("2005-06-23 00:00:00")
    response = rq.get(source_url, params=payload)
    if response.status_code == 200:
        return json.loads(response.text)["data"]
    elif (
        response.status_code == 429
        or response.status_code == 523
        or response.status_code == 502
    ):
        for i in range(60, 0, -1):
            sys.stdout.write(f"The compulasatory break finishes in {str(i)} seconds\r")
            sys.stdout.flush()
            time.sleep(1)
        sys.stdout.write(100 * " " + "\r")
        return collect_data(source_url=source_url, payload=payload)
    else:
        return [{"status": response.status_code, "message": response.content}]


## Define the parameters and options
source_url = "https://api.pushshift.io/reddit/search/comment/"
payload = {"subreddit": "urbanplanning", "size": 100}

## Collect first batch of data
data_comments = collect_data(source_url=source_url, payload=payload)

## Open a file in write mode
with open("comments.jl", "w") as file:
    ## Write out the data you already collected
    for line in data_comments:
        temp = vader.polarity_scores(line["body"])
        line["pos"] = temp["pos"]
        line["neg"] = temp["neg"]
        line["neu"] = temp["neu"]
        line["compound"] = temp["compound"]
        line["created_utc"] = parse_date(line["created_utc"], format="epoch")
        file.write(json.dumps(line) + "\n")
    ## Set the prpogress bar
    pbar = tqdm.tqdm(position=0, leave=True, initial=100)
    ## Create a while-loop
    while len(data_comments) > 0:
        ## Check if we got data from Reddit or a strange status code
        if len(data_comments[0].keys()) > 2:
            ## Get the last collected data date in epoch time
            after = parse_date(data_comments[-1]["created_utc"])
            ## Update the payload after field
            payload["after"] = after
            ## Collect the data
            data_comments = collect_data(source_url=source_url, payload=payload)
            ## Write out the collected data to the file
            for line in data_comments:
                if "created_utc" in line:
                    ## Compute sentiment of the text
                    temp = vader.polarity_scores(line["body"])
                    line["pos"] = temp["pos"]
                    line["neg"] = temp["neg"]
                    line["neu"] = temp["neu"]
                    line["compound"] = temp["compound"]
                    line["created_utc"] = parse_date(
                        line["created_utc"], format="epoch"
                    )
                    file.write(json.dumps(line) + "\n")
            ## Update the progress bar
            pbar.update(len(data_comments))
        else:
            ## Print out the strange status code and its message
            print(
                f"Something went wrong. The status code error was {data_comments.pop}."
            )

## Define the parameters and options
source_url = "https://api.pushshift.io/reddit/search/submission/"
payload = {"subreddit": "urbanplanning", "size": 100}

## Collect first batch of data
data = collect_data(source_url=source_url, payload=payload)

## Open a file in write mode
with open("submissions.jl", "w") as file:
    ## Write out the data you already collected
    for line in data:
        temp = vader.polarity_scores(line["title"])
        line["pos"] = temp["pos"]
        line["neg"] = temp["neg"]
        line["neu"] = temp["neu"]
        line["compound"] = temp["compound"]
        if (
            "selftext" in line
            and len(line["selftext"]) > 0
            and "[deleted]" not in line["selftext"]
        ):
            temp = vader.polarity_scores(line["selftext"])
            line["pos_selftext"] = temp["pos"]
            line["neg_selftext"] = temp["neg"]
            line["neu_selftext"] = temp["neu"]
            line["compound_selftext"] = temp["compound"]
        line["created_utc"] = parse_date(line["created_utc"], format="epoch")
        file.write(json.dumps(line) + "\n")
    ## Set the prpogress bar
    pbar = tqdm.tqdm(position=0, leave=True, initial=100)
    ## Create a while-loop
    while len(data) > 0:
        ## Check if we got data from Reddit or a strange status code
        if len(data[0].keys()) > 2:
            ## Get the last collected data date in epoch time
            after = parse_date(data[-1]["created_utc"])
            ## Update the payload after field
            payload["after"] = after
            ## Collect the data
            data = collect_data(source_url=source_url, payload=payload)
            ## Write out the collected data to the file
            for line in data:
                if "created_utc" in line:
                    ## Compute sentiment of the text
                    temp = vader.polarity_scores(line["title"])
                    line["pos"] = temp["pos"]
                    line["neg"] = temp["neg"]
                    line["neu"] = temp["neu"]
                    line["compound"] = temp["compound"]
                    if (
                        "selftext" in line
                        and len(line["selftext"]) > 0
                        and "[deleted]" not in line["selftext"]
                    ):
                        temp = vader.polarity_scores(line["selftext"])
                        line["pos_selftext"] = temp["pos"]
                        line["neg_selftext"] = temp["neg"]
                        line["neu_selftext"] = temp["neu"]
                        line["compound_selftext"] = temp["compound"]
                    line["created_utc"] = parse_date(
                        line["created_utc"], format="epoch"
                    )
                    file.write(json.dumps(line) + "\n")
            ## Update the progress bar
            pbar.update(len(data))
        else:
            ## Print out the strange status code and its message
            print(f"Something went wrong. The status code error was {data.pop}.")

# %%
from transformers import pipeline
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from nltk.tokenize import sent_tokenize

# %%
## Load the Pre-trained Multilingual Sentiment Classifier
sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual",
    top_k=None,
    truncation=True,
    max_length=512,
)

sentiment_pl = pipeline(
    "sentiment-analysis",
    model="nie3e/sentiment-polish-gpt2-small",
    top_k=None,
    truncation=True,
    max_length=512,
)

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"
data = pd.read_csv(PROC / "food_texts.csv", index_col=0)


# %%
## Define functions
def split_text(text: str, country: str = "uk", max_length: int = 512) -> list:
    """
    Splits the input text into chunks of a specified maximum length.

    Parameters:
    text (str): The input text to be split.
    country (str): The country code to determine the sentence tokenizer to use.
    max_length (int): The maximum length of each chunk.

    Returns:
    list: A list of text chunks.
    """
    language_map = {"uk": "English", "poland": "Polish", "portugal": "Portuguese"}
    language = language_map.get(country.lower(), "English")
    sentences = sent_tokenize(text, language=language)
    chunks = []
    current_chunk = []

    for sent in sentences:
        if len(" ".join(current_chunk + [sent])) <= max_length:
            current_chunk.append(sent)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def compute_sentiment(text: str, country: str, sentiment_pipeline) -> list:
    """
    Computes the sentiment for each chunk of text using the specified sentiment analysis pipeline.

    Parameters:
    chunks (str): The input text to compute sentiment.
    country (str): The country code to determine the language for splitting the text.
    sentiment_pipeline: The sentiment analysis pipeline to use.

    Returns:
    list: A list of sentiment results for each chunk.
    """
    chunks = split_text(text, country=country)
    sentiments = []
    for chunk in chunks:
        if not chunk.strip():
            continue
        result = sentiment_pipeline(chunk)[0]
        sentiments.append(result)
    return sentiments


# %%
def main():
    with open(PROC / "food_texts_sentiment.jsonl", "w") as file:
        for _, item in tqdm(data.iterrows()):
            if item["country"] == "poland":
                result = compute_sentiment(
                    item.body, country="poland", sentiment_pipeline=sentiment_pl
                )
            else:
                result = compute_sentiment(
                    item.body, country="uk", sentiment_pipeline=sentiment
                )

            tmp = {"id": _, "sentiment": result}
            file.write(json.dumps(tmp) + "\n")


# %%
if __name__ == "__main__":
    main()


# %%

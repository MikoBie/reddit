# %%
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import json

# %%
## Load the native models (~110M params)
model_en = SentenceTransformer("bert-base-uncased")
model_pt = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
model_pl = SentenceTransformer("dkleczek/bert-base-polish-cased-v1")
models = {"uk": model_en, "portugal": model_pt, "poland": model_pl}
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


def encode_text(text: str, country: str, model) -> np.ndarray:
    """Computes the embedding for a given text by splitting it into chunks, encoding each chunk, and averaging the embeddings.

    Parameters:
    text (str): The input text to encode.
    country (str): The country code to determine the language for splitting the text.
    model: The sentence transformer model to use for encoding.

    Returns:
    np.ndarray: The averaged embedding vector for the input text.
    """
    chunks = split_text(text, country=country)
    chunk_embeddings = model.encode(chunks)
    doc_vector = np.mean(chunk_embeddings, axis=0)
    return doc_vector


def main():
    with open(PROC / "food_texts_embeddings.jsonl", "w") as file:
        for _, item in tqdm(
            data.iterrows(), total=data.shape[0], desc="Computing embeddings"
        ):
            for model in models:
                if item["country"] == model:
                    embeddings = encode_text(
                        item["body"], country=model, model=models[model]
                    )
                    tmp = {"id": _, "embeddings": embeddings.tolist()}
                    file.write(json.dumps(tmp) + "\n")


# %%
if __name__ == "__main__":
    main()

# %%
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %%
## Define globals
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
PROC = DATA / "processed"

map_sentiment = {"positive": 1, "negative": -1}

# %%
## Load data
df = pd.read_csv(PROC / "food_texts.csv", index_col=None)
sentiment_lst = []
for line in open(PROC / "food_texts_sentiment.jsonl", "r"):
    tmp = json.loads(line)
    sentiment = [max(item, key=lambda x: x["score"]) for item in tmp["sentiment"]]
    sentiment = [
        map_sentiment.get(item["label"].lower(), 0) * item["score"]
        for item in sentiment
    ]
    sentiment = np.mean(sentiment)
    sentiment_lst.append({"id": tmp["id"], "sentiment": sentiment})

df_sentiment = pd.DataFrame.from_dict(sentiment_lst)
df = pd.merge(df, df_sentiment, on="id").melt(
    id_vars=["id", "body", "country", "sentiment"]
)
# %%
poland = df.query("country == 'poland'").reset_index(drop=True)
poland.query("value > 0").groupby("variable")["sentiment"].mean()

lst = [item for item in set(poland["variable"]) if "mot" in item]
for item in lst:
    sent = poland.query("value > 0 and variable == @item")["sentiment"].tolist()
    xbins = np.array([i * 0.1 for i in range(-9, 11, 1)])
    plt.hist(
        sent, bins=xbins, label=item, alpha=0.5, density=True, histtype="barstacked"
    )

plt.legend()
plt.xlabel("Sentiment")
plt.ylabel("Density")
plt.show()

# %%
portugal = df.query("country == 'portugal'").reset_index(drop=True)
portugal.query("value > 0").groupby("variable")["sentiment"].mean()

lst = [item for item in set(portugal["variable"]) if "mot" in item]
for item in lst:
    sent = portugal.query("value > 0 and variable == @item")["sentiment"].tolist()
    xbins = np.array([i * 0.1 for i in range(-9, 11, 1)])
    plt.hist(
        sent, bins=xbins, label=item, alpha=0.5, density=True, histtype="barstacked"
    )

plt.legend()
plt.xlabel("Sentiment")
plt.ylabel("Density")
plt.show()

# %%
uk = df.query("country == 'uk'").reset_index(drop=True)
uk.query("value > 0").groupby("variable")["sentiment"].mean()

lst = [item for item in set(uk["variable"]) if "mot" in item]
for item in lst:
    sent = uk.query("value > 0 and variable == @item")["sentiment"].tolist()
    xbins = np.array([i * 0.1 for i in range(-9, 11, 1)])
    plot = plt.hist(
        sent, bins=xbins, label=item, alpha=0.5, density=True, histtype="barstacked"
    )
plt.legend()
plt.xlabel("Sentiment")
plt.ylabel("Density")
plt.show()
# %%
